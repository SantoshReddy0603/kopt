from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from random import Random
import re
import shutil
import sys
from tempfile import TemporaryDirectory

from _reference import DATA_WORDS, Interpreter, Result, WORD_MASK, difference
from _suite import (CheckError, DEFAULT_MACHINE, cases, fields, load_kit, nonnegative,
                    optimize, positive, profile_for, report_error, run, simulate, suite_parser)


EDGES = (0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64, 255, 256, 1024,
         2047, 3071, 3072, 4094, 4095, 4096, 4097, 65535, (1 << 31) - 1,
         1 << 31, WORD_MASK - 1, WORD_MASK)
EVENT = re.compile(r"^vlog\[\d+\]: (vload|vstore) port=(\d+) value=(\d+)$")
MEMORY = re.compile(r"^mem\[(\d+)\] = (\d+)$")


def input_sets(module, program, sample, count, seed):
    yield sample
    params = len(program.params)
    ports = {0, 1, 2, 3, 4, 5, WORD_MASK} | set(sample.vin)
    addresses = {0, 255, 256, 1023, 3070, 3071, 3072, 3073, 4094, 4095}
    for instruction in program.instrs:
        if instruction.op in ("load", "store", "vload", "vstore"):
            operand = instruction.args[int(instruction.op in ("load", "vload"))]
            if operand[0] == "i":
                if instruction.op.startswith("v"):
                    ports.add(operand[1])
                elif operand[1] < 4096:
                    addresses.add(operand[1])
    ports = sorted(ports)
    addresses = sorted(addresses)

    for value in EDGES:
        yield module.Inputs([value] * params, sample.segments, sample.vin)
        for index in range(params):
            args = list(sample.args)
            args[index] = value
            yield module.Inputs(args, sample.segments, sample.vin)

    if params >= 2:
        for a, b in ((1 << 31, WORD_MASK), (WORD_MASK, 2), (7, WORD_MASK - 1),
                     (1, 0), (0, 4096), (4095, 4096), (3071, 3072)):
            args = list(sample.args)
            args[0], args[1] = a, b
            yield module.Inputs(args, sample.segments, sample.vin)

    rng = Random(seed)
    for index in range(count):
        args = []
        for position in range(params):
            if index % 4 == 0:
                args.append(rng.getrandbits(32))
            elif index % 4 == 1:
                args.append(rng.choice(EDGES))
            elif index % 4 == 2:
                args.append(rng.randrange(4100))
            else:
                args.append(sample.args[position] if rng.randrange(2) else rng.choice(EDGES))
        memory = list(sample.segments)
        memory.append((0, [rng.getrandbits(32) for _ in range(32)]))
        memory.extend((address, [rng.choice(EDGES)]) for address in rng.sample(
            addresses, min(16, len(addresses))))
        selected_ports = rng.sample(ports, min(16, len(ports)))
        queues = {port: [rng.getrandbits(32) for _ in range(rng.randrange(1, 17))]
                  for port in sorted(selected_ports)}
        yield module.Inputs(args, memory, queues)


def simulator_result(output):
    metadata = fields(output)
    status = metadata["status"]
    if status.startswith("TRAP "):
        status, value = status.split(" ", 1)
    else:
        value = int(metadata["ret"]) if status == "RET" else None
    events = []
    memory = [0] * DATA_WORDS
    for line in output.splitlines():
        event = EVENT.fullmatch(line)
        if event:
            op, port, word = event.groups()
            events.append(("L" if op == "vload" else "S", int(port), int(word)))
        word = MEMORY.fullmatch(line)
        if word:
            address, contents = map(int, word.groups())
            memory[address] = contents
    if len(events) != int(metadata["vlog_entries"]):
        raise CheckError("could not read all volatile events from ksim")
    return Result(status, value, events, memory, int(metadata["steps"]))


def read_sample(module, program, path):
    sample = path.with_suffix(".in")
    if sample.is_file():
        return module.load_input(str(sample), len(program.params))
    return module.Inputs([0] * len(program.params), [], {})


def cross_check(options, program_path):
    module = load_kit(options.kit)
    try:
        program = module.load_program(str(program_path))
        sample = read_sample(module, program, program_path)
        machine = options.kit / "machines" / DEFAULT_MACHINE
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / program_path.name
            shutil.copyfile(program_path, source)
            run(options.kit / "kvalidate", source, "--machine", machine)
            official = simulator_result(simulate(options.kit, source, machine,
                                                 program_path.with_suffix(".in"), verbose=True,
                                                 max_steps=options.max_steps))
        independent = Interpreter(program).execute(sample, options.max_steps)
        mismatch = difference(official, independent)
        if mismatch:
            raise CheckError(mismatch)
        if official.steps != independent.steps:
            raise CheckError("executed steps differ: {} vs {}".format(official.steps,
                                                                     independent.steps))
    except (CheckError, OSError, KeyError, ValueError, module.KirError) as exc:
        return "{}: {}".format(program_path.stem, exc)
    return None


def check_program(options, machines, source):
    module = load_kit(options.kit)
    errors = []
    tested = 0
    skipped = 0
    try:
        original = module.load_program(str(source))
        sample = read_sample(module, original, source)
        interpreter = Interpreter(original)
    except (CheckError, OSError, ValueError, module.KirError) as exc:
        return tested, skipped, ["{}: {}".format(source.stem, exc)]
    candidates = {}
    for machine_path in machines:
        try:
            with TemporaryDirectory() as temporary:
                output = Path(temporary) / "out.kir"
                optimize(options.kopt, source, machine_path, output,
                         profile_for(source), timeout=5)
                candidate = module.load_program(str(output))
            machine = module.load_machine(str(machine_path))
            module.check_machine_features(original, machine)
            module.check_machine_features(candidate, machine)
            if len(original.params) != len(candidate.params):
                raise CheckError("parameter count changed")
            candidates[machine_path] = Interpreter(candidate)
        except (CheckError, OSError, ValueError, module.KirError) as exc:
            errors.append("{} / {}: {}".format(source.stem, machine_path.stem, exc))
    if not candidates:
        return tested, skipped, errors
    for index, inp in enumerate(input_sets(module, original, sample, options.inputs, options.seed)):
        left = interpreter.execute(inp, options.max_steps)
        if left.status == "TIMEOUT":
            skipped += len(candidates)
            continue
        for machine_path, candidate in list(candidates.items()):
            right = candidate.execute(inp, 20 * left.steps + 1000)
            tested += 1
            mismatch = difference(left, right)
            if mismatch is not None:
                errors.append("{} / {}: input {}: {}\n{}".format(
                    source.stem, machine_path.stem, index, mismatch, module.format_input(inp)))
                del candidates[machine_path]
        if not candidates:
            break
    if not tested:
        errors.extend("{} / {}: all original executions timed out".format(
            source.stem, machine_path.stem) for machine_path in candidates)
    return tested, skipped, errors


def main():
    parser = suite_parser("Compare observable results using a separate KIR interpreter.")
    parser.add_argument("--inputs", type=nonnegative, default=200,
                        help="random inputs per program, in addition to deterministic edge cases")
    parser.add_argument("--seed", type=int, default=21981)
    parser.add_argument("--max-steps", type=positive, default=200000)
    parser.add_argument("--cross-check-only", action="store_true")
    options = parser.parse_args()
    try:
        programs, machines = cases(options, require_optimizer=not options.cross_check_only)
    except CheckError as exc:
        return report_error(exc)
    with ThreadPoolExecutor(max_workers=options.jobs) as executor:
        errors = [error for error in executor.map(partial(cross_check, options), programs) if error]
    print("Interpreter agrees with ksim: {} of {} samples".format(
        len(programs) - len(errors), len(programs)), flush=True)
    if errors:
        return report_error(CheckError("\n".join(errors)))
    if options.cross_check_only:
        return 0
    checked = 0
    skipped = 0
    if options.jobs == 1:
        results = map(partial(check_program, options, machines), programs)
        for tested, timed_out, failures in results:
            checked += tested
            skipped += timed_out
            errors.extend(failures)
    else:
        with ProcessPoolExecutor(max_workers=options.jobs) as executor:
            for tested, timed_out, failures in executor.map(
                    partial(check_program, options, machines), programs):
                checked += tested
                skipped += timed_out
                errors.extend(failures)
    for error in errors:
        print(error, file=sys.stderr)
    print("Compared {} executions over {} program and machine combinations.".format(
        checked, len(programs) * len(machines)))
    print("Skipped original timeouts: {}. Failed combinations: {}.".format(skipped, len(errors)))
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
