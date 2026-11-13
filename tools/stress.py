import argparse
from pathlib import Path
from random import Random
import sys
from tempfile import TemporaryDirectory

from _reference import DATA_WORDS, Interpreter, difference
from _suite import (CheckError, DEFAULT_MACHINE, DEFAULT_OPTIMIZER, load_kit,
                    optimize, positive, report_error, run)


STRESS_OPS = ("add", "sub", "mul", "xor", "and", "or", "shl", "shr", "lt", "eq")


def large_program(count, seed):
    rng = Random(seed)
    yield "func long_input(v0, v1, v2, v3, v4, v5, v6, v7)"
    yield "entry:"
    for index in range(count - 1):
        target = index % 64
        if index % 127 == 0:
            yield "  store {}, v{}".format(index % DATA_WORDS, (target + 9) % 64)
        elif index % 127 == 1:
            yield "  load v{}, {}".format(target, (index - 1) % DATA_WORDS)
        elif index % 113 == 0:
            yield "  vstore {}, v{}".format(target % 4, target)
        else:
            op = rng.choice(STRESS_OPS)
            a = rng.randrange(64)
            b = rng.randrange(32) if rng.randrange(2) else "v{}".format(rng.randrange(64))
            yield "  {} v{}, v{}, {}".format(op, target, a, b)
    yield "  ret v{}".format((count - 2) % 64)


def many_blocks(count):
    yield "func wide_cfg(v0)"
    yield "entry:"
    yield "  jmp branch0"
    for index in range(count):
        yield "branch{}:".format(index)
        yield "  add v1, v1, {}".format(index % 17)
        yield "  br v0, branch{}, skip{}".format(index + 1, index)
        yield "skip{}:".format(index)
        yield "  jmp branch{}".format(index + 1)
    yield "branch{}:".format(count)
    yield "  ret v1"
    yield "unreachable:"
    yield "  trap 17"


CONTROL = """func control(v0, v1)
entry:
  and v2, v1, 31
  br v0, left, right
left:
  sub v2, v2, 1
  lt v3, v2, 0
  br v3, done, right
right:
  add v4, v4, 3
  br v0, left, spin
spin:
  sub v2, v2, 1
  lt v3, v2, 0
  br v3, done, spin
done:
  store 3071, v4
  store 3072, v2
  vload v6, v1
  vstore v1, v6
  load v5, v0
  ret v5
unreachable:
  jmp unreachable
"""

MACHINE_PROGRAM = """func machine_probe(v0, v1, v2)
entry:
  mul v3, v0, 7
  add v4, v3, v1
  shl v5, v2, 3
  add v6, v4, v5
  load v7, 3071
  br v0, branch, exit
branch:
  add v6, v6, v7
  jmp exit
exit:
  store 1, v6
  ret v6
"""

MISMATCHED_PROFILE = """edge entry branch 0
edge entry exit 10000000
edge branch exit 0
"""

MACHINE_VARIANTS = (
    ("width1", {"issue_width": 1, "mem_ports": 1, "registers": 4}),
    ("width4", {"issue_width": 4, "mem_ports": 2, "registers": 64}),
    ("scarce", {"registers": 4, "reg_spill_penalty": 500, "icache_size": 1,
                "icache_penalty": 100, "taken_branch_penalty": 1000}),
    ("free_penalties", {"reg_spill_penalty": 0, "taken_branch_penalty": 0,
                        "icache_penalty": 0, "icache_size": 500000}),
    ("slow_ops", {"lat_alu": 3, "lat_mul": 100, "lat_div": 300, "lat_load": 70,
                  "lat_mac": 220, "lat_addsh": 220}),
    ("no_mac", {"fused_mac": 0, "fused_addsh": 1}),
    ("no_addsh", {"fused_mac": 1, "fused_addsh": 0}),
    ("no_fused", {"fused_mac": 0, "fused_addsh": 0}),
)


def write_lines(path, lines):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for line in lines:
            stream.write(line + "\n")


def validate_case(options, module, program, machine_path, parameters, timeout, profile=None):
    out = program.parent / "out.kir"
    if out.exists():
        out.unlink()
    existing = {path.name for path in program.parent.iterdir()}
    elapsed = optimize(options.kopt, program, machine_path, out, profile, timeout=timeout)
    if {path.name for path in program.parent.iterdir()} != existing | {out.name}:
        raise CheckError("optimizer created unexpected files")
    source = module.load_program(str(program))
    candidate = module.load_program(str(out))
    machine = module.load_machine(str(machine_path))
    module.check_machine_features(source, machine)
    module.check_machine_features(candidate, machine)
    if len(candidate.params) != len(source.params):
        raise CheckError("parameter count changed")
    left = Interpreter(source)
    right = Interpreter(candidate)
    for args in parameters:
        sample = module.Inputs(list(args), [(3071, [731, 17])], {1: [4, 7]})
        original = left.execute(sample, 1000000)
        if original.status == "TIMEOUT":
            raise CheckError("stress fixture did not finish")
        optimized = right.execute(sample, 20 * original.steps + 1000)
        mismatch = difference(original, optimized)
        if mismatch:
            raise CheckError("{}: {}".format(args, mismatch))
    print("{} / {}: {:.3f} s, {} -> {} instructions, {} matching inputs".format(
        program.stem, machine_path.stem, elapsed, len(source.instrs), len(candidate.instrs),
        len(parameters)), flush=True)


def invalid_case(options, directory, machine_path):
    source = directory / "empty_block.kir"
    output = directory / "out.kir"
    source.write_bytes(b"func invalid()\nentry:\nempty:\n  ret 0\n")
    output.write_bytes(b"preserve this file\n")
    try:
        run(options.kopt, source, "--machine", machine_path, "-o", output,
            cwd=directory, timeout=5)
    except CheckError as exc:
        if "exceeded" in str(exc):
            raise
    else:
        raise CheckError("accepted a structurally empty block")
    if output.read_bytes() != b"preserve this file\n":
        raise CheckError("overwrote output for an invalid input")
    print("Empty block rejected without changing the output file.")


def main():
    parser = argparse.ArgumentParser(description="Exercise large programs, unusual CFGs and machine settings.")
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--kopt", type=Path, default=DEFAULT_OPTIMIZER)
    parser.add_argument("--sizes", nargs="+", type=positive, default=(100000, 500000))
    parser.add_argument("--blocks", type=positive, default=4000)
    parser.add_argument("--seed", type=int, default=71237)
    options = parser.parse_args()
    options.kit = options.kit.resolve()
    options.kopt = options.kopt.resolve()
    if any(size < 2 or size > 500000 for size in options.sizes):
        parser.error("each size must be between 2 and 500000")
    if options.blocks > 166665:
        parser.error("too many blocks for the instruction limit")

    try:
        module = load_kit(options.kit)
    except (OSError, ImportError) as exc:
        return report_error(exc)
    machine_path = options.kit / "machines" / DEFAULT_MACHINE
    try:
        config = module.load_machine(str(machine_path))
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for count in options.sizes:
                source = directory / ("wide_{}.kir".format(count))
                write_lines(source, large_program(count, options.seed))
                validate_case(options, module, source, machine_path,
                              [(17, 0, 31, 32, 4095, 4096, 2147483648, 4294967295)],
                              30 if count >= 100000 else 5)
                source.unlink()

            source = directory / "many_blocks.kir"
            write_lines(source, many_blocks(options.blocks))
            validate_case(options, module, source, machine_path,
                          [(0,), (1,)], 30 if 3 * options.blocks >= 100000 else 5)
            source.unlink()

            source = directory / "control.kir"
            source.write_bytes(CONTROL.encode("utf-8"))
            validate_case(options, module, source, machine_path,
                          [(0, 1), (1, 1), (4096, 5), (0, 31)], 5)

            source = directory / "machine_probe.kir"
            source.write_bytes(MACHINE_PROGRAM.encode("utf-8"))
            profile = directory / "wrong.prof"
            profile.write_bytes(MISMATCHED_PROFILE.encode("utf-8"))
            for label, overrides in MACHINE_VARIANTS:
                machine = directory / (label + ".cfg")
                values = dict(config)
                values.update(overrides)
                write_lines(machine, ("{} = {}".format(key, values[key])
                                      for key in sorted(values)))
                validate_case(options, module, source, machine,
                              [(0, 10, 3), (1, 5, 7), (2147483648, 4096, 4294967295)],
                              5, profile=profile)
            invalid_case(options, directory, machine_path)
    except (CheckError, OSError, ValueError, module.KirError) as exc:
        return report_error(exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
