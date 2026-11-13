"""Randomized checks shaped like the hidden judging cases.

Each case is a program, a machine and sometimes a profile. Programs are either
practice programs with renamed registers, shuffled blocks and changed literals,
or generated programs with traps, memory effects, volatile operations and
loops. Machines are random within the file format limits. Every case checks
that kopt exits in time, writes a valid file, is byte-identical on a second run
and behaves like the original on edge, random and trapping inputs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from random import Random
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

from _suite import DEFAULT_OPTIMIZER, load_kit, positive
from difftest import input_sets

REGISTER = re.compile(r"\bv(\d+)\b")
LABEL = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*$")
NUMBER = re.compile(r"(?<![\w])(-?(?:0x[0-9A-Fa-f]+|\d+))(?![\w])")
EDGE_LITERALS = (0, 1, 2, 3, 7, 31, 32, 33, 4095, 4096, 3071, 3072, 0x7FFFFFFF,
                 0x80000000, 0xFFFFFFFF, -1, -2)
MACHINE_KEYS = ("issue_width", "mem_ports", "registers", "reg_spill_penalty",
                "taken_branch_penalty", "icache_size", "icache_penalty", "lat_alu",
                "lat_mul", "lat_div", "lat_load", "lat_mac", "lat_addsh",
                "fused_mac", "fused_addsh")


def random_machine(rng):
    pick = rng.choice
    machine = {
        "issue_width": pick((1, 1, 2, 2, 3, 4, 8)),
        "mem_ports": pick((1, 1, 2, 4)),
        "registers": pick((4, 4, 5, 6, 8, 8, 12, 16, 32, 64)),
        "reg_spill_penalty": pick((0, 1, 3, 5, 8, 20)),
        "taken_branch_penalty": pick((0, 1, 2, 2, 4, 10)),
        "icache_size": pick((1, 8, 16, 32, 64, 64, 128, 256, 4096)),
        "icache_penalty": pick((0, 1, 2, 2, 5, 10)),
        "lat_alu": pick((1, 1, 1, 2, 3)),
        "lat_mul": pick((1, 2, 3, 3, 5, 10)),
        "lat_div": pick((1, 4, 12, 12, 30)),
        "lat_load": pick((1, 2, 4, 4, 8, 20)),
        "lat_mac": pick((1, 2, 3, 3, 6, 12)),
        "lat_addsh": pick((1, 1, 2, 4)),
        "fused_mac": pick((0, 1, 1)),
        "fused_addsh": pick((0, 1, 1)),
    }
    if machine["mem_ports"] > machine["issue_width"]:
        machine["mem_ports"] = machine["issue_width"]
    return machine


def machine_text(machine, rng):
    keys = list(MACHINE_KEYS)
    rng.shuffle(keys)
    lines = ["# generated machine"]
    for key in keys:
        lines.append("%s = %d   # value" % (key, machine[key]))
    return "\n".join(lines) + "\n"


def literal(rng):
    if rng.random() < 0.3:
        return str(rng.choice(EDGE_LITERALS))
    return str(rng.randrange(-20, 100))


def mutate(text, rng):
    """Rename registers, shuffle non-entry blocks and change some literals."""
    lines = [line.split(";", 1)[0].rstrip() for line in text.splitlines()]
    lines = [line for line in lines if line.strip()]
    names = sorted({int(n) for n in REGISTER.findall("\n".join(lines))})
    targets = rng.sample(range(0, 999999 if rng.random() < 0.3 else 4 * len(names) + 8), len(names))
    mapping = dict(zip(names, targets))
    header, body = lines[0], lines[1:]
    blocks = []
    for line in body:
        if LABEL.match(line.strip()):
            blocks.append([line])
        else:
            blocks[-1].append(line)
    rest = blocks[1:]
    rng.shuffle(rest)
    blocks = blocks[:1] + rest
    change = rng.random() < 0.6
    out = []
    for line in [header] + [line for block in blocks for line in block]:
        line = REGISTER.sub(lambda m: "v%d" % mapping[int(m.group(1))], line)
        stripped = line.strip()
        if change and not stripped.startswith("func") and not LABEL.match(stripped):
            op = stripped.split()[0]
            if op not in ("addsh",) and rng.random() < 0.15:
                pieces = line.split(",")
                for index in range(1, len(pieces)):
                    if NUMBER.fullmatch(pieces[index].strip()) and rng.random() < 0.5:
                        pieces[index] = " " + literal(rng)
                line = ",".join(pieces)
        out.append(line)
    return "\n".join(out) + "\n"


def generate(rng, machine):
    """A random program with loops bounded by counters."""
    params = ["v%d" % i for i in range(rng.randrange(0, 5))]
    pool = ["v%d" % i for i in range(rng.randrange(4, 14))]
    nblocks = rng.randrange(1, 9)
    labels = ["b%d" % i for i in range(nblocks)]
    counters = ["v%d" % (40 + i) for i in range(nblocks)]

    def src():
        return rng.choice(pool) if rng.random() < 0.7 else literal(rng)

    def address():
        r = rng.random()
        if r < 0.3:
            return str(rng.choice((0, 1, 5, 100, 3071, 3072, 3073, 4095, 4096)))
        if r < 0.6:
            return rng.choice(pool)
        return None

    lines = ["func gen(%s)" % ", ".join(params)]
    first = True
    for index, label in enumerate(labels):
        lines.append(label + ":")
        if first:
            for counter in counters:
                if params and rng.random() < 0.5:
                    lines.append("  and %s, %s, 7" % (counter, rng.choice(params)))
                else:
                    lines.append("  const %s, %d" % (counter, rng.randrange(0, 6)))
            first = False
        for _ in range(rng.randrange(1, 10)):
            dest = rng.choice(pool)
            r = rng.random()
            if r < 0.40:
                op = rng.choice(("add", "sub", "mul", "and", "or", "xor", "shl", "shr", "lt", "eq"))
                lines.append("  %s %s, %s, %s" % (op, dest, src(), src()))
            elif r < 0.47:
                lines.append("  %s %s, %s, %s" % (rng.choice(("div", "mod")), dest, src(), src()))
            elif r < 0.55:
                addr = address()
                if addr is None:
                    tmp = rng.choice(pool)
                    lines.append("  and %s, %s, %d" % (tmp, rng.choice(pool), rng.choice((15, 255, 4095))))
                    addr = tmp
                lines.append("  load %s, %s" % (dest, addr))
            elif r < 0.63:
                addr = address()
                if addr is None:
                    tmp = rng.choice(pool)
                    lines.append("  and %s, %s, %d" % (tmp, rng.choice(pool), rng.choice((15, 255, 4095))))
                    addr = tmp
                lines.append("  store %s, %s" % (addr, src()))
            elif r < 0.67:
                lines.append("  vload %s, %d" % (dest, rng.randrange(0, 4)))
            elif r < 0.70:
                lines.append("  vstore %d, %s" % (rng.randrange(0, 4), src()))
            elif r < 0.75:
                lines.append("  select %s, %s, %s, %s" % (dest, src(), src(), src()))
            elif r < 0.80:
                lines.append("  mov %s, %s" % (dest, src()))
            elif r < 0.85:
                lines.append("  const %s, %s" % (dest, literal(rng)))
            elif r < 0.90 and machine["fused_mac"]:
                lines.append("  mac %s, %s, %s, %s" % (dest, src(), src(), src()))
            elif r < 0.95 and machine["fused_addsh"]:
                lines.append("  addsh %s, %s, %s, %d" % (dest, src(), src(), rng.randrange(32)))
            else:
                lines.append("  add %s, %s, 1" % (dest, dest))
        r = rng.random()
        later = labels[index + 1:]
        if r < 0.08:
            lines.append("  trap %s" % src())
        elif r < 0.2 or not later and r < 0.6:
            lines.append("  ret %s" % src())
        elif r < 0.45 and index > 0:
            back = rng.choice(labels[:index + 1])
            counter = counters[index]
            lines.append("  sub %s, %s, 1" % (counter, counter))
            lines.append("  lt %s, %s, 0" % ("v60", counter))
            exit_label = rng.choice(later) if later else None
            if exit_label is None:
                lines.append("  br v60, done, %s" % back)
            else:
                lines.append("  br v60, %s, %s" % (exit_label, back))
        elif later:
            if rng.random() < 0.5:
                lines.append("  jmp %s" % rng.choice(later))
            else:
                lines.append("  br %s, %s, %s" % (rng.choice(pool), rng.choice(later), rng.choice(later)))
        else:
            lines.append("  ret %s" % src())
    lines.append("done:")
    lines.append("  ret %s" % rng.choice(pool))
    if rng.random() < 0.3:
        lines.append("dead:")
        lines.append("  store 7, 9")
        lines.append("  jmp b0")
    return "\n".join(lines) + "\n"


def profile_text(program, rng, sample_profile):
    labels = list(program.block_labels)
    choice = rng.random()
    if sample_profile is not None and choice < 0.4:
        return sample_profile
    lines = ["# profile"]
    for _ in range(rng.randrange(0, 12)):
        lines.append("edge %s %s %d" % (rng.choice(labels), rng.choice(labels), rng.randrange(0, 10**6)))
    if choice > 0.85:
        lines.append("edge nowhere b0 5")
        lines.append("garbage line here")
        lines.append("edge %s %s 99999999999999999999" % (labels[0], labels[-1]))
    return "\n".join(lines) + "\n"


def run_kopt(kopt, program, machine, output, profile, timeout):
    args = [str(kopt), str(program), "--machine", str(machine), "-o", str(output)]
    if profile is not None:
        args += ["--profile", str(profile)]
    start = perf_counter()
    result = subprocess.run(args, capture_output=True, timeout=timeout, cwd=str(output.parent))
    return perf_counter() - start, result


def check_case(options, index):
    rng = Random(options.seed * 1000003 + index)
    module = load_kit(options.kit)
    machine = random_machine(rng)
    if rng.random() < 0.25:
        machine_path = rng.choice(sorted((options.kit / "machines").glob("*.cfg")))
        machine = {}
        for line in machine_path.read_text().splitlines():
            line = line.split("#", 1)[0]
            if "=" in line:
                key, value = line.split("=")
                machine[key.strip()] = int(value)
        mtext = machine_path.read_text()
    else:
        mtext = machine_text(machine, rng)
    bench = sorted((options.kit / "bench").glob("*.kir"))
    sample_profile = None
    if rng.random() < options.generated:
        text = generate(rng, machine)
        name = "gen%d" % index
        sample = None
    else:
        source = rng.choice(bench)
        name = source.stem
        text = mutate(source.read_text(), rng)
        sample = source.with_suffix(".in")
        prof = source.with_suffix(".prof")
        if prof.is_file():
            sample_profile = prof.read_text()
    tag = "case %d (%s)" % (index, name)
    try:
        original = module.parse_program(text)
        mc_check = module.load_machine
    except module.KirError as exc:
        return tag, "skip", "original invalid: %s" % exc, 0.0
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        program_path = root / "in.kir"
        program_path.write_text(text)
        machine_path = root / "m.cfg"
        machine_path.write_text(mtext)
        mc = mc_check(str(machine_path))
        try:
            module.check_machine_features(original, mc)
        except module.KirError:
            return tag, "skip", "original uses missing fused op", 0.0
        profile_path = None
        if rng.random() < 0.5:
            profile_path = root / "p.prof"
            profile_path.write_text(profile_text(original, rng, sample_profile))
        out1 = root / "out1.kir"
        out2 = root / "out2.kir"
        try:
            elapsed, result = run_kopt(options.kopt, program_path, machine_path, out1, profile_path, options.timeout)
        except subprocess.TimeoutExpired:
            return tag, "fail", "timeout", options.timeout
        if result.returncode != 0 or not out1.is_file():
            return tag, "fail", "exit %d: %s" % (result.returncode, result.stderr.decode()[-500:]), elapsed
        extra = sorted(p.name for p in root.iterdir()) != sorted(
            ["in.kir", "m.cfg", "out1.kir"] + (["p.prof"] if profile_path else []))
        if extra:
            return tag, "fail", "extra files: %s" % sorted(p.name for p in root.iterdir()), elapsed
        try:
            run_kopt(options.kopt, program_path, machine_path, out2, profile_path, options.timeout)
        except subprocess.TimeoutExpired:
            return tag, "fail", "timeout on second run", elapsed
        if out1.read_bytes() != out2.read_bytes():
            return tag, "fail", "nondeterministic output", elapsed
        out_text = out1.read_text()
        try:
            candidate = module.parse_program(out_text)
            module.check_machine_features(candidate, mc)
        except module.KirError as exc:
            return tag, "fail", "invalid output: %s" % exc, elapsed
        if len(candidate.params) != len(original.params) or candidate.name != original.name:
            return tag, "fail", "signature changed", elapsed
        left_code = module.compile_program(original)
        right_code = module.compile_program(candidate)
        sample_inputs = module.Inputs([0] * len(original.params), [], {})
        if sample is not None and sample.is_file():
            try:
                sample_inputs = module.load_input(str(sample), len(original.params))
            except module.KirError:
                pass
        inputs = list(module.generate_inputs(len(original.params), options.inputs, options.seed + index))
        inputs += list(input_sets(module, original, sample_inputs, options.inputs // 2, index))
        compared = 0
        for number, inp in enumerate(inputs):
            left = module.execute(left_code, inp.args, inp.segments, inp.vin, options.max_steps)
            if left.status == "TIMEOUT":
                continue
            right = module.execute(right_code, inp.args, inp.segments, inp.vin, left.steps * 20 + 1000)
            compared += 1
            difference = module.first_difference(left, right)
            if difference is not None:
                (root.parent / ("fuzz-fail-%d" % index)).mkdir(exist_ok=True)
                keep = root.parent / ("fuzz-fail-%d" % index)
                (keep / "in.kir").write_text(text)
                (keep / "m.cfg").write_text(mtext)
                (keep / "out.kir").write_text(out_text)
                (keep / "input.in").write_text(module.format_input(inp))
                if profile_path:
                    (keep / "p.prof").write_text(profile_path.read_text())
                return tag, "fail", "input %d: %s (saved in %s)" % (number, difference, keep), elapsed
        slower = ""
        res = module.execute(left_code, sample_inputs.args, sample_inputs.segments, sample_inputs.vin,
                             options.max_steps, want_trace=True)
        if res.status != "TIMEOUT":
            other = module.execute(right_code, sample_inputs.args, sample_inputs.segments,
                                   sample_inputs.vin, res.steps * 20 + 1000, want_trace=True)
            before = module.time_trace(left_code, res.trace, mc)
            after = module.time_trace(right_code, other.trace, mc)
            if after > before:
                slower = "slower on sample: %d -> %d" % (before, after)
        if slower:
            return tag, "slower", slower, elapsed
        return tag, "ok", "%d inputs" % compared, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--kopt", type=Path, default=DEFAULT_OPTIMIZER)
    parser.add_argument("--cases", type=positive, default=200)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--inputs", type=positive, default=100)
    parser.add_argument("--generated", type=float, default=0.5, help="fraction of generated programs")
    parser.add_argument("--max-steps", type=positive, default=200000)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--jobs", type=positive, default=1)
    options = parser.parse_args()
    options.kit = options.kit.resolve()
    options.kopt = options.kopt.resolve()
    counts = {}
    worst = 0.0
    indices = range(options.start, options.start + options.cases)
    with ProcessPoolExecutor(max_workers=options.jobs) as executor:
        for tag, status, detail, elapsed in executor.map(check_case, [options] * len(indices), indices):
            counts[status] = counts.get(status, 0) + 1
            worst = max(worst, elapsed)
            if status in ("fail", "slower"):
                print("%s %s: %s" % (status.upper(), tag, detail), flush=True)
    print("Cases: %s. Slowest kopt run: %.3f s." % (
        ", ".join("%s %d" % item for item in sorted(counts.items())), worst))
    return int(counts.get("fail", 0) > 0)


if __name__ == "__main__":
    sys.exit(main())
