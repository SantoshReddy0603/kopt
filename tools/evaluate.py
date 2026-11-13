from concurrent.futures import ThreadPoolExecutor
from functools import partial
import math
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

from _suite import (CheckError, cases, fields, nonnegative, optimize,
                    profile_for, report_error, run, simulate, suite_parser)


def evaluate_case(options, case):
    program, machine = case
    baseline = None
    candidate = None
    elapsed = None
    status = "OK"
    try:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original = directory / "original.kir"
            output = directory / "optimized.kir"
            shutil.copyfile(program, original)
            sample = program.with_suffix(".in")
            if sample.is_file():
                shutil.copyfile(sample, original.with_suffix(".in"))
            elapsed = optimize(options.kopt, program, machine, output,
                               profile_for(program), timeout=5)
            run(options.kit / "kvalidate", original, "--machine", machine)
            run(options.kit / "kvalidate", output, "--against", original,
                "--machine", machine, "--random", options.random, "--seed", options.seed)
            original_fields = fields(simulate(options.kit, original, machine, sample))
            optimized_fields = fields(simulate(options.kit, output, machine, sample))
            if original_fields["status"] == "TIMEOUT":
                status = "NO TIMING"
            else:
                baseline = int(original_fields["cycles"])
                candidate = int(optimized_fields["cycles"])
    except (CheckError, OSError, KeyError, ValueError) as exc:
        status = "FAIL"
        detail = str(exc)
    else:
        detail = ""
    return (program.stem, machine.stem, baseline, candidate, elapsed, status, detail)


def main():
    parser = suite_parser("Measure cycles and check equivalence with the official tools.")
    parser.add_argument("--random", type=nonnegative, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    options = parser.parse_args()
    try:
        programs, machines = cases(options)
    except CheckError as exc:
        return report_error(exc)

    combinations = [(program, machine) for program in programs for machine in machines]
    print("benchmark                  machine                       baseline    output      ratio   time(s)  validation",
          flush=True)
    errors = []
    ratios = []
    untimed = 0
    slower = 0
    with ThreadPoolExecutor(max_workers=options.jobs) as executor:
        results = executor.map(partial(evaluate_case, options), combinations)
        for program, machine, baseline, candidate, elapsed, status, detail in results:
            if status == "FAIL":
                errors.append((program, machine, detail))
            if status == "NO TIMING":
                untimed += 1
            ratio = "-"
            if baseline is not None and candidate is not None:
                ratio_value = candidate / baseline
                ratios.append(ratio_value)
                slower += candidate > baseline
                ratio = "{:.5f}".format(ratio_value)
            wall_time = "{:.3f}".format(elapsed) if elapsed is not None else "-"
            print("{:26} {:29} {:>8} {:>9} {:>10} {:>9}  {}".format(
                program, machine, baseline if baseline is not None else "-",
                candidate if candidate is not None else "-", ratio, wall_time, status),
                flush=True)

    for program, machine, detail in errors:
        print("{} / {}\n{}".format(program, machine, detail), file=sys.stderr)
    if ratios:
        mean = math.exp(math.fsum(math.log(ratio) for ratio in ratios) / len(ratios))
        print("Geometric mean (output / baseline): {:.6f} over {} measured cases".format(
            mean, len(ratios)))
    else:
        print("Geometric mean unavailable: no completed timing cases")
    print("Validation failures: {} of {} cases".format(len(errors), len(combinations)))
    print("Untimed cases: {}. Slower cases: {}.".format(untimed, slower))
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
