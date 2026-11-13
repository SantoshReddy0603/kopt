from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from _suite import CheckError, cases, optimize, profile_for, report_error, suite_parser


def compare(optimizer, case):
    program, machine = case
    name = "{} / {}".format(program.stem, machine.stem)
    try:
        with TemporaryDirectory() as left, TemporaryDirectory() as right:
            outputs = []
            for seed, directory in enumerate((left, right), 1):
                env = dict(os.environ, PYTHONHASHSEED=str(seed))
                destination = Path(directory) / "out.kir"
                optimize(optimizer, program, machine, destination,
                         profile_for(program), timeout=5, env=env)
                if sorted(path.name for path in destination.parent.iterdir()) != [destination.name]:
                    raise CheckError("optimizer left extra files in its working directory")
                outputs.append(destination.read_bytes())
            if outputs[0] != outputs[1]:
                raise CheckError("output changed across working directories or hash seeds")
    except (CheckError, OSError) as exc:
        return "{}\n{}".format(name, exc)
    return None


def main():
    parser = suite_parser("Compare optimizer output across two working directories and hash seeds.")
    options = parser.parse_args()
    try:
        programs, machines = cases(options)
    except CheckError as exc:
        return report_error(exc)
    combinations = [(program, machine) for program in programs for machine in machines]
    with ThreadPoolExecutor(max_workers=options.jobs) as executor:
        errors = [error for error in executor.map(partial(compare, options.kopt), combinations) if error]
    for error in errors:
        print(error, file=sys.stderr)
    print("Byte-identical outputs: {} of {} cases".format(
        len(combinations) - len(errors), len(combinations)))
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
