import argparse
import fnmatch
from functools import lru_cache
import importlib.util
from pathlib import Path
import subprocess
import sys
from time import perf_counter


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OPTIMIZER = PROJECT / "kopt"
DEFAULT_MACHINE = "machine_default.cfg"


class CheckError(Exception):
    pass


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return number


def suite_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--kopt", type=Path, default=DEFAULT_OPTIMIZER)
    parser.add_argument("--match", default="*", help="benchmark name glob")
    parser.add_argument("--machine", default="*", help="machine name glob")
    parser.add_argument("--jobs", type=positive, default=1)
    return parser


def select(paths, pattern):
    return [path for path in sorted(paths)
            if fnmatch.fnmatchcase(path.name, pattern)
            or fnmatch.fnmatchcase(path.stem, pattern)]


def cases(args, require_optimizer=True):
    args.kit = args.kit.resolve()
    args.kopt = args.kopt.resolve()
    for name in ("kirlib.py", "kvalidate", "ksim"):
        if not (args.kit / name).is_file():
            raise CheckError("missing kit file: {}".format(args.kit / name))
    if require_optimizer and not args.kopt.is_file():
        raise CheckError("missing optimizer: {}".format(args.kopt))
    programs = select((args.kit / "bench").glob("*.kir"), args.match)
    machines = select((args.kit / "machines").glob("*.cfg"), args.machine)
    if not programs:
        raise CheckError("no benchmark matched {!r}".format(args.match))
    if not machines:
        raise CheckError("no machine matched {!r}".format(args.machine))
    return programs, machines


@lru_cache(maxsize=4)
def load_kit(kit):
    spec = importlib.util.spec_from_file_location("organizer_kirlib", kit / "kirlib.py")
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    return module


def run(script, *args, cwd=None, timeout=120, env=None, allowed=(0,)):
    argv = [sys.executable, "-B", str(script)] + [str(arg) for arg in args]
    try:
        result = subprocess.run(argv, cwd=cwd, env=env, timeout=timeout,
                                capture_output=True, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as exc:
        raise CheckError("{} exceeded {} s".format(script.name, timeout)) from exc
    if result.returncode not in allowed:
        detail = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
        raise CheckError("{} exited {}\n{}".format(script.name, result.returncode, detail))
    return result.stdout


def optimize(optimizer, program, machine, output, profile=None, timeout=30, env=None):
    args = [program, "--machine", machine, "-o", output]
    if profile is not None:
        args.extend(("--profile", profile))
    start = perf_counter()
    run(optimizer, *args, cwd=output.parent, timeout=timeout, env=env)
    elapsed = perf_counter() - start
    if not output.is_file():
        raise CheckError("optimizer exited without writing {}".format(output.name))
    return elapsed


def profile_for(program):
    candidate = program.with_suffix(".prof")
    return candidate if candidate.is_file() else None


def simulate(kit, program, machine, sample=None, verbose=False, max_steps=None):
    args = [program, "--machine", machine]
    if sample is not None and sample.is_file():
        args.extend(("--input", sample))
    if verbose:
        args.append("--verbose")
    if max_steps is not None:
        args.extend(("--max-steps", max_steps))
    return run(kit / "ksim", *args, allowed=(0, 3))


def fields(output):
    return dict(line.split(": ", 1) for line in output.splitlines() if ": " in line)


def report_error(exc):
    print(str(exc), file=sys.stderr)
    return 1
