import argparse
import sys
from pathlib import Path

from .affine import synthesize
from .bits import specialize as bit_form
from .finite import cost, samples, specialize, spread
from .flow import has_short_exit
from .ir import from_program, is_reg, read_profile, render
from .kirlib import KirError, check_machine_features, load_machine, parse_program
from .optimizer import optimize, place, place_loose, place_pair
from .paths import specialize as path_form
from .probe import PROBE_STEPS, baseline_runs, candidate_cycles
from .registers import color


def decode(data):
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    return data.decode("utf-8-sig")


def validated(name, params, blocks, machine):
    text = render(name, params, blocks)
    result = parse_program(text)
    if result.name != name or tuple(~param for param in result.params) != params:
        raise KirError("optimizer changed the function signature")
    check_machine_features(result, machine)
    return text, result


def names(blocks, params):
    return set(params) | {arg for body in blocks.values() for _, args in body for arg in args if is_reg(arg)}




def lookup_version(program, original, params, machine, transformed, keep=None):
    """Fastest table version of the program on sample inputs, or None.

    The table comes from the original program when it qualifies, otherwise
    from the transformed one (for example once dead loads are gone).
    """
    sources = [(program, original)]
    if list(transformed.items()) != list(original.items()):
        try:
            sources.append((validated(program.name, params, transformed, machine)[1], transformed))
        except KirError:
            pass
    found = []
    keys = []
    for source, blocks in sources:
        found = list(specialize(source, blocks, params))
        if found:
            keys = spread(blocks, params)
            break
    for source, _ in sources:
        closed = bit_form(source, params)
        if closed is not None:
            found.append(closed)
            break
    if not found:
        return None
    # Keys spread over the whole masked range stand for random inputs
    # better than the fixed samples, which favour small values.
    argsets = keys or samples(len(params))
    reference = cost(program, machine, argsets)
    if reference is None:
        return None
    best = None

    def shapes():
        for blocks in found:
            yield blocks
            pair = place_pair(blocks, params, machine, None)
            tidy = optimize(blocks, params, machine, {}, frozenset({"loops", "select"}))
            pair += place_pair(tidy, params, machine, None)
            for index, placed in enumerate(pair):
                if any(placed is other for other in pair[:index]):
                    continue
                yield placed
                yield color(placed, params, machine)

    for blocks in shapes():
        try:
            text, candidate = validated(program.name, params, blocks, machine)
        except KirError:
            continue
        cycles = cost(candidate, machine, argsets)
        if cycles is not None and cycles < (best[0] if best else reference):
            best = (cycles, text, candidate)
            if keep is not None:
                keep(text.encode("utf-8"))
    return best and (best[1], best[2], argsets, best[0])


def prefer(lookup, text, machine):
    """Pick between the table version and another candidate on the table samples."""
    other = cost(parse_program(text), machine, lookup[2])
    if other is not None and other < lookup[3]:
        return text.encode("utf-8")
    return lookup[0].encode("utf-8")


def optimize_source(source, program, machine, profile, disabled=frozenset(), keep=None):
    keep = keep if keep is not None else (lambda data: None)
    params = tuple(~param for param in program.params)
    original = from_program(program)
    probe = len(program.instrs) <= 4096 or has_short_exit(original, PROBE_STEPS)
    baseline = baseline_runs(program, machine) if probe else []
    best_cycles = sum(run[-1] for run in baseline)
    best = source
    transformed = optimize(original, params, machine, profile, disabled | {"sums", "select"})
    lookup = None
    if "finite" not in disabled:
        lookup = lookup_version(program, original, params, machine, transformed, keep)
    if lookup is not None:
        best = lookup[0].encode("utf-8")
        keep(best)
        if baseline:
            measured = candidate_cycles(lookup[1], machine, baseline, veto=False)
            if measured is not None:
                best_cycles = min(best_cycles, measured)
    placed, spread = place_pair(transformed, params, machine, profile, disabled)
    if not baseline:
        if list(placed.items()) != list(original.items()):
            try:
                text, _ = validated(program.name, params, placed, machine)
                result = prefer(lookup, text, machine) if lookup else text.encode("utf-8")
                keep(result)
                return result
            except KirError:
                pass
        return best

    def candidates():
        split = path_form(program, params) if "paths" not in disabled else None
        if split is not None:
            ordered = place(split, params, machine, profile, disabled)
            yield ordered
            yield split
            yield color(ordered, params, machine)
            tidy = optimize(split, params, machine, profile, disabled | {"loops", "select"})
            yield place(tidy, params, machine, profile, disabled)
            inline = path_form(program, params, inline_larger=True)
            if inline is not None and list(inline) != list(split):
                yield inline
                yield place(inline, params, machine, profile, disabled)
        variants = [(transformed, placed)]
        extras = []
        known = [transformed]
        for extra in ({"rotate", "sums", "select", "horner", "divide"}, {"sums"}, {"rotate", "sums"}, {"group"}, {"narrow"}, {"copies"}, set()):
            if extra and extra <= disabled:
                continue
            blocks = optimize(original, params, machine, profile, disabled | extra)
            if all(list(blocks.items()) != list(other.items()) for other in known):
                known.append(blocks)
                guarded, wide = place_pair(blocks, params, machine, profile, disabled)
                variants.append((blocks, guarded))
                if wide is not guarded:
                    extras.append((blocks, wide))
        if spread is not placed:
            extras.insert(0, (transformed, spread))
        variants += extras
        if "affine" not in disabled:
            for blocks in known:
                rebuilt = synthesize(blocks, params)
                if rebuilt is not None:
                    yield place(rebuilt, params, machine, profile, disabled)
                    yield rebuilt
                    break
        if len(program.instrs) <= 1200:
            variants.append((original, place(original, params, machine, profile, disabled)))
        for blocks, scheduled in variants:
            yield scheduled
            yield blocks
            if "color" not in disabled:
                colored = color(blocks, params, machine)
                yield color(scheduled, params, machine)
                yield colored
                guarded, wide = place_pair(colored, params, machine, profile, disabled)
                yield guarded
                if wide is not guarded:
                    yield wide
        # Eager name sharing, judged on whole runs rather than block by block.
        for blocks in known:
            loose = place_loose(blocks, params, machine, profile, disabled)
            if loose is not None:
                yield loose
                if "color" not in disabled:
                    yield color(loose, params, machine)

    seen = set()
    best_over = False
    # Deterministic work limit: instructions checked plus simulated steps.
    # It depends only on the input, so repeated runs stop at the same point.
    work = [0]
    for blocks in candidates():
        if work[0] > WORK_LIMIT:
            break
        work[0] += sum(len(body) for body in blocks.values())
        if list(blocks.items()) == list(original.items()):
            continue
        try:
            text, candidate = validated(program.name, params, blocks, machine)
        except KirError:
            continue
        if text in seen:
            continue
        seen.add(text)
        cycles = candidate_cycles(candidate, machine, baseline, work=work)
        if cycles is None:
            continue
        # On a tie, prefer a candidate that fits in the register file: the
        # synthetic inputs touch few registers, so they can hide reloads.
        over = len(names(blocks, params)) > machine["registers"]
        if cycles < best_cycles or cycles == best_cycles and best_over and not over:
            best = text.encode("utf-8")
            best_cycles = cycles
            best_over = over
            keep(prefer(lookup, text, machine) if lookup else best)
    if lookup is not None:
        best = prefer(lookup, best.decode("utf-8"), machine)
    return best


# About 1.5 times the most any public program needs (258k units).
WORK_LIMIT = 400000


def read_hints(path):
    if path is None:
        return {}
    try:
        return read_profile(decode(path.read_bytes()))
    except (OSError, ValueError) as error:
        print("kopt: ignoring profile: %s" % error, file=sys.stderr)
        return {}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="kopt")
    parser.add_argument("input", type=Path)
    parser.add_argument("--machine", required=True, type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("-o", required=True, type=Path, dest="output")
    args = parser.parse_args(argv)
    try:
        source = args.input.read_bytes()
        program = parse_program(decode(source))
        machine = load_machine(args.machine)
        check_machine_features(program, machine)
    except (OSError, ValueError, KirError) as error:
        print("kopt: %s" % error, file=sys.stderr)
        return 1
    profile = read_hints(args.profile)
    best = [source]
    try:
        best[0] = optimize_source(source, program, machine, profile)
    except Exception as error:
        print("kopt: optimizer failed (%s), writing the input unchanged" % error, file=sys.stderr)
        best[0] = source
    try:
        args.output.write_bytes(best[0])
    except OSError as error:
        print("kopt: %s" % error, file=sys.stderr)
        return 1
    return 0
