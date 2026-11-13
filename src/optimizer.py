from .backend import tune, tune_pair
from collections import Counter

from .flow import clean
from .ir import is_reg, reads
from .linear import combine
from .horner import split_chains
from .ifconvert import convert
from .selects import simplify_selects
from .divide import expand
from .loops import optimize_loops
from .popcount import replace_loops
from .rewrite import coalesce, common_across_edges, prune
from .values import simplify


def optimize(blocks, params, machine, profile, disabled=frozenset()):
    if "cfg" not in disabled:
        blocks = clean(blocks)
    if "values" not in disabled:
        blocks = simplify(blocks, params, machine, "fusion" not in disabled)
    if "cfg" not in disabled:
        blocks = clean(blocks)
    if "sums" not in disabled:
        blocks = combine_blocks(blocks, params, machine)
    if "common" not in disabled:
        blocks = common_across_edges(blocks)
    if "dead" not in disabled or "fusion" not in disabled:
        blocks = prune(blocks, machine, combine="fusion" not in disabled, remove="dead" not in disabled)
    size = sum(map(len, blocks.values()))
    if size <= 30000:
        if "select" not in disabled:
            # Converting an inner diamond can leave its parent with two
            # small arms, so repeat a few times with merging in between.
            for _ in range(3):
                converted = convert(blocks, params)
                if converted is blocks or list(converted.items()) == list(blocks.items()):
                    break
                blocks = clean(converted) if "cfg" not in disabled else converted
            blocks = simplify_selects(blocks, params)
        if "loops" not in disabled:
            blocks = optimize_loops(blocks, params, machine, profile, disabled)
        if "bitloops" not in disabled:
            blocks = replace_loops(blocks, params, machine)
        if "values" not in disabled:
            blocks = simplify(blocks, params, machine, "fusion" not in disabled)
        if "dead" not in disabled or "fusion" not in disabled:
            blocks = prune(blocks, machine, combine="fusion" not in disabled, remove="dead" not in disabled)
        if "divide" not in disabled:
            blocks = expand(blocks, params, machine)
        if "horner" not in disabled:
            split = split_chains(blocks, params, machine)
            if split is not blocks:
                blocks = prune(split, machine, combine="fusion" not in disabled, remove="dead" not in disabled)
        if "cfg" not in disabled:
            blocks = clean(blocks)
    if "copies" not in disabled:
        blocks = coalesce(blocks, params)
    return blocks


def place(blocks, params, machine, profile, disabled=frozenset()):
    return tune(blocks, params, machine, profile, schedule="schedule" not in disabled,
                reuse="reuse" not in disabled, layout="layout" not in disabled,
                rename=False if "rename" in disabled else True)


def place_pair(blocks, params, machine, profile, disabled=frozenset()):
    """(guarded, spread) placements; see backend.tune_pair."""
    if {"schedule", "reuse", "rename"} & disabled:
        placed = place(blocks, params, machine, profile, disabled)
        return placed, placed
    return tune_pair(blocks, params, machine, profile, layout="layout" not in disabled)


def place_loose(blocks, params, machine, profile, disabled=frozenset()):
    """Placement with eager name sharing and unguarded backward schedules, or None."""
    if {"schedule", "reuse", "rename"} & disabled:
        return None
    return tune(blocks, params, machine, profile, layout="layout" not in disabled, rename=None, loose=True)


def combine_blocks(blocks, params, machine):
    used = set(params)
    for body in blocks.values():
        used.update(arg for _, args in body for arg in args if is_reg(arg))
    fresh = min(used, default=0) - 1
    reads_anywhere = Counter(register for body in blocks.values()
                             for op, args in body for register in reads(op, args))
    result = {}
    for label, body in blocks.items():
        if ~fresh + 2 * len(body) > 999999:
            result[label] = body
            continue
        result[label], count = combine(body, machine, reads_anywhere, fresh)
        fresh -= count
    return result
