from collections import Counter

from .flow import live_out
from .ir import is_reg, parents, reads, writes
from .kirlib import LATKEY, MEM_WORDS
from .ranges import safe_loads
from .values import COMMUTATIVE, PURE


def expression(op, args):
    operands = args[1:]
    return op, tuple(sorted(operands)) if op in COMMUTATIVE else operands


def common_across_edges(blocks):
    entry = next(iter(blocks))
    incoming = parents(blocks)
    summaries = {}
    for label, body in blocks.items():
        if len(body) > 128:
            continue
        definitions = {args[0]: i for i, (op, args) in enumerate(body) if writes(op)}
        available = {}
        for dest, i in definitions.items():
            op, args = body[i]
            if op in PURE and op not in ("mov", "const") and all(
                definitions.get(reg, -1) < i for reg in reads(op, args)
            ):
                available[expression(op, args)] = dest
        summaries[label] = available

    result = {}
    for label, body in blocks.items():
        preds = incoming[label]
        if label == entry or not preds:
            result[label] = body
            continue
        available = summaries.get(preds[0], {})
        for pred in preds[1:]:
            other = summaries.get(pred, {})
            available = {key: reg for key, reg in available.items() if other.get(key) == reg}
            if not available:
                break
        if not available:
            result[label] = body
            continue
        written, updated = set(), []
        for op, args in body:
            if op in PURE and op not in ("mov", "const"):
                dest = available.get(expression(op, args))
                if dest is not None and dest not in written and all(reg not in written for reg in reads(op, args)):
                    if dest == args[0]:
                        continue
                    op, args = "mov", (args[0], dest)
            if writes(op):
                written.add(args[0])
            updated.append((op, args))
        result[label] = updated
    return result


def removable(op, args):
    if op in PURE:
        return True
    if op in ("div", "mod"):
        return args[2] > 0
    if op == "load":
        return 0 <= args[1] < MEM_WORDS
    return False


def combine_pair(previous, current, live, machine):
    op, before = previous
    after_op, args = current
    if after_op == "mov" and writes(op) and before[0] == args[1] and args[1] not in live:
        return op, (args[0],) + before[1:]
    if after_op != "add" or op not in ("mul", "shl"):
        return None
    intermediate = before[0]
    dest, left, right = args
    if (left == intermediate) == (right == intermediate) or (intermediate in live and intermediate != dest):
        return None
    other = right if left == intermediate else left
    fused = "mac" if op == "mul" else "addsh"
    if not machine["fused_" + fused] or machine[LATKEY[fused]] > machine[LATKEY[op]] + machine["lat_alu"]:
        return None
    if op == "mul":
        return "mac", (dest, before[1], before[2], other)
    if not is_reg(before[2]):
        return "addsh", (dest, other, before[1], before[2] & 31)
    return None


def prune(blocks, machine, *, combine=True, remove=True):
    out = live_out(blocks)
    if out is None:
        return blocks
    result = {}
    safe = safe_loads(blocks) if remove else {}
    for label, body in blocks.items():
        live = out[label].copy()
        kept = []
        index = len(body) - 1
        valid = safe.get(label, ())
        while index >= 0:
            op, args = body[index]
            index -= 1
            if writes(op):
                dest = args[0]
                if remove and dest not in live and (removable(op, args) or index + 1 in valid):
                    continue
                if combine and index >= 0:
                    fused = combine_pair(body[index], (op, args), live, machine)
                    if fused is not None:
                        op, args = fused
                        index -= 1
                live.discard(dest)
            live.update(reads(op, args))
            kept.append((op, args))
        result[label] = list(reversed(kept))
    return result


def coalesce(blocks, params):
    """Drop `mov a, b` when a and b are each written exactly once.

    The copy must follow b's definition in the same block, and a must not be
    read in that block before the copy. Every register starts at 0, so on
    any path both registers are then either still unwritten or equal, and a
    can be replaced by b everywhere.
    """
    if sum(map(len, blocks.values())) > 30000:
        return blocks
    fixed = set(params)
    for _ in range(4):
        counts = Counter(args[0] for body in blocks.values() for op, args in body if writes(op))
        names = {}
        for body in blocks.values():
            defined = set()
            read = set()
            for op, args in body:
                if op == "mov" and is_reg(args[1]):
                    a, b = args
                    if (a != b and a not in fixed and b not in fixed and counts[a] == 1
                            and counts[b] == 1 and b in defined and a not in read
                            and a not in names and b not in names and a not in names.values()
                            and b not in names.values()):
                        names[a] = b
                read.update(reads(op, args))
                if writes(op):
                    defined.add(args[0])
        if not names:
            break
        result = {}
        for label, body in blocks.items():
            code = []
            for op, args in body:
                if op == "mov" and args[0] in names:
                    continue
                code.append((op, tuple(names.get(arg, arg) if is_reg(arg) else arg for arg in args)))
            result[label] = code
        blocks = result
    return blocks
