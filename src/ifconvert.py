"""Replace small branch arms with select instructions.

A block that ends in `br c, T, F` where one or both arms are small blocks with
no other predecessor, that jump to the same join block and contain only
calculations, is rewritten to run both arms into fresh registers and pick each
result with `select`. The arms run exactly where the branch was, so a load in an
arm reads the same memory it would have read. A load is allowed only when its
address is a literal below 4096 or was already used by a load or store in the
branching block after its last write, so the extra load can never trap.
"""
from .flow import live_out
from .ir import is_reg, parents, writes
from .kirlib import MEM_WORDS
from .values import PURE

ARM_LIMIT = 4


def _safe_addresses(body):
    safe = set()
    for op, args in body[:-1]:
        if op in ("load", "store"):
            safe.add(args[1] if op == "load" else args[0])
        if writes(op):
            safe.discard(args[0])
    return safe


def _arm(body, safe):
    code = body[:-1]
    if len(code) > ARM_LIMIT:
        return None
    changed = set()
    for op, args in code:
        if op == "load":
            address = args[1]
            if is_reg(address) and (address not in safe or address in changed):
                return None
            if not is_reg(address) and address >= MEM_WORDS:
                return None
        elif op not in PURE:
            return None
        changed.add(args[0])
    return code


def _rename(code, fresh, keep=frozenset()):
    """Gives every register written in `code` a fresh name, except those in `keep`."""
    mapping = {}
    result = []
    for op, args in code:
        sources = tuple(mapping.get(arg, arg) if is_reg(arg) else arg for arg in args[writes(op):])
        if writes(op):
            target = args[0]
            if target not in keep:
                if target not in mapping:
                    mapping[target] = fresh[0]
                    fresh[0] -= 1
                target = mapping[target]
            result.append((op, (target,) + sources))
        else:
            result.append((op, sources))
    return result, mapping


def _written(code):
    return {args[0] for op, args in code if writes(op)}


def convert(blocks, params, fresh_limit=999999):
    live = live_out(blocks)
    if live is None:
        return blocks
    used = set(params)
    for body in blocks.values():
        used.update(arg for _, args in body for arg in args if is_reg(arg))
    fresh = [min(used, default=0) - 1]
    entry = next(iter(blocks))
    incoming = parents(blocks)
    result = dict(blocks)
    removed = set()
    for label in list(blocks):
        if label in removed:
            continue
        body = result[label]
        op, args = body[-1]
        if op != "br":
            continue
        condition, taken, other = args
        if taken == other or label in (taken, other) or not is_reg(condition):
            continue

        def arm_of(target):
            block = result[target]
            if (target == entry or target in removed or incoming[target] != [label]
                    or block[-1][0] != "jmp"):
                return None, None
            return _arm(block, _safe_addresses(body)), block[-1][1][0]

        join = None
        arms = {}
        for target in (taken, other):
            code, goes = arm_of(target)
            if code is not None:
                arms[target] = (code, goes)
        if taken in arms and other in arms and arms[taken][1] == arms[other][1]:
            join = arms[taken][1]
        elif taken in arms and arms[taken][1] == other:
            join = other
            arms.pop(other, None)
        elif other in arms and arms[other][1] == taken:
            join = taken
            arms.pop(taken, None)
        else:
            continue
        if join in (label, taken, other) and join not in (taken, other):
            continue
        if any(target == join for target in arms):
            continue
        first = arms.get(other, ([], None))[0]
        last = arms.get(taken, ([], None))[0]
        if not last:
            first, last = last, first
        needed = set().union(*(live[target] for target in arms))
        written_first, written_last = _written(first), _written(last)
        written = written_first | written_last
        if condition in written or ~fresh[0] + len(written) > fresh_limit:
            continue
        code_first, map_first = _rename(first, fresh)
        code_last, map_last = _rename(last, fresh, (written_last - needed) | written_first)
        if last is arms.get(taken, ([], None))[0]:
            map_taken, map_other = map_last, map_first
        else:
            map_taken, map_other = map_first, map_last
        picks = []
        for register in sorted(written & needed, key=lambda reg: -reg):
            picks.append(("select", (register, condition, map_taken.get(register, register),
                                     map_other.get(register, register))))
        result[label] = body[:-1] + code_first + code_last + picks + [("jmp", (join,))]
        for target in arms:
            removed.add(target)
            del result[target]
        preds = [pred for pred in incoming[join] if pred not in arms and pred != label]
        incoming[join] = preds + [label]
        for target in (taken, other):
            if target not in arms and target != join:
                incoming[target] = [pred for pred in incoming[target] if pred != label]
    return result
