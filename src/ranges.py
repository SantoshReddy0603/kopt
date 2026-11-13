"""Unsigned value ranges, used to prove load addresses valid.

A forward interval analysis over the control-flow graph. Every register starts
unknown at the entry. Branches on `lt` (both sides known to be non-negative as
signed numbers), on `eq` with a literal and on a register itself narrow the
ranges along each edge. Loop heads widen to the next threshold (a literal of
the program, the memory size, the largest signed value or the full word), and
two narrowing rounds then recover bounds such as `i < n` for counted loops.
"""
from .ir import is_reg, targets, writes
from .kirlib import MASK, MEM_WORDS

TOP = (0, MASK)
SIGNED = 0x7FFFFFFF
MAX_INSTRUCTIONS = 30000


def _get(state, arg):
    if not is_reg(arg):
        value = arg & MASK
        return value, value
    return state.get(arg, TOP)


def _bounded(lo, hi):
    return (lo, hi) if hi <= MASK else TOP


def _transfer(op, args, state):
    if not writes(op):
        return
    dest, sources = args[0], args[1:]
    a = _get(state, sources[0]) if sources and op not in ("const",) else None
    b = _get(state, sources[1]) if len(sources) > 1 else None
    if op == "const":
        value = sources[0] & MASK
        out = (value, value)
    elif op == "mov":
        out = a
    elif op == "add":
        out = _bounded(a[0] + b[0], a[1] + b[1])
    elif op == "sub":
        out = (a[0] - b[1], a[1] - b[0]) if a[0] >= b[1] else TOP
    elif op == "mul":
        out = _bounded(a[0] * b[0], a[1] * b[1])
    elif op == "and":
        out = (0, min(a[1], b[1]))
    elif op in ("or", "xor"):
        out = (0, (1 << max(a[1], b[1]).bit_length()) - 1)
    elif op == "shl":
        out = _bounded(a[0] << b[0], a[1] << b[0]) if b[0] == b[1] and b[0] < 32 else TOP
    elif op == "shr":
        out = (a[0] >> b[0], a[1] >> b[0]) if b[0] == b[1] and b[0] < 32 else (0, a[1])
    elif op in ("lt", "eq"):
        out = (0, 1)
    elif op == "select":
        c = _get(state, sources[2])
        out = (min(b[0], c[0]), max(b[1], c[1]))
    elif op in ("div", "mod"):
        if a[1] <= SIGNED and 1 <= b[0] and b[1] <= SIGNED:
            out = (0, min(a[1], b[1] - 1)) if op == "mod" else (a[0] // b[1], a[1] // b[0])
        else:
            out = TOP
    elif op == "mac":
        c = _get(state, sources[2])
        out = _bounded(a[0] * b[0] + c[0], a[1] * b[1] + c[1])
    elif op == "addsh":
        shift = sources[2] & 31
        out = _bounded(a[0] + (b[0] << shift), a[1] + (b[1] << shift))
    else:
        out = TOP
    state[dest] = out


def _refine(body, state, taken):
    """State along one edge of the block's final branch; None if infeasible."""
    op, args = body[-1]
    if op != "br" or args[1] == args[2]:
        return state
    cond = args[0]
    state = dict(state)
    lo, hi = _get(state, cond)
    if taken:
        lo = max(lo, 1)
    else:
        hi = 0
    if lo > hi:
        return None
    state[cond] = (lo, hi)
    for index in range(len(body) - 2, -1, -1):
        dop, dargs = body[index]
        if writes(dop) and dargs[0] == cond:
            break
    else:
        return state
    later = {a[0] for o, a in body[index + 1:-1] if writes(o)}
    if cond in dargs[1:] or any(is_reg(arg) and arg in later for arg in dargs[1:]):
        return state
    if dop == "lt":
        x, y = dargs[1], dargs[2]
        xr, yr = _get(state, x), _get(state, y)
        if xr[1] > SIGNED or yr[1] > SIGNED:
            return state
        if taken:
            xr, yr = (xr[0], min(xr[1], yr[1] - 1)), (max(yr[0], xr[0] + 1), yr[1])
        else:
            xr, yr = (max(xr[0], yr[0]), xr[1]), (yr[0], min(yr[1], xr[1]))
        if xr[0] > xr[1] or yr[0] > yr[1]:
            return None
        for reg, value in ((x, xr), (y, yr)):
            if is_reg(reg):
                state[reg] = value
    elif dop == "eq" and taken:
        x, y = dargs[1], dargs[2]
        xr, yr = _get(state, x), _get(state, y)
        both = (max(xr[0], yr[0]), min(xr[1], yr[1]))
        if both[0] > both[1]:
            return None
        for reg in (x, y):
            if is_reg(reg):
                state[reg] = both
    return state


def _join(left, right):
    if left is None:
        return dict(right)
    out = {}
    for reg in set(left) & set(right):
        a, b = left[reg], right[reg]
        joined = (min(a[0], b[0]), max(a[1], b[1]))
        if joined != TOP:
            out[reg] = joined
    return out


def _widen(old, new, thresholds):
    out = {}
    for reg, value in new.items():
        before = old.get(reg)
        if before is None:
            continue
        lo = value[0] if value[0] >= before[0] else 0
        hi = value[1]
        if hi > before[1]:
            hi = next(t for t in thresholds if t >= hi)
        if (lo, hi) != TOP:
            out[reg] = (lo, hi)
    return out


def _outputs(blocks, label, state):
    body = blocks[label]
    state = dict(state)
    for op, args in body:
        _transfer(op, args, state)
    op, args = body[-1]
    if op == "br" and args[1] != args[2]:
        return [(args[1], _refine(body, state, True)), (args[2], _refine(body, state, False))]
    return [(dest, state) for dest in targets(body[-1])]


def analyze(blocks):
    """Return {label: entry state} or None when the program is too large."""
    if sum(len(body) for body in blocks.values()) > MAX_INSTRUCTIONS:
        return None
    literals = {arg & MASK for body in blocks.values() for _, args in body
                for arg in args if not is_reg(arg) and isinstance(arg, int)}
    thresholds = sorted(literals | {MEM_WORDS - 1, SIGNED, MASK})
    entry = next(iter(blocks))
    states = {entry: {}}
    visits = {}
    work = [entry]
    budget = 64 * len(blocks) + 1024
    while work:
        budget -= 1
        if budget < 0:
            return None
        label = work.pop()
        for dest, out in _outputs(blocks, label, states[label]):
            if out is None:
                continue
            old = states.get(dest)
            new = _join(old, out)
            if old is not None and new == old:
                continue
            visits[dest] = visits.get(dest, 0) + 1
            if old is not None and visits[dest] > 3:
                new = _widen(old, new, thresholds)
                if new == old:
                    continue
            states[dest] = new
            if dest not in work:
                work.append(dest)
    for _ in range(2):
        incoming = {}
        for label in blocks:
            if label not in states:
                continue
            for dest, out in _outputs(blocks, label, states[label]):
                if out is not None:
                    incoming[dest] = _join(incoming.get(dest), out)
        incoming[entry] = {}
        states = {label: incoming[label] for label in blocks if label in incoming}
    return states


def safe_loads(blocks):
    """{label: indices of loads whose register address is always below MEM_WORDS}."""
    states = analyze(blocks)
    if states is None:
        return {}
    result = {}
    for label, body in blocks.items():
        if label not in states:
            continue
        state = dict(states[label])
        for index, (op, args) in enumerate(body):
            if op == "load" and is_reg(args[1]) and _get(state, args[1])[1] < MEM_WORDS:
                result.setdefault(label, set()).add(index)
            _transfer(op, args, state)
    return result
