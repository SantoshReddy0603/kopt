"""Bit-counting self loops replaced by a branch-free population count.

Two loop shapes are recognised on a single block that branches back to
itself:

* clear-lowest: ``x = x & (x - 1)`` and one or more ``c = c + k``, repeated
  while the new ``x`` is not zero;
* shift-out: ``c = c + (x & 1)`` and ``x = x >> 1`` (logical), repeated while
  the new ``x`` is not zero.

Both run once per set bit (the shift form once per bit up to the highest one)
and leave ``x`` at zero, so the loop is equal to ``c += k * popcount(x)``
followed by ``x = 0``. The clear-lowest form runs its body once even when
``x`` starts at zero; unless every way into the loop already tests ``x``
against zero, one is added for that case. Every other register the body
writes must be dead where the loop exits, and the body may contain only pure
arithmetic, so the rewrite cannot change traps, memory or volatile events.
"""

from .flow import live_out
from .ir import is_reg, parents, reads, writes
from .kirlib import MASK
from .values import PURE

BODY_LIMIT = 24


def _value(env, arg):
    if is_reg(arg):
        return env.get(arg, ("in", arg))
    return ("k", arg & MASK)


def _evaluate(body, env=None):
    env = dict(env or {})
    for op, args in body:
        if not writes(op):
            continue
        if op == "const":
            env[args[0]] = ("k", args[1] & MASK)
        elif op == "mov":
            env[args[0]] = _value(env, args[1])
        else:
            values = tuple(_value(env, arg) for arg in args[1:])
            if op in ("add", "and", "or", "xor", "mul", "eq") and values[1] < values[0]:
                values = (values[1], values[0])
            env[args[0]] = (op,) + values
    return env


def _minus_one(expr, x):
    return expr in (("sub", x, ("k", 1)), ("add", x, ("k", MASK)), ("add", ("k", MASK), x))


def _cleared(expr, x):
    """True when expr is x & (x - 1)."""
    return expr[0] == "and" and len(expr) == 3 and (
        (expr[1] == x and _minus_one(expr[2], x)) or (expr[2] == x and _minus_one(expr[1], x)))


def _nonzero(expr, value):
    """1 if expr is true exactly when value != 0, 0 if exactly when value == 0, else None."""
    if expr == value:
        return 1
    flips = 0
    while expr[0] == "eq" and ("k", 0) in expr[1:]:
        expr = expr[2] if expr[1] == ("k", 0) else expr[1]
        flips ^= 1
        if expr == value:
            return 1 ^ flips
    return None


def _match(label, body):
    op, args = body[-1]
    if op != "br" or (args[1] == label) == (args[2] == label) or len(body) > BODY_LIMIT:
        return None
    if any(op not in PURE for op, _ in body[:-1]):
        return None
    env = _evaluate(body[:-1])
    stay = 1 if args[1] == label else 0
    test = _value(env, args[0])
    for x, expr in env.items():
        start = ("in", x)
        if _cleared(expr, start):
            kind = "clear"
        elif expr == ("shr", start, ("k", 1)):
            kind = "shift"
        else:
            continue
        if _nonzero(test, expr) != stay:
            continue
        counters = {}
        for reg, value in env.items():
            if reg == x or value[0] != "add":
                continue
            own = ("in", reg)
            other = value[2] if value[1] == own else value[1] if value[2] == own else None
            if other is None:
                continue
            if kind == "clear" and other[0] == "k" and other[1]:
                counters[reg] = other[1]
            elif kind == "shift" and other in (("and", ("k", 1), start), ("and", start, ("k", 1))):
                counters[reg] = 1
        if not counters:
            continue
        others = [reg for reg in env if reg != x and reg not in counters]
        exit_label = args[2] if stay else args[1]
        return kind, x, counters, others, exit_label
    return None


def _guards(blocks, incoming, label, x):
    """Branches that test x != 0 on every way into the loop, or None.

    Each entry is (block, side) where side is the branch target that leads
    to the loop. Empty forwarding blocks in between are followed.
    """
    sites = []
    todo = [(pred, label) for pred in incoming[label] if pred != label]
    seen = set()
    while todo:
        pred, toward = todo.pop()
        if pred in seen or len(seen) > 16:
            return None
        seen.add(pred)
        body = blocks[pred]
        op, args = body[-1]
        if op == "jmp" and len(body) == 1 and incoming[pred] and pred != next(iter(blocks)):
            todo.extend((parent, pred) for parent in incoming[pred])
            continue
        if op != "br" or args[1] == args[2]:
            return None
        env = _evaluate(body[:-1])
        sense = _nonzero(_value(env, args[0]), env.get(x, ("in", x)))
        if sense is None or (args[1] == toward) != (sense == 1):
            return None
        sites.append((pred, toward))
    return sites


def _live_in(blocks, outgoing, label):
    live = set(outgoing[label])
    for op, args in reversed(blocks[label]):
        if writes(op):
            live.discard(args[0])
        live.update(reads(op, args))
    return live


def _count(x, a, machine):
    """Population count of x, left in x, with a as the only scratch register."""
    code = [
        ("shr", (a, x, 1)), ("and", (a, a, 0x55555555)), ("sub", (x, x, a)),
        ("and", (a, x, 0x33333333)), ("shr", (x, x, 2)), ("and", (x, x, 0x33333333)),
        ("add", (x, x, a)), ("shr", (a, x, 4)), ("add", (x, x, a)), ("and", (x, x, 0x0F0F0F0F)),
    ]
    if machine is not None and machine["lat_mul"] <= 2 * machine["lat_alu"]:
        code += [("mul", (x, x, 0x01010101)), ("shr", (x, x, 24))]
    else:
        code += [("shr", (a, x, 8)), ("add", (x, x, a)), ("shr", (a, x, 16)),
                 ("add", (x, x, a)), ("and", (x, x, 63))]
    return code


def replace_loops(blocks, params, machine):
    if sum(map(len, blocks.values())) > 30000:
        return blocks
    found = [(label, _match(label, body)) for label, body in blocks.items()]
    found = [(label, match) for label, match in found if match is not None]
    if not found:
        return blocks
    outgoing = live_out(blocks)
    if outgoing is None:
        return blocks
    incoming = parents(blocks)
    used = set(params)
    for body in blocks.values():
        used.update(arg for _, args in body for arg in args if is_reg(arg))
    fresh = min(used, default=0) - 1
    result = dict(blocks)
    for label, (kind, x, counters, others, exit_label) in found:
        if ~fresh + 2 > 999999:
            break
        if set(others) & _live_in(blocks, outgoing, exit_label):
            continue
        a, z = fresh, fresh - 1
        fresh -= 2
        sites = _guards(result, incoming, label, x)
        code = []
        if kind == "clear" and sites is None:
            code.append(("eq", (z, x, 0)))
        code += _count(x, a, machine)
        if kind == "clear" and sites is None:
            code.append(("add", (x, x, z)))
        for reg, step in sorted(counters.items(), reverse=True):
            if step == 1:
                code.append(("add", (reg, reg, x)))
            elif step & (step - 1) == 0:
                code += [("shl", (a, x, step.bit_length() - 1)), ("add", (reg, reg, a))]
            else:
                code += [("mul", (a, x, step)), ("add", (reg, reg, a))]
        code += [("const", (x, 0)), ("jmp", (exit_label,))]
        result[label] = code
        # With x == 0 the new block adds nothing and leaves x at zero, and the
        # registers it clobbers are dead at the exit, so the test in front of
        # it can go.
        for site, toward in sites or ():
            result[site] = result[site][:-1] + [("jmp", (toward,))]
    return result
