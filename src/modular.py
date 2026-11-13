"""Deferred remainders in counted self loops.

A loop such as

    a = (a + byte) % M
    b = (b + a) % M

computes the same remainders when the `mod`s are left out of the loop and
applied once after it, as long as no sum can reach 2^31. Addition keeps
values congruent modulo M, and for values in 0..2^31-1 the signed `mod` of
KIR is the ordinary remainder.

The pass treats each `mod d, t, M` with one literal M as a copy and checks,
by running the body on upper bounds for the largest possible number of
trips, that

* every value that may now be unreduced ("congruent") is only copied,
  added to another congruent value or to a value with a known non-negative
  bound, or fed into one of the removed `mod`s;
* every such sum stays below 2^31;
* the values entering the loop are known constants below M.

The remainders are then taken on a new block on the exit edge, for every
congruent register that is live there. A literal M > 0 cannot trap, so
removing the division changes no trap.
"""
from .flow import live_out
from .ir import is_reg, parents, reads, writes
from .kirlib import SIGN
from .ranges import analyze

MAX_TRIPS = 4096
MAX_BODY = 64


def _counter(body):
    """(counter, step, limit) for `add i, i, s; ...; lt c, i, n; br c, self, exit`."""
    if len(body) < 3 or body[-2][0] != "lt" or body[-1][0] != "br":
        return None
    cond, counter, limit = body[-2][1]
    if body[-1][1][0] != cond or not is_reg(counter):
        return None
    updates = [args for op, args in body[:-1] if writes(op) and args[0] == counter]
    if len(updates) != 1:
        return None
    index = [args for _, args in body].index(updates[0])
    op, args = body[index]
    if op != "add" or args[1] != counter or is_reg(args[2]) or not 0 < args[2] < 1 << 16:
        return None
    if is_reg(limit) and any(writes(o) and a[0] == limit for o, a in body[:-1]):
        return None
    return counter, args[2], limit


def _initial(blocks, label, predecessor, register, incoming):
    """Constant held by `register` when the loop is entered, else None."""
    seen = set()
    current = predecessor
    while current not in seen:
        seen.add(current)
        for op, args in reversed(blocks[current]):
            if writes(op) and args[0] == register:
                if op == "const":
                    return args[1]
                return None
        preds = [p for p in incoming[current] if p != label]
        if len(preds) != 1:
            return None
        current = preds[0]
    return None


def _bound(op, args, state, exact):
    """Upper bound of a non-congruent value written by (op, args), or None."""
    def of(arg):
        if not is_reg(arg):
            return arg if arg < SIGN else None
        return exact.get(arg)
    if op == "const":
        return of(args[1])
    if op == "mov":
        return of(args[1])
    if op == "and":
        bounds = [of(a) for a in args[1:]]
        known = [b for b in bounds if b is not None]
        return min(known) if known else None
    if op == "shr" and not is_reg(args[2]):
        value = of(args[1])
        return value >> (args[2] & 31) if value is not None else None
    if op == "add":
        x, y = of(args[1]), of(args[2])
        return x + y if x is not None and y is not None and x + y < SIGN else None
    return None


def _simulate(body, modulus, removed, carried, starts, trips):
    """Bounds of congruent values after every trip count up to `trips`, or None."""
    congruent = {register: value for register, value in starts.items()}
    for _ in range(trips):
        exact = {}
        current = dict(congruent)
        for index, (op, args) in enumerate(body[:-1]):
            sources = list(reads(op, args))
            if index in removed:
                source = args[1]
                if source in current:
                    current[args[0]] = current[source]
                elif not is_reg(source) or source in exact:
                    value = source if not is_reg(source) else exact[source]
                    if value is None:
                        return None
                    current[args[0]] = value
                else:
                    return None
                exact.pop(args[0], None)
                continue
            if any(s in current for s in sources):
                if op == "mov":
                    current[args[0]] = current[args[1]]
                elif op == "add":
                    total = 0
                    for arg in args[1:]:
                        if arg in current:
                            total += current[arg]
                        elif not is_reg(arg):
                            if arg >= SIGN:
                                return None
                            total += arg
                        elif exact.get(arg) is not None:
                            total += exact[arg]
                        else:
                            return None
                    if total >= SIGN:
                        return None
                    current[args[0]] = total
                else:
                    return None
                exact.pop(args[0], None)
                continue
            if writes(op):
                current.pop(args[0], None)
                exact[args[0]] = _bound(op, args, current, exact)
        congruent = {register: current[register] for register in carried if register in current}
        if set(congruent) != set(carried):
            return None
    return congruent


def defer(blocks):
    """Return blocks with deferred remainders (or the blocks unchanged)."""
    if sum(map(len, blocks.values())) > 4000:
        return blocks
    incoming = parents(blocks)
    states = None
    live = None
    result = dict(blocks)
    added = {}
    for label, body in blocks.items():
        if not 3 <= len(body) <= MAX_BODY or len(incoming[label]) != 2 or label not in incoming[label]:
            continue
        op, args = body[-1]
        if op != "br" or args[1] != label or args[2] == label:
            continue
        exit_label = args[2]
        moduli = [a[2] for o, a in body[:-1] if o == "mod" and not is_reg(a[2]) and 2 <= a[2] < SIGN]
        if not moduli:
            continue
        modulus = max(set(moduli), key=moduli.count)
        removed = {i for i, (o, a) in enumerate(body[:-1]) if o == "mod" and a[2] == modulus}
        shape = _counter(body)
        if shape is None:
            continue
        counter, step, limit = shape
        predecessor = next(p for p in incoming[label] if p != label)
        start = _initial(blocks, label, predecessor, counter, incoming)
        if start is None or start >= SIGN:
            continue
        if is_reg(limit):
            if states is None:
                states = analyze(blocks) or {}
            if label not in states or limit not in states[label]:
                continue
            top = states[label][limit][1]
        else:
            top = limit
        if top >= SIGN:
            continue
        # The body runs at least once (the test is at its end), and the
        # counter must not wrap before it reaches the limit.
        trips = max(1, -(-(top - start) // step))
        if trips > MAX_TRIPS or trips * len(body) > 200000 or start + (trips + 1) * step >= SIGN:
            continue
        written = [a[0] for o, a in body[:-1] if writes(o)]
        first_read = set()
        defined = set()
        for o, a in body[:-1]:
            first_read.update(r for r in reads(o, a) if r not in defined)
            if writes(o):
                defined.add(a[0])
        # Registers that may hold unreduced values: written by a removed mod or
        # by a copy/add chain from one; found by running one trip symbolically.
        tainted = set()
        for index, (o, a) in enumerate(body[:-1]):
            if index in removed or (o in ("mov", "add") and any(r in tainted for r in reads(o, a))):
                tainted.add(a[0])
            elif writes(o):
                tainted.discard(a[0])
        carried = [r for r in dict.fromkeys(written) if r in tainted and r in first_read]
        starts = {}
        for register in carried:
            value = _initial(blocks, label, predecessor, register, incoming)
            if value is None or not 0 <= value < modulus:
                starts = None
                break
            starts[register] = value
        if starts is None:
            continue
        if _simulate(body, modulus, removed, carried, starts, trips) is None:
            continue
        if live is None:
            live = live_out(blocks)
            if live is None:
                return blocks
        exit_body = blocks[exit_label]
        used, defs = set(), set()
        for o, a in exit_body:
            used.update(r for r in reads(o, a) if r not in defs)
            if writes(o):
                defs.add(a[0])
        live_in = used | (live[exit_label] - defs)
        fix = [r for r in dict.fromkeys(written) if r in tainted and r in live_in]
        if any(r not in carried for r in fix):
            continue
        new_body = [("mov", (a[0], a[1])) if i in removed else (o, a) for i, (o, a) in enumerate(body)]
        name = label + "_mod"
        if name in blocks or name in added:
            continue
        new_body[-1] = ("br", (args[0], label, name))
        result[label] = new_body
        added[label] = (name, [("mod", (r, r, modulus)) for r in fix] + [("jmp", (exit_label,))])
    if not added:
        return blocks
    ordered = {}
    for label, body in result.items():
        ordered[label] = body
        if label in added:
            name, fixup = added[label]
            ordered[name] = fixup
    return ordered
