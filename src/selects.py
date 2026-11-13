"""Turn selects between 0/1 values and counters into plain arithmetic.

If-conversion leaves chains such as

    z = eq x, 0
    e = eq b, 0
    n = add a, 1
    m = select e, n, a        ; a + e
    d = select z, a, m        ; a + (z == 0 and e == 1)

Facts. A register is boolean (0 or 1) at a point when its latest definition
in the block is `lt`, `eq`, `and r, 1`, a 0/1 constant or one of the 0/1 forms
below, or when every definition of it in the whole program is such an
instruction (or a copy or select of boolean values): registers start at 0, so
it then always holds 0 or 1. A register is small when it is boolean or its
latest definition is `and r, m` with a literal m below 2^31.

Rules, with c the condition:

* select(c, 1, 0) = c for boolean c; select(c, 0, 1) = eq c, 0;
* for boolean c and y: select(c, y, 0) = and c, y; select(c, 0, y) = lt c, y
  (c < y exactly when c = 0 and y = 1); select(c, 1, y) = or c, y;
* select(c, a + s, a) = a + select(c, s, 0) and
  select(c, a, a + s) = a + select(c, 0, s), used only when the inner select
  becomes one of the forms above (s literal 1 or boolean). Addition wraps, so
  these hold for every a;
* eq(eq(a, 0), 0) = lt 0, a for small a (a is then 0..2^31-1);
* and(x, eq(b, 0)) = lt b, x for boolean x and b;
* when none of the above applies, select(eq(b, 0), y, n) = select(b, n, y),
  and select(eq(b, 1), y, n) = select(b, y, n) for boolean b, looking
  through up to four such tests.

A definition is looked through only when none of its operands is written
again between it and the use. Every rule replaces a pure instruction by a
pure one with the same value, so nothing else changes; instructions that
become dead are removed by the later pruning.
"""
from .ir import is_reg, writes

BOOLEAN_OPS = ("lt", "eq")
SIGN = 1 << 31


def _always_boolean(blocks, params):
    """Registers whose every definition yields 0 or 1."""
    defs = {}
    for body in blocks.values():
        for op, args in body:
            if writes(op):
                defs.setdefault(args[0], []).append((op, args))
    known = {reg for reg in defs if reg not in params}
    changed = True
    while changed:
        changed = False
        for reg in list(known):
            for op, args in defs[reg]:
                if op in BOOLEAN_OPS or op == "and" and 1 in args[1:]:
                    continue
                if op == "const" and args[1] in (0, 1):
                    continue
                sources = args[1:] if op == "mov" else args[2:] if op == "select" else None
                if sources is not None and all(s in (0, 1) or is_reg(s) and s in known for s in sources):
                    continue
                known.discard(reg)
                changed = True
                break
    return known


def _rewrite_block(body, fresh, always):
    defs = {}        # register -> (index in output, kind, args)
    stamp = {}       # register -> index of its last write in output
    output = []

    def definition(register):
        if not is_reg(register) or register not in defs:
            return None
        index, op, args = defs[register]
        if any(is_reg(a) and stamp.get(a, -1) > index for a in args[1:]):
            return None
        return op, args

    def boolean(operand):
        # Only the register's own latest definition matters here: a later
        # write to one of its inputs does not change the value it holds.
        if not is_reg(operand):
            return operand in (0, 1)
        if operand in always:
            return True
        if operand not in defs:
            return False
        _, op, args = defs[operand]
        return op in BOOLEAN_OPS or op == "flag" or op == "and" and 1 in args[1:] or (
            op == "const" and args[1] in (0, 1))

    def small(operand):
        if boolean(operand):
            return True
        if not is_reg(operand) or operand not in defs:
            return False
        _, op, args = defs[operand]
        return op == "and" and any(not is_reg(a) and a < SIGN for a in args[1:])

    def literal(operand):
        if is_reg(operand) and operand in defs and defs[operand][1] == "const":
            return defs[operand][2][1]
        return operand

    def negated(register):
        """b when `register` holds eq(b, 0) and b is still in place."""
        found = definition(register)
        if found is None or found[0] != "eq":
            return None
        left, right = found[1][1:]
        if right == 0 and is_reg(left):
            return left
        if left == 0 and is_reg(right):
            return right
        return None

    def pick(dest, cond, yes, no):
        """One instruction for select(cond, yes, no) with 0/1 values, or None."""
        if not boolean(cond):
            return None
        if yes == 1 and no == 0:
            return ("mov", (dest, cond))
        if yes == 0 and no == 1:
            return ("eq", (dest, cond, 0))
        if is_reg(yes) and boolean(yes) and no == 0:
            return ("and", (dest, cond, yes))
        if yes == 0 and is_reg(no) and boolean(no):
            return ("lt", (dest, cond, no))
        if yes == 1 and is_reg(no) and boolean(no):
            return ("or", (dest, cond, no))
        return None

    def emit_pick(dest, cond, yes, no):
        """Emit select(cond, yes, no) as 0/1 arithmetic; False if no form fits.

        For cond = eq(a, 0) with small a, select(cond, 0, y) is (a != 0) and y,
        written as lt t, 0, a; and dest, t, y so that the test on a is
        shared with other uses and the `and` rule below can apply.
        """
        single = pick(dest, cond, yes, no)
        if single is None:
            return False
        if single[0] == "lt" and yes == 0:
            inner = negated(cond)
            if inner is not None and small(inner):
                temp = fresh[0]
                fresh[0] -= 1
                emit("lt", (temp, 0, inner), True)
                emit("and", (dest, temp, no), True)
                return True
        emit(*single, flag=True)
        return True

    def emit(op, args, flag=False):
        if op == "eq" and 0 in args[1:]:
            inner = negated(args[1] if args[2] == 0 else args[2])
            if inner is not None and small(inner):
                op, args, flag = "lt", (args[0], 0, inner), True
        elif op == "and" and len(args) == 3:
            x, y = args[1:]
            for value, other in ((x, y), (y, x)):
                inner = negated(other) if is_reg(other) else None
                if inner is not None and boolean(inner) and boolean(value) and is_reg(value):
                    op, args, flag = "lt", (args[0], inner, value), True
                    break
        output.append((op, args))
        if writes(op):
            index = len(output) - 1
            stamp[args[0]] = index
            defs[args[0]] = (index, "flag" if flag else op, args)

    for op, args in body:
        if op == "select":
            dest, cond, yes, no = args
            # A test of the form eq(b, 0), or eq(b, 1) for boolean b, can
            # be dropped by testing b itself (swapping the arms for eq 0).
            # Used only when none of the 0/1 rules below applies.
            direct, flip = cond, False
            for _ in range(4):
                found = definition(direct)
                if found is None or found[0] != "eq":
                    break
                left, right = found[1][1:]
                inner, other = (left, right) if is_reg(left) else (right, left)
                if not is_reg(inner) or is_reg(other) or other not in (0, 1):
                    break
                if other == 1 and not boolean(inner):
                    break
                direct = inner
                flip ^= other == 0
            direct = (dest, direct, no, yes) if flip else (dest, direct, yes, no)
            yes, no = literal(yes), literal(no)
            if not is_reg(cond) or yes == no:
                emit(op, args)
                continue
            if yes == 0 and no == 1:
                emit("eq", (dest, cond, 0), True)
                continue
            if emit_pick(dest, cond, yes, no):
                continue
            done = False
            for base, other, mirrored in ((no, yes, False), (yes, no, True)):
                found = definition(other)
                if not is_reg(base) or found is None or found[0] != "add":
                    continue
                left, right = found[1][1:]
                if left == base:
                    step = right
                elif right == base:
                    step = left
                else:
                    continue
                if stamp.get(base, -1) > defs[other][0]:
                    continue
                if not (step == 1 or is_reg(step) and boolean(step)):
                    continue
                yes_part, no_part = (0, step) if mirrored else (step, 0)
                inner = pick(fresh[0], cond, yes_part, no_part)
                if inner is None:
                    continue
                if inner[0] == "mov":
                    emit("add", (dest, base, cond))
                else:
                    temp = fresh[0]
                    fresh[0] -= 1
                    emit_pick(temp, cond, yes_part, no_part)
                    emit("add", (dest, base, temp))
                done = True
                break
            if done:
                continue
            args = direct
        emit(op, args)
    return output


def simplify_selects(blocks, params):
    if not any(op == "select" for body in blocks.values() for op, _ in body):
        return blocks
    used = set(params)
    for body in blocks.values():
        used.update(arg for _, args in body for arg in args if is_reg(arg))
    fresh = [min(used, default=0) - 1]
    if ~fresh[0] > 999000:
        return blocks
    always = _always_boolean(blocks, params)
    return {label: _rewrite_block(body, fresh, always) for label, body in blocks.items()}
