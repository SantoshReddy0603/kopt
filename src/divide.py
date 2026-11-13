"""Signed division and remainder by a literal power of two, without div.

`div d, x, 2^k` rounds toward zero, so for negative x it is the arithmetic
shift plus one when any of the low k bits is set. KIR has no arithmetic
shift; with m = 0 - (x >> 31) (all ones for negative x, else zero) it is
((x xor m) >> k) xor m. The remainder keeps the sign of x:
x mod 2^k = l - 2^k when x < 0 and l = x & (2^k - 1) is not zero, else l.
When x is known to be below 2^31 (its latest definition in the block is an
`and` with such a literal, a logical right shift by at least one, or a test)
the forms reduce to one shift or one `and`.

The divisor is a nonzero literal, so neither instruction can trap and the
rewrite only exchanges pure instructions for pure instructions with the same
value. It is applied when the machine's divide latency is above the depth of
the replacement.
"""

from .ir import is_reg, writes

SIGN = 1 << 31


def _nonnegative(defs, register):
    found = defs.get(register)
    if found is None:
        return False
    # The value is fixed when written; later writes to its inputs do not matter.
    _, op, args = found
    if op in ("lt", "eq"):
        return True
    if op == "and":
        return any(not is_reg(a) and a < SIGN for a in args[1:])
    if op == "shr":
        return not is_reg(args[2]) and args[2] & 31 >= 1
    if op == "const":
        return args[1] < SIGN
    return False


def expand(blocks, params, machine):
    lat_alu, lat_div = machine["lat_alu"], machine["lat_div"]
    if lat_div <= lat_alu:
        return blocks
    if not any(op in ("div", "mod") and not is_reg(args[2]) for body in blocks.values() for op, args in body):
        return blocks
    used = set(params)
    for body in blocks.values():
        used.update(arg for _, args in body for arg in args if is_reg(arg))
    fresh = [min(used, default=0) - 1]
    if ~fresh[0] > 999990:
        return blocks

    def new():
        fresh[0] -= 1
        return fresh[0] + 1

    changed = False
    result = {}
    for label, body in blocks.items():
        defs, stamp, output = {}, {}, []
        for op, args in body:
            code = None
            if op in ("div", "mod") and not is_reg(args[2]) and is_reg(args[1]):
                dest, x, divisor = args
                if 2 <= divisor < SIGN and divisor & (divisor - 1) == 0:
                    k = divisor.bit_length() - 1
                    if _nonnegative(defs, x):
                        code = [("shr", (dest, x, k))] if op == "div" else [("and", (dest, x, divisor - 1))]
                    elif op == "div" and lat_div > 6 * lat_alu and ~fresh[0] + 4 <= 999999:
                        s, m, z, low = new(), new(), new(), new()
                        code = [("shr", (s, x, 31)), ("sub", (m, 0, s)), ("and", (low, x, divisor - 1)),
                                ("xor", (z, x, m)), ("lt", (low, 0, low)), ("shr", (z, z, k)),
                                ("and", (low, low, s)), ("xor", (z, z, m)), ("add", (dest, z, low))]
                    elif op == "mod" and lat_div > 4 * lat_alu and ~fresh[0] + 3 <= 999999:
                        s, low, c = new(), new(), new()
                        code = [("shr", (s, x, 31)), ("and", (low, x, divisor - 1)), ("lt", (c, 0, low)),
                                ("and", (c, c, s)), ("shl", (c, c, k)), ("sub", (dest, low, c))]
            for item in code or [(op, args)]:
                output.append(item)
                if writes(item[0]):
                    stamp[item[1][0]] = len(output) - 1
                    defs[item[1][0]] = (len(output) - 1, item[0], item[1])
            changed = changed or code is not None
        result[label] = output
    return result if changed else blocks
