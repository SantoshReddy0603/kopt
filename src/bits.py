"""Closed forms for loops that move, combine and count parameter bits.

The original program is run once with every parameter bit kept symbolic.
A value is either a concrete word, a bit vector (each of the 32 positions is a
constant or the XOR of some parameter bits, plus a constant), or a sum (an
integer combination, modulo 2^32, of such XOR terms). Shifts and masks by
known amounts, xor, and/or with a constant side, and additions or constant
multiples of sums are exact in this domain. Branch conditions, comparisons,
addresses, divisors and every other operation must see concrete words, so the
path taken is the same for all inputs and the final value is exact for all of
them. Memory and volatile instructions are not allowed.

The result is emitted as masked shifts of parameters (or a staged bit
permutation when output bit j is input bit j xor c), and sums as multiples of
SWAR population counts.
"""
from .kirlib import (ADD, ADDSH, AND, BR, DIV, EQ, JMP, LT, MAC, MASK, MOD, MOV, MUL, OR,
                     RET, SELECT, SHL, SHR, SIGN, SUB, XOR, compile_program)

STEPS = 20000
MAX_TERMS = 10


class Reject(Exception):
    pass


class Vec:
    """32 positions, each (mask over parameter bits, constant bit)."""
    __slots__ = ("bits",)

    def __init__(self, bits):
        self.bits = bits


class Sum:
    """const + sum(coef * parity(mask)) modulo 2^32."""
    __slots__ = ("terms", "const")

    def __init__(self, terms, const):
        self.terms, self.const = terms, const & MASK


def _settle(value):
    if isinstance(value, Vec):
        if all(m == 0 for m, _ in value.bits):
            return sum(c << j for j, (_, c) in enumerate(value.bits))
        return value
    if isinstance(value, Sum):
        terms = {m: c for m, c in value.terms.items() if c}
        if not terms:
            return value.const
        if len(terms) > 32 * MAX_TERMS:
            raise Reject("too many terms")
        return Sum(terms, value.const)
    return value


def _vec(value):
    if isinstance(value, Vec):
        return value
    if isinstance(value, int):
        return Vec(tuple((0, value >> j & 1) for j in range(32)))
    raise Reject("sum used as a bit vector")


def _sum(value):
    if isinstance(value, Sum):
        return value
    if isinstance(value, int):
        return Sum({}, value)
    terms, const = {}, 0
    for j, (m, c) in enumerate(value.bits):
        weight = 1 << j
        if m == 0:
            const += c * weight
        elif c == 0:
            terms[m] = (terms.get(m, 0) + weight) & MASK
        else:
            const += weight
            terms[m] = (terms.get(m, 0) - weight) & MASK
    return Sum(terms, const)


def _concrete(value):
    if not isinstance(value, int):
        raise Reject("parameter bits reach a non-linear use")
    return value


def _add(x, y, sign=1):
    if isinstance(x, int) and isinstance(y, int):
        return (x + sign * y) & MASK
    if sign == 1 and isinstance(y, int) and y == 0:
        return x
    if isinstance(x, int) and x == 0 and sign == 1:
        return y
    a, b = _sum(x), _sum(y)
    terms = dict(a.terms)
    for m, c in b.terms.items():
        terms[m] = (terms.get(m, 0) + sign * c) & MASK
    return _settle(Sum(terms, a.const + sign * b.const))


def _scale(x, k):
    k &= MASK
    if isinstance(x, int):
        return (x * k) & MASK
    a = _sum(x)
    return _settle(Sum({m: (c * k) & MASK for m, c in a.terms.items()}, a.const * k))


def _bitwise(op, x, y):
    if isinstance(x, int) and isinstance(y, int):
        return x & y if op == AND else x | y if op == OR else x ^ y
    a, b = _vec(x).bits, _vec(y).bits
    out = []
    for (m1, c1), (m2, c2) in zip(a, b):
        if op == XOR:
            out.append((m1 ^ m2, c1 ^ c2))
            continue
        if m1 and m2:
            raise Reject("two symbolic bits combined by and/or")
        if m2:
            (m1, c1), (m2, c2) = (m2, c2), (m1, c1)
        if op == AND:
            out.append((m1, c1) if c2 else (0, 0))
        else:
            out.append((0, 1) if c2 else (m1, c1))
    return _settle(Vec(tuple(out)))


def _shift(x, count, left):
    count &= 31
    if isinstance(x, int):
        return (x << count) & MASK if left else x >> count
    if isinstance(x, Sum):
        if left:
            return _scale(x, 1 << count)
        raise Reject("right shift of a sum")
    bits = x.bits
    zero = (0, 0)
    if left:
        out = (zero,) * count + bits[:32 - count]
    else:
        out = bits[count:] + (zero,) * count
    return _settle(Vec(out))


def evaluate(compiled, nparams):
    R = list(compiled.R0)
    for p in range(nparams):
        R[p] = Vec(tuple((1 << (p * 32 + j), 0) for j in range(32)))
    code = compiled.code
    pc = steps = 0
    while True:
        steps += 1
        if steps > STEPS:
            raise Reject("runs too long")
        op, d, a, b, c, t1, t2 = code[pc]
        pc += 1
        if op == MOV:
            R[d] = R[a]
        elif op == ADD:
            R[d] = _add(R[a], R[b])
        elif op == SUB:
            R[d] = _add(R[a], R[b], -1)
        elif op == MUL:
            x, y = R[a], R[b]
            if isinstance(y, int):
                R[d] = _scale(x, y)
            elif isinstance(x, int):
                R[d] = _scale(y, x)
            else:
                raise Reject("product of symbolic values")
        elif op in (AND, OR, XOR):
            R[d] = _bitwise(op, R[a], R[b])
        elif op == SHL or op == SHR:
            R[d] = _shift(R[a], _concrete(R[b]), op == SHL)
        elif op == ADDSH:
            R[d] = _add(R[a], _shift(R[b], c, True))
        elif op == MAC:
            x, y = R[a], R[b]
            if isinstance(y, int):
                R[d] = _add(_scale(x, y), R[c])
            elif isinstance(x, int):
                R[d] = _add(_scale(y, x), R[c])
            else:
                raise Reject("product of symbolic values")
        elif op == SELECT:
            R[d] = R[b] if _concrete(R[a]) else R[c]
        elif op == BR:
            pc = t1 if _concrete(R[a]) else t2
        elif op == JMP:
            pc = t1
        elif op == RET:
            return R[a]
        elif op in (LT, EQ, DIV, MOD):
            x, y = _concrete(R[a]), _concrete(R[b])
            if op == LT:
                R[d] = 1 if x ^ SIGN < y ^ SIGN else 0
            elif op == EQ:
                R[d] = 1 if x == y else 0
            else:
                if y == 0:
                    raise Reject("division by zero")
                sx = x - (1 << 32) if x & SIGN else x
                sy = y - (1 << 32) if y & SIGN else y
                q = abs(sx) // abs(sy) if op == DIV else abs(sx) % abs(sy)
                if (op == DIV and (sx < 0) != (sy < 0)) or (op == MOD and sx < 0):
                    q = -q
                R[d] = q & MASK
        else:
            raise Reject("memory, volatile or trap instruction")


class Out:
    def __init__(self, params):
        self.params = params
        self.code = []
        self.next = max((~p for p in params), default=-1) + 1

    def reg(self):
        self.next += 1
        return ~(self.next - 1)

    def emit(self, op, *args):
        out = self.reg()
        self.code.append((op, (out,) + args))
        return out


def _xor_of(out, params):
    reg = None
    for p in params:
        reg = out.params[p] if reg is None else out.emit("xor", reg, out.params[p])
    return reg


def _vector(out, vec):
    """Emit a bit vector; return a register or literal."""
    const = sum(c << j for j, (_, c) in enumerate(vec.bits))
    terms = {}
    for j, (m, _) in enumerate(vec.bits):
        while m:
            low = m & -m
            m ^= low
            index = low.bit_length() - 1
            p, bit = divmod(index, 32)
            key = (p, j - bit)
            terms[key] = terms.get(key, 0) | (1 << j)
    staged = _staged(out, vec, const)
    if staged is not None:
        return staged
    if len(terms) > MAX_TERMS:
        raise Reject("too many shifted copies")
    reg = None
    for (p, distance), mask in sorted(terms.items()):
        part = out.params[p]
        if distance > 0:
            part = out.emit("shl", part, distance)
        elif distance < 0:
            part = out.emit("shr", part, -distance)
        if mask != MASK and (distance <= 0 or mask != (MASK << distance) & MASK) and (
                distance >= 0 or mask != MASK >> -distance):
            part = out.emit("and", part, mask)
        reg = part if reg is None else out.emit("xor", reg, part)
    if const:
        reg = out.emit("xor", reg, const)
    return reg


def _staged(out, vec, const):
    """Output bit j is bit j^c of one parameter (or a constant): swap stages."""
    source, flip, mask = None, None, 0
    for j, (m, _) in enumerate(vec.bits):
        if not m:
            continue
        if m & (m - 1):
            return None
        index = m.bit_length() - 1
        p, bit = divmod(index, 32)
        if source not in (None, p) or flip not in (None, j ^ bit):
            return None
        source, flip, mask = p, j ^ bit, mask | (1 << j)
    if source is None or flip == 0 or bin(flip).count("1") < 2:
        return None
    if any(c and m for m, c in vec.bits):
        return None
    reg = out.params[source]
    for level in (16, 8, 4, 2, 1):
        if not flip & level:
            continue
        low = 0
        for j in range(32):
            if not j & level:
                low |= 1 << j
        right = out.emit("shr", reg, level)
        if level < 16:
            right = out.emit("and", right, low)
        left = reg if level == 16 else out.emit("and", reg, low)
        left = out.emit("shl", left, level)
        reg = out.emit("or", left, right)
    if mask != MASK:
        reg = out.emit("and", reg, mask)
    if const:
        reg = out.emit("or", reg, const)
    return reg


def _popcount(out, reg, width):
    x = reg
    if width > 1:
        t = out.emit("shr", x, 1)
        t = out.emit("and", t, 0x55555555)
        x = out.emit("sub", x, t)
    if width > 2:
        t = out.emit("shr", x, 2)
        t = out.emit("and", t, 0x33333333)
        x = out.emit("and", x, 0x33333333)
        x = out.emit("add", x, t)
    if width > 4:
        t = out.emit("shr", x, 4)
        x = out.emit("add", x, t)
        x = out.emit("and", x, 0x0F0F0F0F)
    if width > 8:
        x = out.emit("mul", x, 0x01010101)
        x = out.emit("shr", x, 24)
    return x


def _total(out, value):
    """Emit a sum; return a register or literal."""
    groups = {}
    for m, coef in value.terms.items():
        positions = set()
        members = set()
        rest = m
        while rest:
            low = rest & -rest
            rest ^= low
            p, bit = divmod(low.bit_length() - 1, 32)
            positions.add(bit)
            members.add(p)
        if len(positions) != 1 or bin(m).count("1") != len(members):
            raise Reject("term mixes bit positions")
        groups.setdefault(tuple(sorted(members)), {})[positions.pop()] = coef
    if len(groups) > 3:
        raise Reject("too many groups")
    parts = []
    for members, coefs in sorted(groups.items()):
        word = _xor_of(out, members)
        positions = sorted(coefs)
        select = sum(1 << j for j in positions)
        shifted = None
        low = positions[0]
        base = coefs[low]
        if all(coefs[j] == (base << (j - low)) & MASK for j in positions):
            shifted = base
        if len(set(coefs.values())) == 1 and len(positions) > 1:
            masked = word if select == MASK else out.emit("and", word, select)
            if positions[-1] - positions[0] < 32 and positions[0] > 0 and select != MASK:
                masked = out.emit("shr", masked, positions[0])
                span = positions[-1] - positions[0] + 1
            else:
                span = positions[-1] + 1
            width = 1
            while width < span:
                width *= 2
            count = _popcount(out, masked, width)
            coef = coefs[positions[0]]
            parts.append(count if coef == 1 else out.emit("mul", count, coef))
        elif shifted is not None:
            masked = word if select == MASK else out.emit("and", word, select)
            if low:
                masked = out.emit("shr", masked, low)
            parts.append(masked if base == 1 else out.emit("mul", masked, base))
        else:
            raise Reject("irregular weights")
    reg = None
    for part in parts:
        reg = part if reg is None else out.emit("add", reg, part)
    if value.const:
        reg = out.emit("add", reg, value.const)
    return reg


def specialize(program, params):
    """Return straight-line blocks equal to the program, or None."""
    if not params or len(program.instrs) > 2000:
        return None
    try:
        value = evaluate(compile_program(program), len(params))
        out = Out(params)
        if isinstance(value, int):
            result = value
        elif isinstance(value, Vec):
            result = _vector(out, value)
        else:
            result = _total(out, value)
    except Reject:
        return None
    return {"entry": out.code + [("ret", (result,))]}
