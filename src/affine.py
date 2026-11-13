"""Straight-line bit-affine functions rebuilt from their input bits.

A program that is one block of xor, shifts by literals, masks and ors with
literals, 0/1 tests and selects on a single input-dependent bit computes
every result bit as an xor (GF(2) sum) of input bits plus a constant; CRC
steps and parity folds are typical. Each register is tracked as 32 such
forms, kept as bit sets: bit 0 is the constant, bit i+1 is input bit i. An
instruction outside that algebra (a product of two input-dependent bits, a
carry, a memory access, ...) stops the analysis, so the forms are exact for
every input whenever they are produced.

The result is then emitted as C xor the sum, over input bits v, of
(v ? M_v : 0), where M_v is the set of result bits that contain v. Input
bits whose M_v is a single bit at the same distance from their own position
are gathered into one shift and mask; the rest take an and and a select.
The terms are joined by a balanced xor tree, so the depth is logarithmic
instead of one step per original bit operation.
"""

from .ir import is_reg
from .kirlib import MASK

BLOCK_LIMIT = 1500
TERM_LIMIT = 160


class _Unsupported(Exception):
    pass


def _literal(value):
    return [(value >> bit) & 1 for bit in range(32)]


def _single(form):
    """(position, constant bits) when at most one bit of form is input dependent."""
    position = None
    for bit, item in enumerate(form):
        if item > 1:
            if position is not None:
                raise _Unsupported
            position = bit
    return position


def _analyse(body, params):
    names = {}
    order = []

    def form(arg):
        if not is_reg(arg):
            return _literal(arg & MASK)
        if arg not in env:
            return [0] * 32
        return env[arg]

    env = {}
    for param in params:
        index = len(order)
        order.append(param)
        names[param] = index
        env[param] = [1 << (index * 32 + bit + 1) for bit in range(32)]
    for op, args in body[:-1]:
        if op == "const":
            env[args[0]] = _literal(args[1] & MASK)
        elif op == "mov":
            env[args[0]] = list(form(args[1]))
        elif op == "xor":
            left, right = form(args[1]), form(args[2])
            env[args[0]] = [a ^ b for a, b in zip(left, right)]
        elif op in ("and", "or"):
            left, right = form(args[1]), form(args[2])
            out = []
            for a, b in zip(left, right):
                if a > 1 and b > 1:
                    raise _Unsupported
                if a <= 1:
                    a, b = b, a
                # b is constant here
                if op == "and":
                    out.append(a if b else 0)
                else:
                    out.append(1 if b else a)
            env[args[0]] = out
        elif op in ("shl", "shr"):
            if is_reg(args[2]):
                shift_form = form(args[2])
                if any(item > 1 for item in shift_form):
                    raise _Unsupported
                amount = sum(bit << i for i, bit in enumerate(shift_form)) & 31
            else:
                amount = args[2] & 31
            source = form(args[1])
            if op == "shl":
                env[args[0]] = [0] * amount + source[:32 - amount]
            else:
                env[args[0]] = source[amount:] + [0] * amount
        elif op == "eq":
            left, right = form(args[1]), form(args[2])
            diff = [a ^ b for a, b in zip(left, right)]
            position = _single(diff)
            constants = [item for bit, item in enumerate(diff) if bit != position]
            if any(constants):
                env[args[0]] = [0] * 32
            elif position is None:
                env[args[0]] = _literal(1)
            else:
                env[args[0]] = [diff[position] ^ 1] + [0] * 31
        elif op == "select":
            cond = form(args[1])
            yes, no = form(args[2]), form(args[3])
            position = _single(cond)
            others = [item for bit, item in enumerate(cond) if bit != position]
            if any(others):
                env[args[0]] = list(yes)
            elif position is None:
                env[args[0]] = list(no)
            else:
                test = cond[position]
                out = []
                for a, b in zip(yes, no):
                    d = a ^ b
                    if d > 1:
                        raise _Unsupported
                    out.append(b ^ test if d else b)
                env[args[0]] = out
        else:
            raise _Unsupported
    return form(body[-1][1][0]), order


def synthesize(blocks, params):
    """An equivalent single block built from the affine forms, or None."""
    if len(blocks) != 1:
        return None
    label, body = next(iter(blocks.items()))
    if len(body) > BLOCK_LIMIT or body[-1][0] != "ret" or not is_reg(body[-1][1][0]):
        return None
    try:
        result, order = _analyse(body, params)
    except _Unsupported:
        return None
    constant = sum((item & 1) << bit for bit, item in enumerate(result))
    masks = {}
    for bit, item in enumerate(result):
        item >>= 1
        while item:
            low = item & -item
            variable = low.bit_length() - 1
            masks[variable] = masks.get(variable, 0) | (1 << bit)
            item ^= low
    groups = {}
    singles = []
    for variable in sorted(masks):
        param, bit = divmod(variable, 32)
        mask = masks[variable]
        if mask & (mask - 1) == 0:
            groups.setdefault((param, mask.bit_length() - 1 - bit), []).append(bit)
        else:
            singles.append((param, bit, mask))
    if len(groups) + len(singles) > TERM_LIMIT:
        return None
    used = set(params) | {arg for _, args in body for arg in args if is_reg(arg)}
    fresh = [min(used, default=0) - 1]
    if ~fresh[0] + 2 * TERM_LIMIT > 999999:
        return None

    def new():
        fresh[0] -= 1
        return fresh[0] + 1

    code = []
    stack = []

    def push(term):
        # Pairwise tree, emitted like a binary counter right after each term
        # so that only about log2(terms) partial sums are alive at once.
        level = 0
        while stack and stack[-1][1] == level:
            other, _ = stack.pop()
            code.append(("xor", (other, other, term)))
            term = other
            level += 1
        stack.append((term, level))

    for (param, shift), bits in sorted(groups.items()):
        mask = 0
        for bit in bits:
            mask |= 1 << (bit + shift)
        source = order[param]
        temp = new()
        if shift:
            code.append(("shl" if shift > 0 else "shr", (temp, source, abs(shift))))
            source = temp
        if mask == MASK:
            if source == order[param]:
                code.append(("mov", (temp, source)))
        else:
            code.append(("and", (temp, source, mask)))
        push(temp)
    for index, (param, bit, mask) in enumerate(singles):
        temp = new()
        code.append(("and", (temp, order[param], 1 << bit)))
        if constant and index == len(singles) - 1:
            code.append(("select", (temp, temp, mask ^ constant, constant)))
            constant = 0
        else:
            code.append(("select", (temp, temp, mask, 0)))
        push(temp)
    while len(stack) > 1:
        term, _ = stack.pop()
        other, level = stack.pop()
        code.append(("xor", (other, other, term)))
        stack.append((other, level + 1))
    if stack:
        out = stack[0][0]
        if constant:
            code.append(("xor", (out, out, constant)))
    else:
        out = new()
        code.append(("const", (out, constant)))
    code.append(("ret", (out,)))
    if len(code) >= len(body):
        return None
    return {label: code}
