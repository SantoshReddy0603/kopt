"""Replace a function whose result depends on a few input bits by a lookup.

Proof. A forward dataflow over the original program tracks which registers may
still hold a raw parameter and which registers certainly hold a constant (every
register except the parameters starts at zero). Every read of a raw parameter
must be `and` with a constant mask, so only the bits in the union of those masks
can influence anything. Memory and volatile instructions reject, so the return
value is the only observable result.

Evaluation. The original program is interpreted for every value of the masked
bits. Any trap or timeout, or more than the total step budget, rejects. The
interpreter is the organizer's model, so the table is exactly what the input
program returns for each class of inputs.

Emission. The masked bits are packed into a key. Results are stored relative to
a base in power-of-two fields of 32-bit literals. A shift count's low five bits
pick the field inside a word, and `select` trees or key-bit branches pick the
word. Bit-test trees skip the key: each key bit is tested on its parameter,
an internal node is one `br`, the lowest bits may be picked by `select` over
literals, and every leaf returns a literal. Several shapes are produced; the
caller measures them, on the argument sets from `spread` when the key range
is small enough to cover.
"""
from .ir import is_reg, writes
from .kirlib import MASK, compile_program, execute, time_trace

MAX_BITS = 10
MAX_INSTRUCTIONS = 4000
RUN_STEPS = 40000
TOTAL_STEPS = 1500000
MEMORY_OPS = ("load", "store", "vload", "vstore")
FOLD = {
    "add": lambda a, b: (a + b) & MASK,
    "sub": lambda a, b: (a - b) & MASK,
    "mul": lambda a, b: (a * b) & MASK,
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
    "shl": lambda a, b: (a << (b & 31)) & MASK,
    "shr": lambda a, b: a >> (b & 31),
}


POLY_OPS = ("add", "sub", "mul", "shl", "mac", "addsh", "select", "ret")
MAX_TERMS = 12
MAX_DEGREE = 4


class Reject(Exception):
    pass


def _value(arg, const):
    if not is_reg(arg):
        return arg & MASK
    return const.get(arg)


def _step(op, args, raw, const, masks):
    """Apply one instruction to (raw, const); masks collects proven masks."""
    dest = args[0] if writes(op) else None
    sources = args[1:] if writes(op) else args
    tainted = [arg for arg in sources if is_reg(arg) and raw.get(arg)]
    new_raw = frozenset()
    if tainted:
        if op == "mov":
            new_raw = raw[sources[0]]
        elif op == "and" and len(tainted) == 1 and sources[0] != sources[1]:
            other = sources[1] if sources[0] == tainted[0] else sources[0]
            mask = _value(other, const)
            if mask is None:
                raise Reject("parameter combined with a non-constant")
            if masks is not None:
                for index in raw[tainted[0]]:
                    masks[index] |= mask
        elif op not in POLY_OPS:
            raise Reject("raw parameter read by %s" % op)
    if dest is None:
        return
    if new_raw:
        raw[dest] = new_raw
    else:
        raw.pop(dest, None)
    value = None
    if op == "const" or op == "mov":
        value = _value(sources[0], const)
    elif op in FOLD:
        left, right = _value(sources[0], const), _value(sources[1], const)
        if left is not None and right is not None:
            value = FOLD[op](left, right)
    if value is None or new_raw:
        const.pop(dest, None)
    else:
        const[dest] = value


def prove(blocks, params):
    """Return per-parameter masks, or raise Reject."""
    registers = set(params)
    for body in blocks.values():
        for op, args in body:
            if op in MEMORY_OPS:
                raise Reject("memory or volatile instruction")
            registers.update(arg for arg in args if is_reg(arg))
    entry = next(iter(blocks))
    start_raw = {param: frozenset([index]) for index, param in enumerate(params)}
    start_const = {reg: 0 for reg in registers if reg not in start_raw}
    state = {entry: (start_raw, start_const)}
    work = [entry]
    rounds = 0
    while work:
        rounds += 1
        if rounds > 20000:
            raise Reject("dataflow did not settle")
        label = work.pop()
        raw, const = dict(state[label][0]), dict(state[label][1])
        body = blocks[label]
        for op, args in body:
            _step(op, args, raw, const, None)
        op, args = body[-1]
        succ = args if op == "jmp" else args[1:] if op == "br" else ()
        for dest in dict.fromkeys(succ):
            if dest not in state:
                state[dest] = (raw, const)
                work.append(dest)
                continue
            old_raw, old_const = state[dest]
            merged_raw = dict(old_raw)
            changed = False
            for reg, which in raw.items():
                both = merged_raw.get(reg, frozenset()) | which
                if both != merged_raw.get(reg):
                    merged_raw[reg] = both
                    changed = True
            merged_const = {reg: v for reg, v in old_const.items() if const.get(reg) == v}
            if len(merged_const) != len(old_const):
                changed = True
            if changed:
                state[dest] = (merged_raw, merged_const)
                if dest not in work:
                    work.append(dest)
    masks = [0] * len(params)
    for label, (raw, const) in state.items():
        raw, const = dict(raw), dict(const)
        for op, args in blocks[label]:
            _step(op, args, raw, const, masks)
    return masks


def _bits(mask):
    return [bit for bit in range(32) if mask >> bit & 1]


def _poly(value):
    return value if isinstance(value, dict) else ({(): value} if value else {})


def _padd(left, right, sign=1):
    out = dict(_poly(left))
    for mono, coef in _poly(right).items():
        coef = (out.get(mono, 0) + sign * coef) & MASK
        if coef:
            out[mono] = coef
        else:
            out.pop(mono, None)
    return _settle(out)


def _pmul(left, right):
    out = {}
    for m1, c1 in _poly(left).items():
        for m2, c2 in _poly(right).items():
            mono = tuple(sorted(m1 + m2))
            if len(mono) > MAX_DEGREE:
                raise Reject("degree too high")
            coef = (out.get(mono, 0) + c1 * c2) & MASK
            if coef:
                out[mono] = coef
            else:
                out.pop(mono, None)
    return _settle(out)


def _settle(poly):
    if len(poly) > MAX_TERMS:
        raise Reject("too many terms")
    if not poly:
        return 0
    if len(poly) == 1 and () in poly:
        return poly[()]
    return poly


def _concrete(value):
    if isinstance(value, dict):
        raise Reject("parameter reaches a non-polynomial use")
    return value


def symbolic(compiled, args, masks, limit):
    """Run with parameters as polynomial unknowns; masked reads use args.

    Control flow, addresses and every non-polynomial operation must see
    concrete values, so the path and the result's form hold for all inputs
    that share the masked bits in args. Returns (value, steps).
    """
    from .kirlib import ADD, SUB, MUL, AND, OR, XOR, SHL, SHR, LT, EQ, MOV, SELECT
    from .kirlib import DIV, MOD, MAC, ADDSH, JMP, BR, RET, SIGN
    R = list(compiled.R0)
    for index in range(len(masks)):
        R[index] = {(index,): 1}
    code = compiled.code
    pc = steps = 0
    while True:
        if steps >= limit:
            raise Reject("a class runs too long")
        op, d, a, b, c, t1, t2 = code[pc]
        steps += 1
        pc += 1
        if op == MOV:
            R[d] = R[a]
        elif op == ADD:
            x, y = R[a], R[b]
            R[d] = (x + y) & MASK if type(x) is int and type(y) is int else _padd(x, y)
        elif op == SUB:
            x, y = R[a], R[b]
            R[d] = (x - y) & MASK if type(x) is int and type(y) is int else _padd(x, y, -1)
        elif op == MUL:
            x, y = R[a], R[b]
            R[d] = (x * y) & MASK if type(x) is int and type(y) is int else _pmul(x, y)
        elif op == MAC:
            x, y, z = R[a], R[b], R[c]
            if type(x) is int and type(y) is int and type(z) is int:
                R[d] = (x * y + z) & MASK
            else:
                R[d] = _padd(_pmul(x, y), z)
        elif op == SHL:
            x, y = R[a], _concrete(R[b])
            R[d] = (x << (y & 31)) & MASK if type(x) is int else _pmul(x, 1 << (y & 31))
        elif op == ADDSH:
            x, y = R[a], R[b]
            if type(x) is int and type(y) is int:
                R[d] = (x + (y << c)) & MASK
            else:
                R[d] = _padd(x, _pmul(y, (1 << c) & MASK))
        elif op == SELECT:
            R[d] = R[b] if _concrete(R[a]) != 0 else R[c]
        elif op == AND:
            x, y = R[a], R[b]
            if type(x) is int and type(y) is int:
                R[d] = x & y
            else:
                if type(x) is int:
                    x, y = y, x
                if type(y) is not int or len(x) != 1:
                    raise Reject("parameter reaches a non-polynomial use")
                (mono, coef), = x.items()
                if len(mono) != 1 or coef != 1 or y & ~masks[mono[0]]:
                    raise Reject("mask outside the proven bits")
                R[d] = args[mono[0]] & y
        elif op == BR:
            pc = t1 if _concrete(R[a]) != 0 else t2
        elif op == JMP:
            pc = t1
        elif op == RET:
            return R[a], steps
        else:
            x, y = _concrete(R[a]), _concrete(R[b])
            if op == OR:
                R[d] = x | y
            elif op == XOR:
                R[d] = x ^ y
            elif op == SHR:
                R[d] = x >> (y & 31)
            elif op == LT:
                R[d] = 1 if x ^ SIGN < y ^ SIGN else 0
            elif op == EQ:
                R[d] = 1 if x == y else 0
            elif op in (DIV, MOD):
                if y == 0:
                    raise Reject("a class divides by zero")
                sx = x - (1 << 32) if x & SIGN else x
                sy = y - (1 << 32) if y & SIGN else y
                q = abs(sx) // abs(sy) if op == DIV else abs(sx) % abs(sy)
                if (op == DIV and (sx < 0) != (sy < 0)) or (op == MOD and sx < 0):
                    q = -q
                R[d] = q & MASK
            else:
                raise Reject("unsupported instruction")


def table(program, masks):
    """Evaluate the original for every key; return (layout, results)."""
    layout = [(index, bit) for index, mask in enumerate(masks) for bit in _bits(mask)]
    if len(layout) > MAX_BITS:
        raise Reject("too many input bits")
    compiled = compile_program(program)
    results = []
    budget = TOTAL_STEPS
    for key in range(1 << len(layout)):
        args = [0] * len(masks)
        for position, (index, bit) in enumerate(layout):
            if key >> position & 1:
                args[index] |= 1 << bit
        value, steps = symbolic(compiled, args, masks, min(RUN_STEPS, budget))
        budget -= steps
        results.append(value)
    return layout, results


class Emitter:
    def __init__(self, params):
        self.params = params
        self.blocks = {}
        top = max((~p for p in params), default=-1)
        self.next_reg = top + 1
        self.next_label = 0
        self.memo = {}

    def reg(self):
        self.next_reg += 1
        return ~(self.next_reg - 1)

    def label(self):
        self.next_label += 1
        return "fd%d" % (self.next_label - 1)


def _runs(mask):
    runs, bit = [], 0
    while bit < 32:
        if mask >> bit & 1:
            start = bit
            while bit < 32 and mask >> bit & 1:
                bit += 1
            runs.append((start, bit - start))
        else:
            bit += 1
    return runs


def _key_code(emitter, masks):
    code, key, offset = [], None, 0
    for index, mask in enumerate(masks):
        for start, length in _runs(mask):
            piece = emitter.reg()
            source = emitter.params[index]
            if start:
                code.append(("shr", (piece, source, start)))
                source = piece
            if start + length < 32:
                code.append(("and", (piece, source, (1 << length) - 1)))
            elif source != piece:
                code.append(("mov", (piece, source)))
            if offset:
                code.append(("shl", (piece, piece, offset)))
            if key is None:
                key = piece
            else:
                code.append(("or", (key, key, piece)))
            offset += length
    return code, key


def _fields(values):
    """Choose base and power-of-two width so every value fits."""
    best = None
    for base in (min(values), min(values, key=lambda v: v ^ 0x80000000)):
        span = max((v - base) & MASK for v in values)
        width = 1
        while width < 32 and span >> width:
            width *= 2
        if best is None or width < best[1]:
            best = (base, width)
    return best


def _leaf(emitter, key, values, code):
    """Append code computing values[key mod len(values)] into a register."""
    n = len(values)
    if len(set(values)) == 1:
        return values[0]
    base, width = _fields(values)
    per_word = 32 // width
    if n <= per_word:
        width = 32 // n
        per_word = n
    words = []
    for start in range(0, n, per_word):
        word = 0
        for slot, value in enumerate(values[start:start + per_word]):
            word |= ((value - base) & MASK & ((1 << width) - 1 if width < 32 else MASK)) << (slot * width)
        words.append(word & MASK)
    shift = key
    if width > 1:
        shift = emitter.reg()
        code.append(("shl", (shift, key, width.bit_length() - 1)))
    level = list(words)
    bit = per_word
    while len(level) > 1:
        cond = emitter.reg()
        code.append(("and", (cond, key, bit)))
        nxt = []
        for low, high in zip(level[0::2], level[1::2]):
            if low == high:
                nxt.append(low)
                continue
            out = emitter.reg()
            code.append(("select", (out, cond, high, low)))
            nxt.append(out)
        level = nxt
        bit *= 2
    out = emitter.reg()
    code.append(("shr", (out, level[0], shift)))
    if width < 32:
        code.append(("and", (out, out, (1 << width) - 1)))
    if base:
        code.append(("add", (out, out, base)))
    return out


def _tree(emitter, key, values, leaf_words, lo_bits):
    """Return the label of a block that returns values[key] for this subrange."""
    signature = tuple(values)
    if signature in emitter.memo:
        return emitter.memo[signature]
    label = emitter.label()
    emitter.memo[signature] = label
    emitter.blocks[label] = None
    n = len(values)
    base, width = _fields(values)
    words = max(1, (n * width + 31) // 32)
    code = []
    if len(set(values)) == 1 or words <= leaf_words:
        emitter.next_reg = emitter.leaf_base
        out = _leaf(emitter, key, values, code)
        code.append(("ret", (out,)))
        emitter.blocks[label] = code
        return label
    half = n // 2
    low = _tree(emitter, key, values[:half], leaf_words, lo_bits)
    high = _tree(emitter, key, values[half:], leaf_words, lo_bits)
    if low == high:
        emitter.blocks[label] = [("jmp", (low,))]
        return label
    cond = emitter.cond
    emitter.blocks[label] = [("and", (cond, key, half)), ("br", (cond, high, low))]
    return label


def _order(blocks, entry):
    """Depth-first layout that keeps each false target right after its branch."""
    order, seen, stack = [], set(), [entry]
    while stack:
        label = stack.pop()
        if label in seen:
            continue
        seen.add(label)
        order.append(label)
        op, args = blocks[label][-1]
        if op == "br":
            stack.append(args[1])
            stack.append(args[2])
        elif op == "jmp":
            stack.append(args[0])
    return {label: blocks[label] for label in order}


def _affine(values):
    if len(values) < 2:
        return None
    step = (values[1] - values[0]) & MASK
    if all((values[k] - values[0] - k * step) & MASK == 0 for k in range(len(values))):
        return values[0], step
    return None


def build(params, masks, values, leaf_words):
    emitter = Emitter(params)
    code, key = _key_code(emitter, masks)
    entry = "entry"
    if key is None or len(set(values)) == 1:
        emitter.blocks[entry] = [("ret", (values[0] & MASK,))]
        return emitter.blocks
    affine = _affine(values)
    if affine is not None and leaf_words == 0:
        start, step = affine
        out = emitter.reg()
        code.append(("mul", (out, key, step)))
        code.append(("add", (out, out, start)))
        emitter.blocks[entry] = code + [("ret", (out,))]
        return emitter.blocks
    if leaf_words == 0:
        leaf_words = 1
    emitter.cond = emitter.reg()
    emitter.leaf_base = emitter.next_reg
    root = _tree(emitter, key, values, leaf_words, 0)
    body = emitter.blocks.pop(root)
    if body[-1][0] == "ret":
        emitter.blocks = {entry: code + body}
        return emitter.blocks
    blocks = {entry: code + body}
    blocks.update((label, b) for label, b in emitter.blocks.items())
    return _order(blocks, entry)


def _key_bits(params, masks):
    """(parameter, bit) feeding each key bit, lowest key bit first."""
    bits = []
    for index, mask in enumerate(masks):
        for start, length in _runs(mask):
            bits.extend((params[index], start + j) for j in range(length))
    return bits


def _inner_first(blocks, entry):
    """Depth-first order with every branch block ahead of the return blocks."""
    order = list(_order(blocks, entry))
    return {label: blocks[label]
            for label in [l for l in order if blocks[l][-1][0] != "ret"]
            + [l for l in order if blocks[l][-1][0] == "ret"]}


def build_tests(params, masks, values, select_levels, early, inner):
    """Bit-test tree: one `br` per key bit, `select` over the lowest bits.

    The key bits are tested straight on the parameters. With `early` every
    test is computed once in the entry block, so an internal node is a single
    `br`; otherwise each node tests its own bit into one shared register.
    The lowest `select_levels` bits pick between literals instead of
    branching. `inner` places the branch blocks ahead of the returns, which
    keeps the code executed on every path inside a small instruction cache.
    """
    bits = _key_bits(params, masks)
    values = tuple(values)
    if len(values) != 1 << len(bits) or len(values) > 256 or len(set(values)) == 1:
        return None
    emitter = Emitter(params)
    tests = [emitter.reg() for _ in bits]
    shared = emitter.reg()
    entry_code = [("and", (tests[j], source, 1 << bit)) for j, (source, bit) in enumerate(bits)] if early else []
    blocks = {}

    def test(j, code):
        if early:
            return tests[j]
        source, bit = bits[j]
        register = tests[j] if j < select_levels else shared
        code.append(("and", (register, source, 1 << bit)))
        return register

    def node(low, size):
        part = values[low:low + size]
        if part in memo:
            return memo[part]
        label = emitter.label()
        memo[part] = label
        if len(set(part)) == 1:
            blocks[label] = [("ret", (part[0] & MASK,))]
            return label
        if size <= 1 << select_levels:
            code = []
            level = [v & MASK for v in part]
            j = 0
            while len(level) > 1:
                cond = test(j, code)
                following = []
                for zero, one in zip(level[0::2], level[1::2]):
                    if zero == one:
                        following.append(zero)
                        continue
                    out = emitter.reg()
                    code.append(("select", (out, cond, one, zero)))
                    following.append(out)
                level = following
                j += 1
            blocks[label] = code + [("ret", (level[0],))]
            return label
        blocks[label] = None
        half = size // 2
        zero = node(low, half)
        one = node(low + half, half)
        code = []
        cond = test(size.bit_length() - 2, code)
        blocks[label] = code + [("br", (cond, one, zero))]
        return label

    memo = {}
    root = node(0, len(values))
    body = blocks.pop(root)
    ordered = {"entry": entry_code + body}
    ordered.update(blocks)
    return (_inner_first if inner else _order)(ordered, "entry")


INVERSE3 = 0xAAAAAAAB


def _newton(values):
    """Forward-difference coefficients a_j with values[n] = sum a_j * C(n, j)."""
    row, coefs = list(values), []
    while row:
        coefs.append(row[0])
        row = [(row[i + 1] - row[i]) & MASK for i in range(len(row) - 1)]
    while coefs and not coefs[-1]:
        coefs.pop()
    return coefs


def _binomials_fit(degree, size):
    """True when C(n, j) for j <= degree can be built without wrapping for n < size."""
    if degree > 4:
        return False
    for n in range(size):
        choose = 1
        for j in range(1, degree + 1):
            if choose * (n - j + 1) > MASK:
                return False
            choose = choose * (n - j + 1) // j
    return True


class Terms:
    """Emit sums of monomials over parameters with shared products."""

    def __init__(self, emitter, code):
        self.emitter, self.code, self.products = emitter, code, {}

    def product(self, mono):
        if mono in self.products:
            return self.products[mono]
        if len(mono) == 1:
            reg = self.emitter.params[mono[0]]
        else:
            reg = self.emitter.reg()
            self.code.append(("mul", (reg, self.product(mono[:-1]), self.emitter.params[mono[-1]])))
        self.products[mono] = reg
        return reg

    def scaled(self, reg, coef):
        if coef == 1:
            return reg
        out = self.emitter.reg()
        if coef & (coef - 1) == 0:
            self.code.append(("shl", (out, reg, coef.bit_length() - 1)))
        else:
            self.code.append(("mul", (out, reg, coef)))
        return out

    def total(self, parts, constant=0):
        """parts: list of registers; returns a register or literal."""
        acc = None
        for reg in parts:
            if acc is None:
                acc = reg
            else:
                out = self.emitter.reg()
                self.code.append(("add", (out, acc, reg)))
                acc = out
        if acc is None:
            return constant & MASK
        if constant:
            out = self.emitter.reg()
            self.code.append(("add", (out, acc, constant & MASK)))
            acc = out
        return acc

    def poly(self, coefs):
        """coefs: {monomial: coefficient}; returns a register or literal.

        Variables shared by several monomials are factored out (Horner form).
        """
        constant = coefs.get((), 0)
        rest = {m: c for m, c in coefs.items() if m and c}
        counts = {}
        for m in rest:
            for var in set(m):
                counts[var] = counts.get(var, 0) + 1
        if len(rest) > 1 and max(counts.values()) > 1:
            var = min(counts, key=lambda v: (-counts[v], v))
            inner, outer = {}, {}
            for m, c in rest.items():
                if var in m:
                    trimmed = list(m)
                    trimmed.remove(var)
                    inner[tuple(trimmed)] = c
                else:
                    outer[m] = c
            factor = self.poly(inner)
            if is_reg(factor):
                out = self.emitter.reg()
                self.code.append(("mul", (out, factor, self.emitter.params[var])))
            else:
                out = self.scaled(self.emitter.params[var], factor)
            parts = [out]
            if outer:
                tail = self.poly(outer)
                parts.append(tail)
            return self.total(parts, constant)
        parts = [self.scaled(self.product(m), c) for m, c in sorted(rest.items())]
        return self.total(parts, constant)


def build_poly(params, masks, values, shape):
    emitter = Emitter(params)
    code, key = _key_code(emitter, masks)
    polys = [_poly(v) for v in values]
    monos = sorted({m for p in polys for m in p} | {()})
    terms = Terms(emitter, code)
    if key is None:
        out = terms.poly(polys[0])
    elif shape == "newton":
        fits = {m: _newton([p.get(m, 0) for p in polys]) for m in monos}
        degree = max(len(c) for c in fits.values()) - 1
        if not _binomials_fit(degree, len(values)):
            return None
        basis = [None, key]
        for j in range(2, degree + 1):
            factor = emitter.reg()
            code.append(("sub", (factor, key, j - 1)))
            out = emitter.reg()
            code.append(("mul", (out, basis[-1], factor)))
            if j == 3:
                code.append(("mul", (out, out, INVERSE3)))
            elif j in (2, 4):
                code.append(("shr", (out, out, 1 if j == 2 else 2)))
            basis.append(out)
        parts, constant = [], 0
        for j in range(degree + 1):
            group = {m: c[j] for m, c in fits.items() if j < len(c) and c[j]}
            if not group:
                continue
            if j == 0:
                if list(group) == [()]:
                    constant = group[()]
                else:
                    parts.append(terms.poly(group))
                continue
            inner = terms.poly(group)
            if is_reg(inner):
                out = emitter.reg()
                code.append(("mul", (out, basis[j], inner)))
                parts.append(out)
            else:
                parts.append(terms.scaled(basis[j], inner))
        out = terms.total(parts, constant)
    else:
        parts, constant = [], 0
        for m in monos:
            column = [p.get(m, 0) for p in polys]
            if len(set(column)) == 1:
                if not m:
                    constant = column[0]
                elif column[0]:
                    parts.append(terms.scaled(terms.product(m), column[0]))
                continue
            if max(1, len(column) * _fields(column)[1] // 32) > 16:
                return None
            coef = _leaf(emitter, key, column, code)
            if not m:
                parts.append(coef)
                continue
            out = emitter.reg()
            code.append(("mul", (out, coef, terms.product(m))))
            parts.append(out)
        out = terms.total(parts, constant)
    return {"entry": code + [("ret", (out,))]}


EDGE = (0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 255, 256, 1000, 65535,
        0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF)


def samples(nparams, count=48):
    """Deterministic argument lists: edge values, then a mix of small and wide values."""
    out = [[value] * nparams for value in EDGE]
    state = 0x2545F491
    while len(out) < count:
        args = []
        for _ in range(nparams):
            state = (state * 1103515245 + 12345) & MASK
            pick = state >> 16 & 15
            state = (state * 1103515245 + 12345) & MASK
            if pick < 9:
                args.append(state >> 20 & 15)
            elif pick < 12:
                args.append(EDGE[(state >> 16) % len(EDGE)])
            else:
                args.append(state ^ (state << 7 & MASK))
        out.append(args)
    return out


def spread(blocks, params, count=64):
    """Argument lists whose masked key bits cover the key range evenly."""
    try:
        masks = prove(blocks, params)
    except Reject:
        return []
    bits = _key_bits(params, masks)
    if not bits or len(bits) > MAX_BITS:
        return []
    size = 1 << len(bits)
    count = min(count, size)
    stride = size // count
    out = []
    for index in range(count):
        key = index * stride + (index * 7) % stride
        args = [0] * len(params)
        for j, (source, bit) in enumerate(bits):
            if key >> j & 1:
                args[params.index(source)] |= 1 << bit
        out.append(args)
    return out


def cost(program, machine, argsets):
    """Total cycles of program over argsets, or None if a run does not return."""
    compiled = compile_program(program)
    total = 0
    for args in argsets:
        result = execute(compiled, args, [], {}, RUN_STEPS, want_trace=True)
        if result.status != "RET":
            return None
        total += time_trace(compiled, result.trace, machine)
    return total


def specialize(program, blocks, params):
    """Yield lookup versions of the program, or nothing when not provable."""
    if len(program.instrs) > MAX_INSTRUCTIONS or not params:
        return
    try:
        masks = prove(blocks, params)
        if sum(bin(m).count("1") for m in masks) > MAX_BITS:
            return
        _, values = table(program, masks)
    except Reject:
        return
    seen = set()
    if any(isinstance(v, dict) for v in values):
        shapes = [build_poly(params, masks, values, shape) for shape in ("newton", "lookup")]
    else:
        shapes = [build(params, masks, values, w) for w in (0, 1, 2, 4, 8, 16, 32)]
        shapes.append(build_poly(params, masks, values, "newton"))
        if _affine(values) is None:
            for levels, early, inner in ((1, True, False), (0, True, True), (1, True, True),
                                         (2, True, False), (3, True, False), (1, False, True),
                                         (2, False, False)):
                shapes.append(build_tests(params, masks, values, levels, early, inner))
    for result in shapes:
        if result is None:
            continue
        signature = repr(list(result.items()))
        if signature not in seen:
            seen.add(signature)
            yield result
