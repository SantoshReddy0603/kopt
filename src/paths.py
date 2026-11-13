"""Straight-line versions of the paths of a program without memory.

The original program is executed with its parameters as unknowns. Values are
polynomials (modulo 2^32) over atoms; an atom is a parameter, a volatile load,
or an operation that is not polynomial (and, compare, divide, ...) applied to
other values. Atoms are shared by structure, so a loop that keeps adding the
same quotient builds `8 * q + 28` instead of a chain.

A branch whose condition is not a known constant splits the execution: one
side assumes the condition is non-zero, the other that it is zero, and later
uses of the same condition follow the assumption. A divisor that is not a
known constant splits the same way on `divisor == 0`; the zero side ends in
the division trap. Volatile loads and stores are kept in their original order
on each path. Memory instructions need a known address: a load of a word the
path has stored reads the stored value, any other load becomes an atom that is
emitted at its original place, and stores are emitted in order. An address at
or above the memory size ends the path in the address trap. A store is dropped
when every continuation of its path stores the same word again before ending,
and stores to the scratch words (never observed) are always dropped, since
later loads of them read the tracked value. Each path ends in `ret` or `trap`.

The result is a decision tree of the conditions, and every leaf computes its
effects and return value straight-line. The number of paths, steps and terms
are bounded.
"""
from .ir import is_reg
from .kirlib import (ADD, ADDSH, AND, BR, DIV, EQ, JMP, LOAD, LT, MAC, MASK, MOD, MOV, MUL,
                     OR, RET, SELECT, SHL, SHR, SIGN, STORE, SUB, TRAP, VLOAD, VSTORE, XOR,
                     MEM_WORDS, SCRATCH_BASE, compile_program)

MAX_PATHS = 40
STEPS = 150000
MAX_OUTPUT = 3000
MAX_TERMS = 16
MAX_DEGREE = 4
NAMES = {AND: "and", OR: "or", XOR: "xor", SHL: "shl", SHR: "shr", LT: "lt", EQ: "eq",
         DIV: "div", MOD: "mod"}


class Reject(Exception):
    pass


def _freeze(value):
    return value if isinstance(value, int) else tuple(sorted(value.items()))


def _poly(value):
    return value if isinstance(value, dict) else ({(): value} if value else {})


def _settle(poly):
    if len(poly) > MAX_TERMS:
        raise Reject("too many terms")
    if not poly:
        return 0
    if len(poly) == 1 and () in poly:
        return poly[()]
    return poly


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


def _fold(op, x, y):
    if op == AND:
        return x & y
    if op == OR:
        return x | y
    if op == XOR:
        return x ^ y
    if op == SHL:
        return (x << (y & 31)) & MASK
    if op == SHR:
        return x >> (y & 31)
    if op == LT:
        return 1 if x ^ SIGN < y ^ SIGN else 0
    if op == EQ:
        return 1 if x == y else 0
    sx = x - (1 << 32) if x & SIGN else x
    sy = y - (1 << 32) if y & SIGN else y
    q = abs(sx) // abs(sy) if op == DIV else abs(sx) % abs(sy)
    if (op == DIV and (sx < 0) != (sy < 0)) or (op == MOD and sx < 0):
        q = -q
    return q & MASK


class Atoms:
    def __init__(self, nparams):
        self.defs = [("param", p) for p in range(nparams)]
        self.index = {d: i for i, d in enumerate(self.defs)}

    def atom(self, definition):
        found = self.index.get(definition)
        if found is None:
            found = len(self.defs)
            self.defs.append(definition)
            self.index[definition] = found
        return {(found,): 1}


def _atom_of(value):
    """The atom id when value is exactly one atom, else None."""
    if isinstance(value, dict) and len(value) == 1:
        (mono, coef), = value.items()
        if coef == 1 and len(mono) == 1:
            return mono[0]
    return None


class Path:
    __slots__ = ("pc", "regs", "known", "effects", "steps", "mem")

    def __init__(self, pc, regs, known, effects, steps, mem):
        self.pc, self.regs, self.known, self.effects, self.steps = pc, regs, known, effects, steps
        self.mem = mem

    def split(self):
        return Path(self.pc, list(self.regs), dict(self.known), [], self.steps, dict(self.mem))


class Runner:
    def __init__(self, compiled, nparams):
        self.compiled = compiled
        self.atoms = Atoms(nparams)
        self.paths = 0
        self.steps = 0
        self.vloads = 0

    def operate(self, op, x, y, path):
        if isinstance(x, int) and isinstance(y, int):
            return _fold(op, x, y)
        if op == AND and (x == 0 or y == 0):
            return 0
        if op in (OR, XOR) and (x == 0 or y == 0):
            return y if x == 0 else x
        if op in (SHL, SHR) and y == 0:
            return x
        if op in (SHL, SHR) and isinstance(y, int):
            if op == SHL:
                return _pmul(x, (1 << (y & 31)) & MASK)
        if op in (EQ, LT):
            xr, yr = self.bound(x), self.bound(y)
            if xr and yr and xr[1] <= SIGN - 1 and yr[1] <= SIGN - 1:
                if op == LT and xr[1] < yr[0]:
                    return 1
                if op == LT and xr[0] >= yr[1]:
                    return 0
                if op == EQ and (xr[1] < yr[0] or yr[1] < xr[0]):
                    return 0
        if op in (EQ, LT) and _freeze(x) == _freeze(y):
            return 1 if op == EQ else 0
        if op in (EQ, XOR, OR, AND) and isinstance(x, int):
            x, y = y, x
        if op == EQ and y == 0:
            inner = _atom_of(x)
            if inner is not None and inner in path.known:
                return 0 if path.known[inner] else 1
        if op == EQ and isinstance(y, int):
            inner = _atom_of(x)
            if inner is not None and self.atoms.defs[inner][0] in ("lt", "eq") and y > 1:
                return 0
        value = self.atoms.atom((NAMES[op], _freeze(x), _freeze(y)))
        known = path.known.get(_atom_of(value))
        if known is not None and known is not True:
            return known
        return value

    def bound(self, value):
        """Unsigned (lo, hi) known for a constant or a single masked atom."""
        if isinstance(value, int):
            return value, value
        atom = _atom_of(value)
        if atom is None:
            return None
        definition = self.atoms.defs[atom]
        kind = definition[0]
        if kind in ("lt", "eq"):
            return 0, 1
        if kind == "and":
            masks = [v for v in definition[1:] if isinstance(v, int)]
            return (0, masks[0]) if masks else None
        if kind == "shr" and isinstance(definition[2], int):
            return 0, MASK >> (definition[2] & 31)
        return None

    def truth(self, value, path):
        """1, 0 or None for a condition value under the path's assumptions."""
        if isinstance(value, int):
            return 1 if value else 0
        atom = _atom_of(value)
        if atom is None:
            atom = _atom_of(self.atoms.atom(("nz", _freeze(value), 0)))
        if atom in path.known:
            return 1 if path.known[atom] else 0
        return None

    def assume(self, value, path, holds):
        atom = _atom_of(value)
        if atom is None:
            atom = _atom_of(self.atoms.atom(("nz", _freeze(value), 0)))
        kind = self.atoms.defs[atom][0]
        if holds:
            path.known[atom] = 1 if kind in ("lt", "eq") else True
        else:
            path.known[atom] = 0
        definition = self.atoms.defs[atom]
        if kind == "eq" and definition[2] == 0 and isinstance(definition[1], tuple):
            inner = _atom_of(dict(definition[1]))
            if inner is not None:
                if holds:
                    path.known[inner] = 0
                elif self.atoms.defs[inner][0] in ("lt", "eq"):
                    path.known[inner] = 1
                else:
                    path.known.setdefault(inner, True)

    def run(self, path):
        """Return a tree: ('leaf', effects, kind, value) or ('fork', effects, cond, yes, no)."""
        self.paths += 1
        if self.paths > MAX_PATHS:
            raise Reject("too many paths")
        code = self.compiled.code
        R = path.regs
        while True:
            path.steps += 1
            self.steps += 1
            if self.steps > STEPS:
                raise Reject("runs too long")
            op, d, a, b, c, t1, t2 = code[path.pc]
            path.pc += 1
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
                R[d] = _padd(_pmul(R[a], R[b]), R[c])
            elif op == ADDSH:
                R[d] = _padd(R[a], _pmul(R[b], (1 << c) & MASK))
            elif op in (DIV, MOD):
                divisor = R[b]
                if not isinstance(divisor, int):
                    zero = self.operate(EQ, divisor, 0, path)
                    state = self.truth(zero, path)
                    if state is None:
                        return self.fork(path, zero, ("trap_div",), op, d, a, b)
                    if state == 1:
                        return ("leaf", path.effects, "trap", "div_by_zero")
                    divisor = R[b]
                elif divisor == 0:
                    return ("leaf", path.effects, "trap", "div_by_zero")
                R[d] = self.operate(op, R[a], divisor, path)
            elif op in NAMES:
                R[d] = self.operate(op, R[a], R[b], path)
            elif op == SELECT:
                state = self.truth(R[a], path)
                if state is None:
                    if _freeze(R[b]) == _freeze(R[c]):
                        R[d] = R[b]
                    else:
                        R[d] = self.atoms.atom(("select", _freeze(R[a]), _freeze(R[b]), _freeze(R[c])))
                else:
                    R[d] = R[b] if state else R[c]
            elif op == BR:
                state = self.truth(R[a], path)
                if state is None:
                    return self.fork(path, R[a], ("branch", t1, t2))
                path.pc = t1 if state else t2
            elif op == JMP:
                path.pc = t1
            elif op == VLOAD:
                self.vloads += 1
                value = self.atoms.atom(("vload", _freeze(R[a]), self.vloads))
                path.effects.append(("vload", _atom_of(value), R[a]))
                R[d] = value
            elif op == VSTORE:
                path.effects.append(("vstore", R[a], R[b]))
            elif op == RET:
                return ("leaf", path.effects, "ret", R[a])
            elif op == TRAP:
                return ("leaf", path.effects, "trap", R[a])
            elif op in (LOAD, STORE):
                address = R[a]
                if not isinstance(address, int):
                    raise Reject("unknown address")
                if address >= MEM_WORDS:
                    return ("leaf", path.effects, "trap", "bad_address")
                if op == STORE:
                    path.mem[address] = R[b]
                    if address < SCRATCH_BASE:
                        path.effects.append(("store", address, R[b]))
                elif address in path.mem:
                    R[d] = path.mem[address]
                else:
                    self.vloads += 1
                    value = self.atoms.atom(("load", address, self.vloads))
                    path.effects.append(("load", _atom_of(value), address))
                    path.mem[address] = R[d] = value
            else:
                raise Reject("unknown instruction")

    def fork(self, path, cond, how, *redo):
        effects = path.effects
        yes, no = path.split(), path.split()
        self.assume(cond, yes, True)
        self.assume(cond, no, False)
        if how[0] == "branch":
            yes.pc, no.pc = how[1], how[2]
            return ("fork", effects, cond, self.run(yes), self.run(no))
        # divisor: the condition is divisor == 0; the zero side traps
        yes_tree = ("leaf", [], "trap", "div_by_zero")
        no.pc -= 1
        no.steps -= 1
        self.steps -= 1
        return ("fork", effects, cond, yes_tree, self.run(no))


class Emit:
    def __init__(self, params, atoms, inline_larger=False):
        self.params = params
        self.atoms = atoms
        self.inline_larger = inline_larger
        self.sizes = {}
        self.blocks = {}
        self.next_reg = max((~p for p in params), default=-1) + 1
        self.next_label = 0

    def reg(self):
        self.next_reg += 1
        return ~(self.next_reg - 1)

    def label(self):
        self.next_label += 1
        return "pt%d" % (self.next_label - 1)

    def value(self, value, code, memo):
        if isinstance(value, tuple):
            value = dict(value) if value else 0
        if isinstance(value, int):
            return value & MASK
        key = _freeze(value)
        if key in memo:
            return memo[key]
        constant = value.get((), 0)
        rest = {m: c for m, c in value.items() if m}
        out = self.poly(rest, code, memo)
        if constant:
            if is_reg(out):
                total = self.reg()
                code.append(("add", (total, out, constant)))
                out = total
            else:
                out = (out + constant) & MASK
        memo[key] = out
        return out

    def poly(self, coefs, code, memo):
        if not coefs:
            return 0
        counts = {}
        for m in coefs:
            for atom in set(m):
                counts[atom] = counts.get(atom, 0) + 1
        if len(coefs) > 1 and max(counts.values()) > 1:
            atom = min(counts, key=lambda a: (-counts[a], a))
            inner, outer = {}, {}
            for m, c in coefs.items():
                if atom in m:
                    trimmed = list(m)
                    trimmed.remove(atom)
                    inner[tuple(trimmed)] = c
                else:
                    outer[m] = c
            factor = self.value(inner, code, memo)
            base = self.atom(atom, code, memo)
            part = self.times(base, factor, code)
            if outer:
                tail = self.poly(outer, code, memo)
                return self.plus(part, tail, code)
            return part
        acc = None
        for m, c in sorted(coefs.items()):
            reg = self.atom(m[0], code, memo)
            for atom in m[1:]:
                reg = self.times(reg, self.atom(atom, code, memo), code)
            reg = self.times(reg, c, code)
            acc = reg if acc is None else self.plus(acc, reg, code)
        return acc

    def times(self, left, right, code):
        if not is_reg(right):
            if right == 1:
                return left
            if not is_reg(left):
                return (left * right) & MASK
            out = self.reg()
            if right & (right - 1) == 0:
                code.append(("shl", (out, left, right.bit_length() - 1)))
            else:
                code.append(("mul", (out, left, right)))
            return out
        if not is_reg(left):
            return self.times(right, left, code)
        out = self.reg()
        code.append(("mul", (out, left, right)))
        return out

    def plus(self, left, right, code):
        if not is_reg(left) and not is_reg(right):
            return (left + right) & MASK
        out = self.reg()
        code.append(("add", (out, left, right)))
        return out

    def atom(self, index, code, memo):
        key = ("atom", index)
        if key in memo:
            return memo[key]
        definition = self.atoms.defs[index]
        kind = definition[0]
        if kind == "param":
            out = self.params[definition[1]]
        elif kind in ("vload", "load"):
            raise Reject("load used before it is emitted")
        elif kind == "nz":
            out = self.value(definition[1], code, memo)
        elif kind == "select":
            cond = self.value(definition[1], code, memo)
            yes = self.value(definition[2], code, memo)
            no = self.value(definition[3], code, memo)
            out = self.reg()
            code.append(("select", (out, cond, yes, no)))
        else:
            left = self.value(definition[1], code, memo)
            right = self.value(definition[2], code, memo)
            out = self.reg()
            code.append((kind, (out, left, right)))
        memo[key] = out
        return out

    def effects(self, effects, code, memo):
        for effect in effects:
            if len(self.blocks) + sum(len(body) for body in self.blocks.values()) > MAX_OUTPUT:
                raise Reject("output too large")
            if effect[0] == "load":
                out = self.reg()
                code.append(("load", (out, effect[2])))
                memo[("atom", effect[1])] = out
            elif effect[0] == "store":
                code.append(("store", (effect[1], self.value(effect[2], code, memo))))
            elif effect[0] == "vload":
                port = self.value(effect[2], code, memo)
                out = self.reg()
                code.append(("vload", (out, port)))
                memo[("atom", effect[1])] = out
            else:
                port = self.value(effect[1], code, memo)
                data = self.value(effect[2], code, memo)
                code.append(("vstore", (port, data)))

    def tree(self, node, memo, label):
        code = []
        self.blocks[label] = code
        self.effects(node[1], code, memo)
        if node[0] == "leaf":
            _, _, kind, value = node
            if kind == "trap" and value == "div_by_zero":
                code.append(("div", (self.reg(), 1, 0)))
                code.append(("ret", (0,)))
                return
            if kind == "trap" and value == "bad_address":
                code.append(("load", (self.reg(), MEM_WORDS)))
                code.append(("ret", (0,)))
                return
            out = self.value(value, code, memo)
            if not is_reg(out) and kind == "ret":
                code.append(("ret", (out,)))
            else:
                code.append((kind, (out,)))
            return
        _, _, cond, yes, no = node
        while True:
            atom = _atom_of(cond) if isinstance(cond, dict) else None
            definition = self.atoms.defs[atom] if atom is not None else None
            if not definition or definition[0] != "eq" or definition[2] != 0 or isinstance(definition[1], int):
                break
            cond, yes, no = dict(definition[1]), no, yes
        test = self.value(cond, code, memo)
        if not is_reg(test):
            raise Reject("constant condition")
        yes_label, no_label = self.label(), self.label()
        code.append(("br", (test, yes_label, no_label)))
        if self.inline_larger and self.size(yes) > self.size(no):
            # The larger side usually holds more of the paths, so it follows
            # in the layout and its branch is not taken.
            self.tree(yes, dict(memo), yes_label)
            self.tree(no, dict(memo), no_label)
        else:
            self.tree(no, dict(memo), no_label)
            self.tree(yes, dict(memo), yes_label)

    def size(self, node):
        """Number of leaves under a node."""
        key = id(node)
        if key not in self.sizes:
            self.sizes[key] = 1 if node[0] == "leaf" else self.size(node[3]) + self.size(node[4])
        return self.sizes[key]


def _drop_stores(node):
    """Remove stores overwritten on every continuation; return the words always stored."""
    if node[0] == "leaf":
        later = set()
    else:
        later = _drop_stores(node[3]) & _drop_stores(node[4])
    effects = node[1]
    kept = []
    for effect in reversed(effects):
        if effect[0] == "store":
            if effect[1] in later:
                continue
            later.add(effect[1])
        kept.append(effect)
    effects[:] = reversed(kept)
    return later


def specialize(program, params, inline_larger=False):
    """Return blocks equal to the program built from its symbolic paths, or None.

    With inline_larger, the larger side of every split follows its branch in
    the layout instead of the false side.
    """
    if len(program.instrs) > 2000:
        return None
    try:
        runner = Runner(compile_program(program), len(params))
        regs = list(runner.compiled.R0)
        for p in range(len(params)):
            regs[p] = {(p,): 1}
        tree = runner.run(Path(0, regs, {}, [], 0, {}))
        _drop_stores(tree)
        if tree[0] == "leaf" and tree[2] == "ret" and not tree[1] and isinstance(tree[3], int):
            return {"entry": [("ret", (tree[3],))]}
        emit = Emit(params, runner.atoms, inline_larger)
        emit.tree(tree, {}, "entry")
    except Reject:
        return None
    return emit.blocks
