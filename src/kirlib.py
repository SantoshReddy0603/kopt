import re
from collections import OrderedDict, deque
MASK = 4294967295
SIGN = 2147483648
MEM_WORDS = 4096
SCRATCH_BASE = 3072
MAX_INSTRS = 500000
MAX_PARAMS = 8
DEFAULT_MAX_STEPS = 5000000

class KirError(Exception):

    def __init__(self, msg, line=None):
        Exception.__init__(self, msg)
        self.msg = msg
        self.line = line

    def __str__(self):
        return 'line %d: %s' % (self.line, self.msg) if self.line else self.msg
SIGS = {'const': 'di', 'mov': 'ds', 'add': 'dss', 'sub': 'dss', 'mul': 'dss', 'and': 'dss', 'or': 'dss', 'xor': 'dss', 'shl': 'dss', 'shr': 'dss', 'lt': 'dss', 'eq': 'dss', 'div': 'dss', 'mod': 'dss', 'select': 'dsss', 'mac': 'dsss', 'addsh': 'dssk', 'load': 'ds', 'store': 'ss', 'vload': 'ds', 'vstore': 'ss', 'trap': 's', 'jmp': 'L', 'br': 'sLL', 'ret': 's'}
TERMINATORS = ('jmp', 'br', 'ret', 'trap')
MEM_OPS = ('load', 'store', 'vload', 'vstore')
LATKEY = {'const': 'lat_alu', 'mov': 'lat_alu', 'add': 'lat_alu', 'sub': 'lat_alu', 'and': 'lat_alu', 'or': 'lat_alu', 'xor': 'lat_alu', 'shl': 'lat_alu', 'shr': 'lat_alu', 'lt': 'lat_alu', 'eq': 'lat_alu', 'select': 'lat_alu', 'mul': 'lat_mul', 'div': 'lat_div', 'mod': 'lat_div', 'load': 'lat_load', 'vload': 'lat_load', 'mac': 'lat_mac', 'addsh': 'lat_addsh'}
ADD, SUB, MUL, AND, OR, XOR, SHL, SHR, LT, EQ, MOV, SELECT, LOAD, STORE, DIV, MOD, MAC, ADDSH, VLOAD, VSTORE, TRAP, JMP, BR, RET = range(24)
OPID = {'add': ADD, 'sub': SUB, 'mul': MUL, 'and': AND, 'or': OR, 'xor': XOR, 'shl': SHL, 'shr': SHR, 'lt': LT, 'eq': EQ, 'const': MOV, 'mov': MOV, 'select': SELECT, 'load': LOAD, 'store': STORE, 'div': DIV, 'mod': MOD, 'mac': MAC, 'addsh': ADDSH, 'vload': VLOAD, 'vstore': VSTORE, 'trap': TRAP, 'jmp': JMP, 'br': BR, 'ret': RET}
REG_RE = re.compile('^v(0|[1-9][0-9]{0,5})$')
NAME_RE = re.compile('^[A-Za-z_][A-Za-z0-9_]*$')
LIT_RE = re.compile('^-?(0x[0-9a-fA-F]+|[0-9]+)$')
FUNC_RE = re.compile('^func\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\((.*)\\)$')
LABEL_RE = re.compile('^([A-Za-z_][A-Za-z0-9_]*):$')

def parse_literal_raw(tok, line=None):
    if not LIT_RE.match(tok):
        raise KirError("bad integer literal '%s'" % tok, line)
    neg = tok.startswith('-')
    body = tok[1:] if neg else tok
    n = int(body, 16) if body.startswith('0x') else int(body, 10)
    if neg:
        n = -n
    if n < -(1 << 31) or n > MASK:
        raise KirError("literal '%s' does not fit in 32 bits (allowed: -2147483648 .. 4294967295)" % tok, line)
    return n

class Instr(object):
    __slots__ = ('op', 'args', 'line', 'text')

    def __init__(self, op, args, line, text):
        self.op = op
        self.args = args
        self.line = line
        self.text = text

class Program(object):

    def __init__(self):
        self.name = None
        self.params = []
        self.instrs = []
        self.labels = {}
        self.block_labels = []
        self.block_of = []

def _parse_operand(kind, tok, line):
    if kind == 'd':
        if not REG_RE.match(tok):
            raise KirError("expected a register (v0, v1, ...) but found '%s'" % tok, line)
        return ('r', int(tok[1:]))
    if kind == 's':
        if REG_RE.match(tok):
            return ('r', int(tok[1:]))
        if LIT_RE.match(tok):
            return ('i', parse_literal_raw(tok, line) & MASK)
        raise KirError("expected a register or an integer literal but found '%s'" % tok, line)
    if kind == 'i':
        return ('i', parse_literal_raw(tok, line) & MASK)
    if kind == 'k':
        if REG_RE.match(tok):
            raise KirError('the last operand of addsh must be a literal in 0..31, not a register', line)
        n = parse_literal_raw(tok, line)
        if n < 0 or n > 31:
            raise KirError("shift amount of addsh must be a literal in 0..31, found '%s'" % tok, line)
        return ('k', n)
    if kind == 'L':
        if not NAME_RE.match(tok) or REG_RE.match(tok):
            raise KirError("expected a label name but found '%s'" % tok, line)
        return ('l', tok)
    raise AssertionError(kind)

def parse_program(text):
    prog = Program()
    seen_func = False
    cur_label = None
    blocks = []
    label_lines = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split(';', 1)[0].strip()
        if not line:
            continue
        m = FUNC_RE.match(line)
        if m:
            if seen_func:
                raise KirError("only one 'func' header is allowed", lineno)
            if blocks:
                raise KirError("'func' header must come before the first label", lineno)
            seen_func = True
            prog.name = m.group(1)
            plist = m.group(2).strip()
            params = []
            if plist:
                for tok in [t.strip() for t in plist.split(',')]:
                    if not REG_RE.match(tok):
                        raise KirError("parameter '%s' is not a register" % tok, lineno)
                    idx = int(tok[1:])
                    if idx in params:
                        raise KirError("parameter '%s' is listed twice" % tok, lineno)
                    params.append(idx)
            if len(params) > MAX_PARAMS:
                raise KirError('at most %d parameters are allowed' % MAX_PARAMS, lineno)
            prog.params = params
            continue
        if not seen_func:
            raise KirError("expected 'func name(params)' as the first statement", lineno)
        m = LABEL_RE.match(line)
        if m:
            name = m.group(1)
            if REG_RE.match(name):
                raise KirError("label '%s' looks like a register name" % name, lineno)
            if name in label_lines:
                raise KirError("label '%s' is defined twice (first on line %d)" % (name, label_lines[name]), lineno)
            label_lines[name] = lineno
            blocks.append((name, lineno, []))
            continue
        parts = line.split(None, 1)
        op = parts[0]
        if op not in SIGS:
            if op.endswith(':'):
                raise KirError('a label must be alone on its line', lineno)
            raise KirError("unknown instruction '%s'" % op, lineno)
        if not blocks:
            raise KirError('instruction outside of a block (write a label first)', lineno)
        rest = parts[1].strip() if len(parts) > 1 else ''
        toks = [t.strip() for t in rest.split(',')] if rest else []
        sig = SIGS[op]
        if len(toks) != len(sig):
            raise KirError("'%s' takes %d operand(s), found %d" % (op, len(sig), len(toks)), lineno)
        args = []
        for kind, tok in zip(sig, toks):
            if tok == '':
                raise KirError("empty operand in '%s'" % line, lineno)
            args.append(_parse_operand(kind, tok, lineno))
        blocks[-1][2].append(Instr(op, args, lineno, op + (' ' + ', '.join(toks) if toks else '')))
    if not seen_func:
        raise KirError("missing 'func name(params)' header")
    if not blocks:
        raise KirError('the function has no blocks')
    total = 0
    for label, lineno, instrs in blocks:
        if not instrs:
            raise KirError("block '%s' is empty" % label, lineno)
        for ins in instrs[:-1]:
            if ins.op in TERMINATORS:
                raise KirError("'%s' ends the block, but more instructions follow it" % ins.op, ins.line)
        if instrs[-1].op not in TERMINATORS:
            raise KirError("block '%s' must end with jmp, br, ret or trap" % label, instrs[-1].line)
        total += len(instrs)
    if total > MAX_INSTRS:
        raise KirError('program has %d instructions; the limit is %d' % (total, MAX_INSTRS))
    for label, lineno, instrs in blocks:
        prog.labels[label] = len(prog.instrs)
        prog.block_labels.append(label)
        for ins in instrs:
            prog.instrs.append(ins)
            prog.block_of.append(label)
    for ins in prog.instrs:
        for kind, val in ins.args:
            if kind == 'l' and val not in prog.labels:
                raise KirError("jump to undefined label '%s'" % val, ins.line)
    return prog

def read_text(path):
    with open(path, 'rb') as f:
        data = f.read()
    if data.startswith(b'\xef\xbb\xbf'):
        enc = 'utf-8-sig'
    elif data.startswith(b'\xff\xfe') or data.startswith(b'\xfe\xff'):
        enc = 'utf-16'
    else:
        enc = 'utf-8'
    return data.decode(enc)

def load_program(path):
    try:
        text = read_text(path)
    except (IOError, OSError, UnicodeDecodeError) as e:
        raise KirError('cannot read %s: %s' % (path, e))
    return parse_program(text)
MACHINE_KEYS = [('issue_width', 1), ('mem_ports', 1), ('registers', 4), ('reg_spill_penalty', 0), ('taken_branch_penalty', 0), ('icache_size', 1), ('icache_penalty', 0), ('lat_alu', 1), ('lat_mul', 1), ('lat_div', 1), ('lat_load', 1), ('lat_mac', 1), ('lat_addsh', 1), ('fused_mac', 0), ('fused_addsh', 0)]
MACHINE_MAX = 1000000

def load_machine(path):
    try:
        lines = read_text(path).splitlines()
    except (IOError, OSError, UnicodeDecodeError) as e:
        raise KirError('cannot read machine config %s: %s' % (path, e))
    known = dict(MACHINE_KEYS)
    mc = {}
    for lineno, raw in enumerate(lines, 1):
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        if '=' not in line:
            raise KirError("%s:%d: expected 'key = value'" % (path, lineno))
        key, val = [t.strip() for t in line.split('=', 1)]
        if key not in known:
            raise KirError("%s:%d: unknown key '%s'" % (path, lineno, key))
        if key in mc:
            raise KirError("%s:%d: key '%s' given twice" % (path, lineno, key))
        if not re.match('^[0-9]+$', val):
            raise KirError("%s:%d: value of '%s' must be a non-negative integer" % (path, lineno, key))
        n = int(val)
        if n < known[key] or n > MACHINE_MAX:
            raise KirError("%s:%d: value of '%s' must be in %d..%d" % (path, lineno, key, known[key], MACHINE_MAX))
        if key.startswith('fused_') and n > 1:
            raise KirError("%s:%d: '%s' must be 0 or 1" % (path, lineno, key))
        mc[key] = n
    for key, _ in MACHINE_KEYS:
        if key not in mc:
            raise KirError("%s: missing key '%s'" % (path, key))
    return mc

def check_machine_features(prog, mc):
    for ins in prog.instrs:
        if ins.op == 'mac' and (not mc['fused_mac']):
            raise KirError("'mac' is not available on this machine (fused_mac = 0)", ins.line)
        if ins.op == 'addsh' and (not mc['fused_addsh']):
            raise KirError("'addsh' is not available on this machine (fused_addsh = 0)", ins.line)

class Compiled(object):
    pass

def compile_program(prog):
    regmap = {}

    def rid(idx):
        if idx not in regmap:
            regmap[idx] = len(regmap)
        return regmap[idx]
    for p in prog.params:
        rid(p)
    for ins in prog.instrs:
        for kind, val in ins.args:
            if kind == 'r':
                rid(val)
    nregs = len(regmap)
    consts = {}

    def slot(operand):
        kind, val = operand
        if kind == 'r':
            return regmap[val]
        if val not in consts:
            consts[val] = nregs + len(consts)
        return consts[val]
    cp = Compiled()
    code, srcs, dst, opnames = ([], [], [], [])
    for ins in prog.instrs:
        op, a = (ins.op, ins.args)
        d = x = y = z = t1 = t2 = 0
        src_slots = []
        dest = -1
        if op == 'const':
            d, x = (slot(a[0]), slot(a[1]))
            dest = d
        elif op == 'mov':
            d, x = (slot(a[0]), slot(a[1]))
            dest = d
            src_slots = [x]
        elif op in ('add', 'sub', 'mul', 'and', 'or', 'xor', 'shl', 'shr', 'lt', 'eq', 'div', 'mod'):
            d, x, y = (slot(a[0]), slot(a[1]), slot(a[2]))
            dest = d
            src_slots = [x, y]
        elif op in ('select', 'mac'):
            d, x, y, z = (slot(a[0]), slot(a[1]), slot(a[2]), slot(a[3]))
            dest = d
            src_slots = [x, y, z]
        elif op == 'addsh':
            d, x, y, z = (slot(a[0]), slot(a[1]), slot(a[2]), a[3][1])
            dest = d
            src_slots = [x, y]
        elif op in ('load', 'vload'):
            d, x = (slot(a[0]), slot(a[1]))
            dest = d
            src_slots = [x]
        elif op in ('store', 'vstore'):
            x, y = (slot(a[0]), slot(a[1]))
            src_slots = [x, y]
        elif op in ('trap', 'ret'):
            x = slot(a[0])
            src_slots = [x]
        elif op == 'jmp':
            t1 = prog.labels[a[0][1]]
        elif op == 'br':
            x = slot(a[0])
            t1 = prog.labels[a[1][1]]
            t2 = prog.labels[a[2][1]]
            src_slots = [x]
        code.append((OPID[op], d, x, y, z, t1, t2))
        srcs.append(tuple((s for s in src_slots if s < nregs)))
        dst.append(dest)
        opnames.append(op)
    cp.code = code
    cp.srcs = srcs
    cp.dst = dst
    cp.opnames = opnames
    cp.touch = []
    for i in range(len(code)):
        seen, order = (set(), [])
        for r in list(srcs[i]) + ([dst[i]] if dst[i] >= 0 else []):
            if r not in seen:
                seen.add(r)
                order.append(r)
        cp.touch.append(tuple(order))
    cp.isterm = [n in TERMINATORS for n in opnames]
    cp.ismem = [n in MEM_OPS for n in opnames]
    cp.nregs = nregs
    cp.nparams = len(prog.params)
    cp.R0 = [0] * nregs + [v for v, _ in sorted(consts.items(), key=lambda kv: kv[1])]
    cp.text = [ins.text for ins in prog.instrs]
    cp.block_of = prog.block_of
    cp.name = prog.name
    return cp

class Result(object):
    __slots__ = ('status', 'value', 'vlog', 'mem', 'steps', 'trace')

    def __init__(self, status, value, vlog, mem, steps, trace):
        self.status = status
        self.value = value
        self.vlog = vlog
        self.mem = mem
        self.steps = steps
        self.trace = trace

def execute(cp, args, segments, vin, max_steps, want_trace=False):
    R = cp.R0[:]
    for i, v in enumerate(args):
        R[i] = v & MASK
    mem = [0] * MEM_WORDS
    for addr, words in segments:
        mem[addr:addr + len(words)] = words
    queues = dict(((p, deque(w)) for p, w in vin.items()))
    vlog = []
    code = cp.code
    tr = [] if want_trace else None
    pc = 0
    steps = 0
    while True:
        if steps >= max_steps:
            return Result('TIMEOUT', None, vlog, mem, steps, tr)
        op, d, a, b, c, t1, t2 = code[pc]
        steps += 1
        if want_trace:
            tr.append(pc)
        if op == ADD:
            R[d] = R[a] + R[b] & MASK
            pc += 1
        elif op == MOV:
            R[d] = R[a]
            pc += 1
        elif op == LT:
            R[d] = 1 if R[a] ^ SIGN < R[b] ^ SIGN else 0
            pc += 1
        elif op == BR:
            pc = t1 if R[a] != 0 else t2
        elif op == JMP:
            pc = t1
        elif op == SUB:
            R[d] = R[a] - R[b] & MASK
            pc += 1
        elif op == MUL:
            R[d] = R[a] * R[b] & MASK
            pc += 1
        elif op == AND:
            R[d] = R[a] & R[b]
            pc += 1
        elif op == OR:
            R[d] = R[a] | R[b]
            pc += 1
        elif op == XOR:
            R[d] = R[a] ^ R[b]
            pc += 1
        elif op == SHL:
            R[d] = R[a] << (R[b] & 31) & MASK
            pc += 1
        elif op == SHR:
            R[d] = R[a] >> (R[b] & 31)
            pc += 1
        elif op == EQ:
            R[d] = 1 if R[a] == R[b] else 0
            pc += 1
        elif op == SELECT:
            R[d] = R[b] if R[a] != 0 else R[c]
            pc += 1
        elif op == LOAD:
            addr = R[a]
            if addr >= MEM_WORDS:
                return Result('TRAP', 'bad_address', vlog, mem, steps, tr)
            R[d] = mem[addr]
            pc += 1
        elif op == STORE:
            addr = R[a]
            if addr >= MEM_WORDS:
                return Result('TRAP', 'bad_address', vlog, mem, steps, tr)
            mem[addr] = R[b]
            pc += 1
        elif op == MAC:
            R[d] = R[a] * R[b] + R[c] & MASK
            pc += 1
        elif op == ADDSH:
            R[d] = R[a] + (R[b] << c) & MASK
            pc += 1
        elif op == DIV or op == MOD:
            y = R[b]
            if y == 0:
                return Result('TRAP', 'div_by_zero', vlog, mem, steps, tr)
            x = R[a]
            sx = x - 4294967296 if x & SIGN else x
            sy = y - 4294967296 if y & SIGN else y
            if op == DIV:
                q = abs(sx) // abs(sy)
                if (sx < 0) != (sy < 0):
                    q = -q
                R[d] = q & MASK
            else:
                r = abs(sx) % abs(sy)
                if sx < 0:
                    r = -r
                R[d] = r & MASK
            pc += 1
        elif op == VLOAD:
            port = R[a]
            q = queues.get(port)
            val = q.popleft() if q else 0
            vlog.append(('L', port, val))
            R[d] = val
            pc += 1
        elif op == VSTORE:
            vlog.append(('S', R[a], R[b]))
            pc += 1
        elif op == TRAP:
            return Result('TRAP', 'explicit(%d)' % R[a], vlog, mem, steps, tr)
        else:
            return Result('RET', R[a], vlog, mem, steps, tr)

def time_trace(cp, trace, mc, issue_out=None):
    W, MP, K = (mc['issue_width'], mc['mem_ports'], mc['registers'])
    SP, TB = (mc['reg_spill_penalty'], mc['taken_branch_penalty'])
    IC, ICP = (mc['icache_size'], mc['icache_penalty'])
    lat = [mc[LATKEY[n]] if n in LATKEY else 1 for n in cp.opnames]
    srcs, dst, touch, isterm, ismem = (cp.srcs, cp.dst, cp.touch, cp.isterm, cp.ismem)
    ready = [0] * cp.nregs
    track = cp.nregs > K
    resident = OrderedDict()
    spilled = set()
    if track:
        for r in range(cp.nparams):
            if len(resident) >= K:
                spilled.add(resident.popitem(last=False)[0])
            resident[r] = True
    last = 0
    used = 0
    memused = 0
    fence = 0
    n = len(trace)
    for idx in range(n):
        pc = trace[idx]
        F = ICP if pc >= IC else 0
        if track:
            for r in touch[pc]:
                if r in resident:
                    resident.move_to_end(r)
                else:
                    if r in spilled:
                        F += SP
                        spilled.discard(r)
                    if len(resident) >= K:
                        spilled.add(resident.popitem(last=False)[0])
                    resident[r] = True
        t = last + F
        if fence > t:
            t = fence
        for r in srcs[pc]:
            if ready[r] > t:
                t = ready[r]
        d = dst[pc]
        if d >= 0 and ready[d] > t:
            t = ready[d]
        m = ismem[pc]
        if t == last and (used >= W or (m and memused >= MP)):
            t += 1
        if t > last:
            used = 0
            memused = 0
        used += 1
        if m:
            memused += 1
        last = t
        if d >= 0:
            ready[d] = t + lat[pc]
        if isterm[pc] and idx + 1 < n and (trace[idx + 1] != pc + 1):
            fence = t + 1 + TB
        if issue_out is not None:
            issue_out.append(t)
    return last + 1
