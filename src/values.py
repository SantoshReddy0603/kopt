import operator
from collections import deque

from .ir import is_reg, reads, targets, writes
from .kirlib import MASK, MEM_WORDS, SCRATCH_BASE, SIGN


BINARY = {
    "add": operator.add, "sub": operator.sub, "mul": operator.mul,
    "and": operator.and_, "or": operator.or_, "xor": operator.xor,
    "eq": operator.eq, "lt": lambda a, b: (a ^ SIGN) < (b ^ SIGN),
    "shl": lambda a, b: a << (b & 31), "shr": lambda a, b: a >> (b & 31),
}
COMMUTATIVE = frozenset(("add", "mul", "and", "or", "xor", "eq"))
PURE = frozenset(BINARY) | {"const", "mov", "select", "mac", "addsh"}


def fold(op, args):
    if op in BINARY:
        return int(BINARY[op](*args)) & MASK
    if op in ("const", "mov"):
        return args[0]
    if op == "select":
        cond, left, right = args
        return left if cond else right
    if op in ("mac", "addsh"):
        a, b, c = args
        return (a * b + c if op == "mac" else a + (b << c)) & MASK
    if op in ("div", "mod"):
        a, b = (arg - (MASK + 1) if arg & SIGN else arg for arg in args)
        if b:
            if op == "mod":
                result = abs(a) % abs(b)
                return (-result if a < 0 else result) & MASK
            result = abs(a) // abs(b)
            return (-result if (a < 0) != (b < 0) else result) & MASK
    return None


def global_constants(blocks, params, assigned):
    if len(blocks) == 1:
        return {}
    entry = next(iter(blocks))
    tracked = {}
    for body in blocks.values():
        for op, args in body:
            if writes(op) and len(tracked) < 64:
                tracked[args[0]] = None
            if len(tracked) == 64:
                break
        if len(tracked) == 64:
            break
    if len(blocks) * len(tracked) > 1000000:
        return {}
    initial = {reg: 0 for reg in tracked if reg not in params}
    incoming = {entry: initial}
    queue = deque([entry])
    pending = {entry}
    budget = 6 * sum(map(len, blocks.values())) + 128
    while queue:
        label = queue.popleft()
        pending.remove(label)
        body = blocks[label]
        budget -= len(body) + 1
        if budget < 0:
            return {}
        constants = incoming[label].copy()
        for op, args in body[:-1]:
            if not writes(op) or args[0] not in tracked:
                continue
            sources = tuple(constants.get(a, None if a in assigned else 0) if is_reg(a) else a
                            for a in args[1:])
            value = fold(op, sources) if None not in sources else None
            if value is None:
                constants.pop(args[0], None)
            else:
                constants[args[0]] = value
        for dest in targets(body[-1]):
            old = incoming.get(dest)
            updated = constants if old is None else {r: v for r, v in old.items() if constants.get(r) == v}
            if updated != old:
                incoming[dest] = updated
                if dest not in pending:
                    pending.add(dest)
                    queue.append(dest)
    return incoming


class Values:
    def __init__(self, initial, assigned):
        self.registers = dict(initial)
        self.assigned = assigned
        self.holders = {}
        self.next_value = MASK + 1
        self.expressions = {}
        self.origins = {}

    def fresh(self):
        value = self.next_value
        self.next_value += 1
        return value

    def get(self, operand):
        if not is_reg(operand):
            return operand
        if operand not in self.registers:
            value = self.fresh() if operand in self.assigned else 0
            self.registers[operand] = value
            if value > MASK:
                self.holders[value] = {operand: None}
        return self.registers[operand]

    def holder(self, value):
        if value <= MASK:
            return value
        return next(iter(self.holders.get(value, ())), None)

    def write(self, register, value):
        old = self.get(register)
        if old != value:
            if old > MASK:
                del self.holders[old][register]
            self.registers[register] = value
            if value > MASK:
                self.holders.setdefault(value, {})[register] = None
        return old != value


def identity(op, args, values):
    if op == "select":
        cond, yes, no = args
        if cond <= MASK or yes == no:
            return yes if cond else no
    if op not in BINARY and op not in ("div", "mod"):
        return None
    a, b = args
    if op in ("add", "or", "xor") and (a == 0 or b == 0):
        return a or b
    if op in ("and", "mul") and (a == 0 or b == 0):
        return 0
    if op == "mul" and (a == 1 or b == 1):
        return b if a == 1 else a
    if op == "and" and (a == MASK or b == MASK):
        return b if a == MASK else a
    if a == b:
        if op in ("sub", "xor", "lt"):
            return 0
        if op == "eq":
            return 1
        if op in ("or", "and"):
            return a
    if op == "sub" and b == 0 or op in ("shl", "shr") and b <= MASK and not b & 31:
        return a
    if op in ("shl", "shr") and a == 0:
        return 0
    if op == "div" and b == 1:
        return a
    if op == "mod" and b in (1, MASK):
        return 0
    if op == "sub" and a == 0:
        inner = values.origins.get(b)
        if inner is not None and inner[0] == "sub" and inner[1][0] == 0:
            return inner[1][1]
    prior = values.origins.get(a)
    if op == "sub" and prior is not None and prior[0] == "add":
        left, right = prior[1]
        if b == left:
            return right
        if b == right:
            return left
    if op == "add":
        for result, other in ((a, b), (b, a)):
            prior = values.origins.get(result)
            if prior is not None and prior[0] == "sub" and prior[1][1] == other:
                return prior[1][0]
    return None


def strength(op, args, machine):
    if op != "mul":
        return op, args
    dest, left, right = args
    if is_reg(right):
        left, right = right, left
    if is_reg(right) or not right:
        return op, args
    if right & (right - 1) == 0 and machine["lat_alu"] <= machine["lat_mul"]:
        return "shl", (dest, left, right.bit_length() - 1)
    if (right > 2 and (right - 1) & (right - 2) == 0 and machine["fused_addsh"]
            and machine["lat_addsh"] <= machine["lat_mul"]):
        return "addsh", (dest, left, left, (right - 1).bit_length() - 1)
    return op, args


def simplify_block(body, initial, assigned, machine, combine):
    values = Values(initial, assigned)
    memory = {}
    unknown_memory = {}
    pending_scratch = {}
    output = []
    for op, args in body:
        destination = args[0] if writes(op) else None
        inputs = args[1:] if destination is not None else args
        evaluated = tuple(values.get(arg) for arg in inputs)
        sources = tuple(values.holder(value) if is_reg(arg) else arg
                        for arg, value in zip(inputs, evaluated))
        if destination is None:
            if op == "br" and not is_reg(sources[0]):
                op, sources = "jmp", (sources[1 if sources[0] else 2],)
            elif op == "store":
                address, value = evaluated
                unknown_memory.clear()
                if address < MEM_WORDS:
                    memory[address] = value
                    if SCRATCH_BASE <= address < MEM_WORDS:
                        previous = pending_scratch.get(address)
                        if previous is not None:
                            output[previous] = None
                        pending_scratch[address] = len(output)
                elif address > MASK:
                    memory.clear()
                    pending_scratch.clear()
                    unknown_memory[address] = value
            elif op == "vstore":
                memory.clear()
                unknown_memory.clear()
                pending_scratch.clear()
            elif op in ("ret", "trap"):
                for previous in pending_scratch.values():
                    output[previous] = None
            output.append((op, sources))
            continue

        value = None
        if op == "vload":
            memory.clear()
            unknown_memory.clear()
            pending_scratch.clear()
        elif op == "load":
            address = evaluated[0]
            if address <= MASK:
                value = memory.get(address)
                pending_scratch.pop(address, None)
            else:
                value = unknown_memory.get(address)
                pending_scratch.clear()
        elif op in ("mov", "const"):
            value = evaluated[0]
        elif all(arg <= MASK for arg in evaluated):
            value = fold(op, evaluated)
        else:
            value = identity(op, evaluated, values)

        safe = op in PURE or op in ("div", "mod") and 0 < evaluated[1] <= MASK
        key = None
        if value is None and safe:
            key = (op, tuple(sorted(evaluated)) if op in COMMUTATIVE else evaluated)
            value = values.expressions.get(key)

        replacement = values.holder(value) if value is not None else None
        if value is not None and (replacement is not None or values.get(destination) == value):
            if values.write(destination, value):
                output.append(("mov" if is_reg(replacement) else "const", (destination, replacement)))
            continue
        if value is None:
            value = values.fresh()
        if key is not None:
            values.expressions[key] = value
            values.origins[value] = (op, evaluated)
        if op == "load":
            address = evaluated[0]
            if address < MEM_WORDS:
                memory[address] = value
            elif address > MASK:
                unknown_memory[address] = value
        values.write(destination, value)
        instruction = (op, (destination,) + sources)
        output.append(strength(*instruction, machine) if combine else instruction)
    return [instruction for instruction in output if instruction is not None]


def simplify(blocks, params, machine, combine=True):
    assigned = set(params)
    for body in blocks.values():
        assigned.update(args[0] for op, args in body if writes(op))
    facts = global_constants(blocks, params, assigned)
    entry = next(iter(blocks))
    initial_entry = not any(entry in targets(body[-1]) for body in blocks.values())
    result = {}
    for label, body in blocks.items():
        initial = facts.get(label, {})
        if label == entry and initial_entry:
            initial = {reg: 0 for op, args in body for reg in reads(op, args) if reg not in params}
        result[label] = simplify_block(body, initial, assigned, machine, combine)
    return result
