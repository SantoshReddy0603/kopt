from collections import deque
from dataclasses import dataclass
from operator import add, and_, eq, mul, or_, sub, xor


WORD_MASK = (1 << 32) - 1
SIGN_BIT = 1 << 31
MEMORY_WORDS = 4096
DATA_WORDS = 3072


def signed(word):
    return word - (1 << 32) if word & SIGN_BIT else word


def quotient(a, b):
    magnitude = abs(a) // abs(b)
    return -magnitude if (a < 0) != (b < 0) else magnitude


def read(operand, registers):
    return registers[~operand] if operand < 0 else operand


BINARY = {
    "add": add, "sub": sub, "mul": mul, "and": and_, "or": or_, "xor": xor,
    "shl": lambda a, b: a << (b & 31),
    "shr": lambda a, b: a >> (b & 31),
    "lt": lambda a, b: signed(a) < signed(b), "eq": eq,
}

WRITES_REGISTER = frozenset(BINARY) | {
    "const", "mov", "select", "div", "mod", "mac", "addsh", "load", "vload"
}


@dataclass
class Result:
    status: str
    value: object
    events: list
    memory: list
    steps: int


class Interpreter:
    def __init__(self, program):
        slots = {}

        def slot(register):
            return slots.setdefault(register, len(slots))

        self.param_slots = [slot(register) for register in program.params]
        self.code = []
        for instruction in program.instrs:
            op, operands = instruction.op, instruction.args
            dest = None
            if op in WRITES_REGISTER:
                dest = slot(operands[0][1])
                operands = operands[1:]
            values = []
            for kind, operand in operands:
                if kind == "r":
                    values.append(~slot(operand))
                elif kind == "l":
                    values.append(program.labels[operand])
                else:
                    values.append(operand)
            self.code.append((op, dest, tuple(values)))
        self.slot_count = len(slots)

    def execute(self, inputs, limit):
        registers = [0] * self.slot_count
        for slot, value in zip(self.param_slots, inputs.args):
            registers[slot] = value
        memory = [0] * MEMORY_WORDS
        for address, words in inputs.segments:
            memory[address:address + len(words)] = words
        devices = {port: deque(words) for port, words in inputs.vin.items()}
        events = []
        pc = 0
        code = self.code

        def finish(status, value, steps):
            return Result(status, value, events, memory[:DATA_WORDS], steps)

        for step in range(1, limit + 1):
            op, destination, operands = code[pc]
            pc += 1
            if op == "jmp":
                pc = operands[0]
                continue
            first = read(operands[0], registers)
            if op == "br":
                pc = operands[1] if first else operands[2]
                continue
            if op == "ret":
                return finish("RET", first, step)
            if op == "trap":
                return finish("TRAP", "explicit({})".format(first), step)
            second = read(operands[1], registers) if len(operands) > 1 else 0
            operation = BINARY.get(op)
            if operation is not None:
                value = operation(first, second)
            elif op in ("const", "mov"):
                value = first
            elif op in ("div", "mod"):
                if second == 0:
                    return finish("TRAP", "div_by_zero", step)
                numerator, divisor = signed(first), signed(second)
                divided = quotient(numerator, divisor)
                value = divided if op == "div" else numerator - divided * divisor
            elif op == "load":
                if first >= MEMORY_WORDS:
                    return finish("TRAP", "bad_address", step)
                value = memory[first]
            elif op == "store":
                if first >= MEMORY_WORDS:
                    return finish("TRAP", "bad_address", step)
                memory[first] = second
                continue
            elif op == "vload":
                queue = devices.get(first)
                value = queue.popleft() if queue else 0
                events.append(("L", first, value))
            elif op == "vstore":
                events.append(("S", first, second))
                continue
            elif op == "mac":
                value = first * second + read(operands[2], registers)
            elif op == "addsh":
                value = first + (second << operands[2])
            elif op == "select":
                value = second if first else read(operands[2], registers)
            else:
                raise ValueError("unexpected opcode {!r}".format(op))
            registers[destination] = value & WORD_MASK

        return finish("TIMEOUT", None, limit)


def difference(original, candidate):
    if (original.status, original.value) != (candidate.status, candidate.value):
        return "outcome differs: {} {} vs {} {}".format(
            original.status, original.value, candidate.status, candidate.value)
    if original.events != candidate.events:
        for index, (left, right) in enumerate(zip(original.events, candidate.events)):
            if left != right:
                return "volatile event {} differs: {} vs {}".format(index, left, right)
        return "volatile log length differs: {} vs {}".format(
            len(original.events), len(candidate.events))
    if original.memory != candidate.memory:
        for address, (left, right) in enumerate(zip(original.memory, candidate.memory)):
            if left != right:
                return "memory word {} differs: {} vs {}".format(address, left, right)
    return None
