from .kirlib import SIGS


def from_program(program):
    blocks = {label: [] for label in program.block_labels}
    for label, instruction in zip(program.block_of, program.instrs):
        args = tuple(~value if kind == "r" else value for kind, value in instruction.args)
        blocks[label].append((instruction.op, args))
    return blocks


def render(name, params, blocks):
    header = ", ".join("v%d" % ~param for param in params)
    lines = ["func %s(%s)" % (name, header)]
    for label, body in blocks.items():
        lines.append(label + ":")
        for op, args in body:
            operands = ("v%d" % ~a if is_reg(a) else str(a) for a in args)
            lines.append("  %s %s" % (op, ", ".join(operands)))
    return "\n".join(lines) + "\n"


def is_reg(arg):
    return isinstance(arg, int) and arg < 0


def writes(op):
    return SIGS[op][0] == "d"


def reads(op, args):
    return (arg for arg in args[writes(op):] if is_reg(arg))


def targets(instruction):
    op, args = instruction
    if op == "jmp":
        return args
    if op == "br":
        return args[1:]
    return ()


def parents(blocks):
    result = {label: [] for label in blocks}
    for label, body in blocks.items():
        for dest in dict.fromkeys(targets(body[-1])):
            result[dest].append(label)
    return result


def read_profile(text):
    edges = {}
    for raw in text.splitlines():
        parts = raw.split("#", 1)[0].split()
        if len(parts) == 4 and parts[0] == "edge" and parts[3].isascii() and parts[3].isdecimal():
            count = parts[3].lstrip("0") or "0"
            edges[parts[1], parts[2]] = int(count) if len(count) <= 12 else 10**12
    return edges
