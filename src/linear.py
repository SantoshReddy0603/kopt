"""Collapse chains of additions and constant multiplications in one block.

Every value built only from add, sub, constant mul/shl, addsh, constant mac,
mov and const is a sum of block inputs times constants (modulo 2^32). When the
instructions that built such a value would all become unused, and the sum can
be written with fewer instructions, the value is recomputed directly from the
inputs. An input is used only while it still holds the value the sum refers to.
"""
from .ir import is_reg, reads, writes
from .kirlib import MASK

LIMIT = 3


def _scale(form, factor):
    terms, constant = form
    scaled = {key: coeff * factor & MASK for key, coeff in terms.items()}
    return {key: coeff for key, coeff in scaled.items() if coeff}, constant * factor & MASK


def _add(left, right, sign=1):
    terms = dict(left[0])
    for key, coeff in right[0].items():
        terms[key] = (terms.get(key, 0) + sign * coeff) & MASK
    return {key: coeff for key, coeff in terms.items() if coeff}, (left[1] + sign * right[1]) & MASK


def _power(value):
    return value.bit_length() - 1 if value and value & (value - 1) == 0 else None


def _cost(form, machine, multiply_last):
    terms, constant = form
    total = 0
    first = True
    for _, coeff in sorted(terms.items(), key=lambda item: item[1] != 1):
        if first:
            total += coeff != 1
            first = False
        elif coeff in (1, MASK):
            total += 1
        elif _power(coeff) is not None and machine["fused_addsh"]:
            total += 1
        elif machine["fused_mac"] and not multiply_last:
            total += 1
        else:
            total += 2
    total += bool(constant) or not terms
    return max(total, 1)


def _emit(form, destination, temporaries, machine, multiply_last):
    terms, constant = form
    order = sorted(terms.items(), key=lambda item: (item[1] != 1, item[0][1], item[0][0]))
    code = []
    current = None
    accumulator, scratch = temporaries
    for (register, _), coeff in order:
        if current is None:
            if coeff == 1:
                current = register
            else:
                shift = _power(coeff)
                code.append(("shl", (accumulator, register, shift)) if shift is not None
                            else ("mul", (accumulator, register, coeff)))
                current = accumulator
        elif coeff == 1:
            code.append(("add", (accumulator, current, register)))
            current = accumulator
        elif coeff == MASK:
            code.append(("sub", (accumulator, current, register)))
            current = accumulator
        elif _power(coeff) is not None and machine["fused_addsh"]:
            code.append(("addsh", (accumulator, current, register, _power(coeff))))
            current = accumulator
        elif machine["fused_mac"] and not multiply_last:
            code.append(("mac", (accumulator, register, coeff, current)))
            current = accumulator
        else:
            code.append(("mul", (scratch, register, coeff)))
            code.append(("add", (accumulator, current, scratch)))
            current = accumulator
    if current is None:
        code.append(("const", (accumulator, constant)))
    elif constant:
        code.append(("add", (accumulator, current, constant)))
    if not code:
        code.append(("mov", (accumulator, current)))
    op, args = code[-1]
    code[-1] = (op, (destination,) + args[1:])
    return code


def combine(body, machine, reads_anywhere, fresh, multiply_last=False):
    """Returns the new body and the number of fresh registers used."""
    version = {}
    forms = {}
    built = {}
    result = []
    used = 0

    def form_of(operand):
        if not is_reg(operand):
            return {}, operand & MASK
        if operand in forms:
            return forms[operand]
        return {(operand, version.get(operand, -1)): 1}, 0

    for index, (op, args) in enumerate(body):
        form = None
        sources = tuple(reads(op, args))
        if op == "const":
            form = {}, args[1] & MASK
        elif op == "mov":
            form = form_of(args[1])
        elif op == "add":
            form = _add(form_of(args[1]), form_of(args[2]))
        elif op == "sub":
            form = _add(form_of(args[1]), form_of(args[2]), -1)
        elif op in ("mul", "shl"):
            left, right = form_of(args[1]), form_of(args[2])
            if op == "shl":
                if not right[0]:
                    form = _scale(left, 1 << (right[1] & 31))
            elif not right[0]:
                form = _scale(left, right[1])
            elif not left[0]:
                form = _scale(right, left[1])
        elif op == "addsh":
            form = _add(form_of(args[1]), _scale(form_of(args[2]), 1 << args[3]))
        elif op == "mac":
            left, right = form_of(args[1]), form_of(args[2])
            product = None
            if not right[0]:
                product = _scale(left, right[1])
            elif not left[0]:
                product = _scale(right, left[1])
            if product is not None:
                form = _add(product, form_of(args[3]))
        replaced = None
        if form is not None and len(form[0]) <= LIMIT:
            count = 1 + sum(built.get(source, 0) for source in dict.fromkeys(sources)
                            if reads_anywhere.get(source, 0) == 1)
            valid = all(version.get(register, -1) == stamp for register, stamp in form[0])
            cost = _cost(form, machine, multiply_last)
            if valid and cost < count:
                temporaries = (fresh - used, fresh - used - 1)
                replaced = _emit(form, args[0], temporaries, machine, multiply_last)
                used += 2
                built[args[0]] = cost
        if writes(op):
            destination = args[0]
            if replaced is None:
                built[destination] = (
                    1 + sum(built.get(source, 0) for source in dict.fromkeys(sources)
                            if reads_anywhere.get(source, 0) == 1)
                    if form is not None else 0)
            version[destination] = index
            for register in [key for key, value in forms.items()
                             if any(item[0] == destination for item in value[0])]:
                del forms[register]
            if form is not None and all(item[0] != destination for item in form[0]):
                forms[destination] = form
            else:
                forms.pop(destination, None)
        result.extend(replaced if replaced is not None else [(op, args)])
    return result, used
