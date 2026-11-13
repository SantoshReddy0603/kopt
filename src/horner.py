"""Split serial Horner chains into independent chains in a power of x.

A chain inside one block is a run of steps

    r1 = r0 * x + t1,  r2 = r1 * x + t2,  ...,  rn = r(n-1) * x + tn

written as `mul m, r, x` followed by `add r', m, t` (or one `mac`), with the
same value of x in every step. Its value is

    rn = sum over i of t_i * x^(n-i)        (t_0 = r0, the start value)

so every step waits for the previous multiply. With y = x^k the terms whose
exponent e has e mod k = j form their own Horner chain in y:

    C_j = sum over those terms of t_i * y^(e div k)

and rn = C_0 + C_1*x + ... + C_(k-1)*x^(k-1). The k chains run side by side
and are joined at the end (for k = 4 as (C3*x + C2)*x^2 + (C1*x + C0)).
32-bit wrapping arithmetic is a commutative ring, so the result is exactly
the same for every input. A literal 0 start value contributes nothing.

Safety. Every replaced value (each intermediate r and each multiply result)
must be read only by the next step of the chain and must not be live after
the block. A term defined in the block and read only by its step gets a
fresh destination so it survives until its chain uses it; other terms are
read where they still hold the value or copied. x must not be written
between the first multiply and the last step. Only pure instructions are
replaced or added, so memory, volatile events and traps keep their order.

Choice of k. The estimated latency of the split form,
lat_mul * log2(k) + (ceil(T/k) - 1) * s + join, with s the latency of one
step and T the number of terms, is compared over k = 1, 2, 4; k is limited
so that k accumulators, x, y and a term fit in the machine's registers.
Chains of fewer than four steps are left alone.
"""
from .flow import live_out
from .ir import is_reg, reads, writes

MIN_STEPS = 4
BLOCK_LIMIT = 4000


def _values(body):
    """Value ids of each instruction's register operands, and their readers."""
    current = {}
    operands = []
    users = {}
    for index, (op, args) in enumerate(body):
        found = {}
        for position in range(writes(op), len(args)):
            arg = args[position]
            if is_reg(arg):
                value = current.get(arg, ("in", arg))
                found[position] = value
                users.setdefault(value, []).append(index)
        operands.append(found)
        if writes(op):
            current[args[0]] = index
    return operands, users, current


def _operand(index, position, body, operands):
    arg = body[index][1][position]
    return operands[index][position] if is_reg(arg) else ("lit", arg)


def _steps(body, operands, users, final, out):
    """Candidate steps: index of the result -> list of (prev, x, term, parts)."""
    def private(value):
        if len(users.get(value, ())) != 1:
            return False
        reg = body[value][1][0]
        return not (final.get(reg) == value and reg in out)

    found = {}
    for index, (op, args) in enumerate(body):
        options = []
        if op == "mac":
            a, b, c = (_operand(index, p, body, operands) for p in (1, 2, 3))
            options = [(a, b, c, (index,)), (b, a, c, (index,))]
        elif op == "add":
            for side in (1, 2):
                product = operands[index].get(side)
                if not isinstance(product, int) or body[product][0] != "mul" or not private(product):
                    continue
                if not is_reg(args[side]) or args[3 - side] == args[side]:
                    continue
                term = _operand(index, 3 - side, body, operands)
                a = _operand(product, 1, body, operands)
                b = _operand(product, 2, body, operands)
                options += [(a, b, term, (product, index)), (b, a, term, (product, index))]
                break
        if options:
            found[index] = options
    return found, private


def _chains(body, found, private):
    chains = []          # each: {"x": value or None, "steps": [(index, option)], "options": [...]}
    owner = {}
    for index in sorted(found):
        linked = False
        for option in found[index]:
            prev, x = option[0], option[1]
            if not isinstance(prev, int) or prev not in owner or not private(prev):
                continue
            chain = owner[prev]
            if chain["steps"][-1][0] != prev:
                continue
            if len(chain["steps"]) == 1:
                first = [o for o in chain["options"] if o[1] == x]
                if not first:
                    continue
                chain["steps"][0] = (prev, first[0])
                chain["x"] = x
            elif chain["x"] != x:
                continue
            chain["steps"].append((index, option))
            owner[index] = chain
            linked = True
            break
        if not linked:
            chain = {"x": None, "steps": [(index, found[index][0])], "options": found[index]}
            chains.append(chain)
            owner[index] = chain
    return [chain for chain in chains if len(chain["steps"]) >= MIN_STEPS]


def _cost(terms, k, machine, step):
    if k == 1:
        return (terms - 1) * step
    join = step if k == 2 else 2 * step
    return machine["lat_mul"] * (k.bit_length() - 1) + (-(-terms // k) - 1) * step + join


def _choose(terms, machine, step):
    best = 1
    for k in (2, 4):
        if k + k.bit_length() + 2 > machine["registers"] or terms < 2 * k:
            continue
        if _cost(terms, k, machine, step) < _cost(terms, best, machine, step):
            best = k
    return best


def _split_block(body, out, machine, fresh):
    if not 2 * MIN_STEPS <= len(body) <= BLOCK_LIMIT:
        return body
    operands, users, final = _values(body)
    found, private = _steps(body, operands, users, final, out)
    if len(found) < MIN_STEPS:
        return body
    fused = machine["fused_mac"] and machine["lat_mac"] <= machine["lat_mul"] + machine["lat_alu"]
    step = machine["lat_mac"] if fused else machine["lat_mul"] + machine["lat_alu"]

    removed = set()
    renamed = {}         # index -> new destination
    before = {}          # index -> instructions emitted before it
    replace = {}         # index -> instructions emitted instead of it
    claimed = set()

    def new():
        reg = fresh[0]
        fresh[0] -= 1
        return reg

    for chain in _chains(body, found, private):
        steps = chain["steps"]
        parts = [i for _, option in steps for i in option[3]]
        if claimed & set(parts):
            continue
        x = chain["x"]
        first, last = min(parts), steps[-1][0]
        # x must hold the same value from the first multiply to the last step.
        if isinstance(x, tuple) and x[0] == "lit":
            xarg = x[1]
        else:
            xarg = body[x][1][0] if isinstance(x, int) else x[1]
            if any(writes(op) and args[0] == xarg for op, args in body[first:last + 1]):
                continue
        start = steps[0][1][0]
        terms = [] if start == ("lit", 0) else [(start, steps[0][1][3][0])]
        terms += [(option[2], index) for index, option in steps]
        k = _choose(len(terms), machine, step)
        if k == 1:
            continue

        claimed.update(parts)
        removed.update(parts)
        sources = {}

        def operand(value, at, read_until):
            """An argument holding `value` from position `at` to `read_until`."""
            if isinstance(value, tuple) and value[0] == "lit":
                return value[1]
            if isinstance(value, tuple):
                reg = value[1]
                defined = -1
            else:
                reg = body[value][1][0]
                defined = value
            if value not in sources:
                if isinstance(value, int) and len(users.get(value, [])) == 1 and value not in removed:
                    holder = body[value][1][0]
                    if not (final.get(holder) == value and holder in out):
                        renamed[value] = new()
                        sources[value] = renamed[value]
                        return sources[value]
                clobbered = any(writes(body[i][0]) and body[i][1][0] == reg and i not in removed
                                for i in range(max(defined + 1, at), read_until))
                if not clobbered:
                    sources[value] = reg
                else:
                    copy = new()
                    before.setdefault(at, []).append(("mov", (copy, reg)))
                    sources[value] = copy
            return sources[value]

        def emit(at, dest, a, b, c):
            if fused:
                code = [("mac", (dest, a, b, c))]
            else:
                product = new()
                code = [("mul", (product, a, b)), ("add", (dest, product, c))]
            replace.setdefault(at, []).extend(code)

        # Powers of x, computed where the first multiply was.
        if isinstance(xarg, int) and not is_reg(xarg):
            square = (xarg * xarg) & 0xFFFFFFFF
            power = square if k == 2 else (square * square) & 0xFFFFFFFF
        else:
            square = new()
            before.setdefault(first, []).append(("mul", (square, xarg, xarg)))
            power = square
            if k == 4:
                power = new()
                before.setdefault(first, []).append(("mul", (power, square, square)))

        count = len(terms)
        accumulators = {}
        pending = {}         # chain -> (value, position) of a first term not yet used
        for position, (value, at) in enumerate(terms):
            lane = (count - 1 - position) % k
            if lane not in accumulators and lane not in pending:
                pending[lane] = (value, at)
                continue
            if lane in pending:
                head, head_at = pending.pop(lane)
                accumulator = operand(head, head_at, at + 1)
            else:
                accumulator = accumulators[lane]
            dest = new()
            emit(at, dest, accumulator, power, operand(value, at, at + 1))
            accumulators[lane] = dest

        def lane_value(lane):
            if lane in accumulators:
                return accumulators[lane]
            head, head_at = pending.pop(lane)
            return operand(head, head_at, last + 1)

        result = body[last][1][0]
        if k == 2:
            emit(last, result, lane_value(1), xarg, lane_value(0))
        else:
            high, low = new(), new()
            emit(last, high, lane_value(3), xarg, lane_value(2))
            emit(last, low, lane_value(1), xarg, lane_value(0))
            emit(last, result, high, square, low)

    if not removed:
        return body
    output = []
    for index, (op, args) in enumerate(body):
        output.extend(before.get(index, ()))
        if index in removed:
            output.extend(replace.get(index, ()))
            continue
        if index in renamed:
            args = (renamed[index],) + args[1:]
        output.append((op, args))
        output.extend(replace.get(index, ()))
    return output


def split_chains(blocks, params, machine):
    if not any(op in ("mul", "mac") for body in blocks.values() for op, _ in body):
        return blocks
    out = live_out(blocks)
    if out is None:
        return blocks
    used = set(params)
    for body in blocks.values():
        used.update(arg for _, args in body for arg in args if is_reg(arg))
    fresh = [min(used, default=0) - 1]
    if ~fresh[0] > 999000:
        return blocks
    result = {label: _split_block(body, out[label], machine, fresh) for label, body in blocks.items()}
    if all(result[label] is body for label, body in blocks.items()):
        return blocks
    return result
