from collections import Counter

from .ir import is_reg, parents, reads, targets, writes
from .kirlib import LATKEY, MASK, SIGN
from .linear import combine
from .values import PURE
from .modular import defer


def signed(value):
    return value - (MASK + 1) if value & SIGN else value


def simple_loops(blocks):
    entry = next(iter(blocks))
    incoming = parents(blocks)
    for label, body in blocks.items():
        if label == entry or len(body) > 64 or len(incoming[label]) != 2:
            continue
        op, args = body[-1]
        if op != "br" or args[1] != label or args[2] == label or label not in incoming[label]:
            continue
        predecessor = next(pred for pred in incoming[label] if pred != label)
        if blocks[predecessor][-1] == ("jmp", (label,)):
            yield label, predecessor


ROTATE_LIMIT = 8


def _reaches(blocks, start, goal, avoid):
    seen = {start}
    todo = [start]
    while todo:
        label = todo.pop()
        if label == goal:
            return True
        for successor in targets(blocks[label][-1]):
            if successor not in seen and successor != avoid:
                seen.add(successor)
                todo.append(successor)
    return False


def _fresh_label(blocks, base):
    index = 0
    while "%s_pre%d" % (base, index) in blocks:
        index += 1
    return "%s_pre%d" % (base, index)


def rotate(blocks):
    """Copy a small loop test into every latch that jumps to it.

    The latch then branches straight back into the loop. A body that is a
    single block becomes a self loop, and gets an empty preheader so the
    single-block loop passes can place code in front of it.
    """
    entry = next(iter(blocks))
    result = dict(blocks)
    incoming = {label: list(preds) for label, preds in parents(blocks).items()}
    inserted = {}
    for head in list(blocks):
        test = result[head]
        if len(test) > ROTATE_LIMIT or test[-1][0] != "br":
            continue
        condition, taken, other = test[-1][1]
        if taken == other or head in (taken, other):
            continue
        latches = [label for label in dict.fromkeys(incoming[head])
                   if label != head and result[label][-1] == ("jmp", (head,))
                   and (_reaches(result, taken, label, head) or _reaches(result, other, label, head))
                   and (head == entry or not _reaches(result, entry, label, head))]
        outside = [label for label in incoming[head] if label not in latches]
        if not latches or (head != entry and not outside):
            continue
        for latch in latches:
            result[latch] = result[latch][:-1] + test
            incoming[head] = [label for label in incoming[head] if label != latch]
            incoming[taken].append(latch)
            incoming[other].append(latch)
        if latches == [taken] and sorted(incoming[taken]) == sorted((head, taken)):
            preheader = _fresh_label(result, taken)
            result[head] = test[:-1] + [("br", (condition, preheader, other))]
            result[preheader] = [("jmp", (taken,))]
            incoming[taken] = [preheader if label == head else label for label in incoming[taken]]
            incoming[preheader] = [head]
            inserted[head] = preheader
    if not inserted:
        return result
    ordered = {}
    for label in blocks:
        ordered[label] = result[label]
        if label in inserted:
            ordered[inserted[label]] = result[inserted[label]]
    return ordered


def entry_value(blocks, label, register, params, incoming):
    """The constant a register holds when control leaves `label`, if known.

    `incoming` may list more predecessors than exist, never fewer.
    """
    entry = next(iter(blocks))
    seen = set()
    while label not in seen:
        seen.add(label)
        for op, args in reversed(blocks[label]):
            if writes(op) and args[0] == register:
                if op in ("const", "mov") and not is_reg(args[1]):
                    return args[1]
                return None
        if label == entry:
            return None if incoming[label] or register in params else 0
        if len(incoming[label]) != 1:
            return None
        label = incoming[label][0]
    return None


def hoist(body):
    definitions = Counter(args[0] for op, args in body if writes(op))
    seen, lifted, kept = set(), [], []
    for op, args in body:
        sources = tuple(reads(op, args))
        if (op in PURE and definitions[args[0]] == 1 and args[0] not in seen
                and all(definitions[reg] == 0 for reg in sources)):
            lifted.append((op, args))
            definitions[args[0]] = 0
        else:
            kept.append((op, args))
        seen.update(sources)
    return lifted, kept


def trip_count(body, start):
    """Iterations of a counted self loop entered with `start` in its counter."""
    if len(body) < 3:
        return None
    op, compare = body[-2]
    cond = body[-1][1][0]
    if op != "lt" or compare[0] != cond or not is_reg(compare[1]) or is_reg(compare[2]):
        return None
    counter, limit = compare[1], signed(compare[2])
    updates = [(op, args) for op, args in body[:-1] if writes(op) and args[0] == counter]
    if len(updates) != 1:
        return None
    op, args = updates[0]
    if op != "add" or args[1] != counter or is_reg(args[2]):
        return None
    increment = signed(args[2])
    if increment <= 0 or start is None:
        return None
    start = signed(start)
    trips = max(1, (limit - start + increment - 1) // increment)
    if start + trips * increment >= SIGN:
        return None
    return trips


def counter_divisions(body, start):
    """Turn div/mod of a never-negative counter by a power of two into shr/and.

    The counter starts at a value in 0..2^31-1, is changed only by one
    `add c, c, k` with k > 0 and the loop continues only while `c < limit`
    (signed). Every value it holds inside the loop is then below 2^31: it was
    below the limit at the previous test, and adding k cannot pass 2^31 when
    k is 1 or the limit is a literal at most 2^31 - k.
    """
    counter = trip_counter(body)
    if counter is None or start is None or start >= SIGN:
        return body
    limit = body[-2][1][2]
    updates = [args for op, args in body[:-1] if writes(op) and args[0] == counter]
    if len(updates) != 1 or body[[args for _, args in body].index(updates[0])][0] != "add":
        return body
    _, source, step = updates[0]
    if source != counter or is_reg(step) or not 0 < step < SIGN:
        return body
    if step != 1 and (is_reg(limit) or signed(limit) + step > SIGN):
        return body
    result = []
    for op, args in body:
        if op in ("div", "mod") and args[1] == counter and not is_reg(args[2]):
            divisor = args[2]
            if 1 <= divisor < SIGN and divisor & (divisor - 1) == 0:
                if op == "div":
                    op, args = "shr", (args[0], counter, divisor.bit_length() - 1)
                else:
                    op, args = "and", (args[0], counter, divisor - 1)
        result.append((op, args))
    return result


def trip_counter(body):
    if len(body) >= 3 and body[-2][0] == "lt" and is_reg(body[-2][1][1]):
        return body[-2][1][1]
    return None


def unroll(body, preheader, start, headroom):
    """Returns the new preheader tail to insert and the new loop body."""
    trips = trip_count(body, start)
    if trips is None:
        return [], body
    size = len(body) - 1
    if 2 <= trips <= 8 and (trips - 1) * size <= headroom:
        return [], body[:-1] * trips + [("jmp", (body[-1][1][2],))]
    for factor in (8, 4, 2):
        rounds, rest = divmod(trips, factor)
        if rounds >= 2 and (factor - 1 + rest) * size <= headroom:
            return body[:-1] * rest, body[:-1] * factor + body[-1:]
    return [], body


GROUP_STEP_LIMIT = 1 << 28


def grouped_counter(body):
    """(counter, step, limit) of a self loop ending `add i, i, s; ...; lt c, i, n; br c`.

    The counter is written once, by that add; the step and the limit are
    literals or registers the body never writes.
    """
    if len(body) < 3 or body[-2][0] != "lt" or body[-1][0] != "br":
        return None
    cond, counter, limit = body[-2][1]
    if body[-1][1][0] != cond or not is_reg(counter) or cond in (counter, limit):
        return None
    written = Counter(args[0] for op, args in body[:-1] if writes(op))
    if written[counter] != 1 or (is_reg(limit) and written[limit]):
        return None
    for op, args in body[:-2]:
        if writes(op) and args[0] == counter:
            if op != "add" or args[1] != counter:
                return None
            step = args[2]
            if is_reg(step):
                if written[step] or step == counter:
                    return None
            elif not 0 < step < GROUP_STEP_LIMIT:
                return None
            return counter, step, limit
    return None


def group(body, preheader, label, fresh, headroom, copies=4):
    """Run `copies` iterations of a counted self loop under one test.

    With D = (copies - 1) * step and t = limit - D, an iteration that starts
    with the counter below t (signed) passes the next copies - 1 loop tests:
    the counter grows by a positive step and stays below the limit without
    wrapping. The preheader computes t once, setting it to the smallest
    signed value (so the test always fails) when the step is not in 1..2^28-1
    or limit - D would wrap. A group block holds the copies and loops while
    the counter is below t; otherwise the original loop finishes, entered
    through a block that repeats its own last test. Every executed
    instruction is one the original loop would execute, in the same order.
    Returns (preheader, group label -> block, registers used) or None.
    """
    found = grouped_counter(body)
    size = len(body) - 1
    if found is None or (copies - 1) * size > headroom:
        return None
    counter, step, limit = found
    cond, exit_label = body[-2][1][0], body[-1][1][2]
    regs = [fresh - k for k in range(8)]
    bound, test = regs[0], regs[1]
    tail = []
    if not is_reg(step):
        distance = (copies - 1) * step
        if is_reg(limit):
            tail += [("lt", (regs[2], limit, (SIGN + distance) & MASK)),
                     ("sub", (regs[3], limit, distance)),
                     ("select", (bound, regs[2], SIGN, regs[3]))]
        else:
            if signed(limit) - distance < -SIGN:
                return None
            tail += [("const", (bound, (limit - distance) & MASK))]
    else:
        distance, wide, positive, small, valid, fits, ok = regs[2:8] + [regs[3]]
        tail += [("mul", (distance, step, copies - 1)),
                 ("lt", (positive, 0, step)),
                 ("lt", (small, step, GROUP_STEP_LIMIT)),
                 ("and", (valid, positive, small)),
                 ("add", (wide, distance, SIGN)),
                 ("lt", (fits, limit, wide)),
                 ("sub", (distance, limit, distance)),
                 ("select", (distance, fits, SIGN, distance)),
                 ("select", (bound, valid, distance, SIGN))]
    name = "%s_group" % label
    rest = "%s_rest" % label
    tail += [("lt", (test, counter, bound)), ("br", (test, name, label))]
    blocks = {
        name: body[:-1] * copies + [("lt", (test, counter, bound)), ("br", (test, name, rest))],
        rest: [("br", (cond, label, exit_label))],
    }
    return preheader[:-1] + tail, blocks, 8


def induction(body, machine, scratch):
    definitions = Counter(args[0] for op, args in body if writes(op))
    for i, (op, args) in enumerate(body[:-1]):
        if op != "mul" or machine["lat_mul"] <= machine["lat_alu"] or machine["issue_width"] < 2:
            continue
        dest, index, factor = args
        if is_reg(factor):
            index, factor = factor, index
        if not is_reg(index) or is_reg(factor) or definitions[index] != 1 or definitions[dest] != 1:
            continue
        for update_op, update in body[i + 1:-1]:
            if (update_op == "add" and update[0] == index and update[1] == index
                    and not is_reg(update[2])):
                delta = (factor * update[2]) & MASK
                start = ("mul", (scratch, index, factor))
                result = body[:i] + [("mov", (dest, scratch))] + body[i + 1:-1]
                result += [("add", (scratch, scratch, delta)), body[-1]]
                return [start], result
    return [], body


def optimize_loops(blocks, params, machine, profile, disabled=frozenset()):
    result = dict(blocks)
    total = sum(map(len, blocks.values()))
    if total > 30000:
        return result
    used = set(params)
    for body in blocks.values():
        used.update(arg for _, args in body for arg in args if is_reg(arg))
    fresh = min(used, default=0) - 1
    if "rotate" not in disabled:
        blocks = rotate(blocks)
        if "modular" not in disabled:
            blocks = defer(blocks)
        result = dict(blocks)
        total = sum(map(len, blocks.values()))
    reads_anywhere = Counter(register for body in blocks.values()
                             for op, args in body for register in reads(op, args))
    candidates = list(simple_loops(blocks))
    incoming = parents(blocks)
    candidates.sort(key=lambda pair: -profile.get((pair[0], pair[0]), 0))
    added = {}
    for label, predecessor in candidates:
        old = result[label]
        before = result[predecessor]
        lifted, body = hoist(old)
        if lifted:
            before = before[:-1] + lifted + before[-1:]
        counter = trip_counter(body)
        start = None
        if counter is not None:
            start = entry_value({**result, predecessor: before}, predecessor, counter, params, incoming)
            body = counter_divisions(body, start)
        if "linear" not in disabled and ~fresh + 2 * len(body) <= 999999:
            body, count = combine(body, machine, reads_anywhere, fresh, multiply_last=True)
            if count:
                used.update(range(fresh - count + 1, fresh + 1))
                fresh -= count
        if machine["taken_branch_penalty"]:
            headroom = machine["icache_size"] - total
            if not machine["icache_penalty"]:
                headroom = max(headroom, 64)
            if "narrow" in disabled:
                headroom += len(body) * 8
            peeled, unrolled = unroll(body, before, start, headroom)
        else:
            peeled, unrolled = [], body
        plain = body
        if unrolled is not body:
            total += len(unrolled) + len(peeled) - len(body)
            body = unrolled
            before = before[:-1] + peeled + before[-1:]
        elif (~fresh <= 999999 and len(used) < machine["registers"]
                and (not machine["icache_penalty"] or total + 2 <= machine["icache_size"])):
            seed, body = induction(body, machine, fresh)
            if seed:
                before = before[:-1] + seed + before[-1:]
                used.add(fresh)
                fresh -= 1
                total += 2
        fits = ~fresh + 8 <= 999999 and not {label + "_group", label + "_rest"} & set(result)
        if unrolled is plain and "group" not in disabled and fits:
            room = machine["icache_size"] - total if machine["icache_penalty"] else 4 * len(body)
            grouped = group(body, before, label, fresh, room)
            if grouped is not None:
                before, extra, count = grouped
                used.update(range(fresh - count + 1, fresh + 1))
                fresh -= count
                total += sum(map(len, extra.values())) + len(before) - len(result[predecessor])
                added[label] = extra
        result[predecessor] = before
        result[label] = body
    if not added:
        return result
    ordered = {}
    for label, body in result.items():
        ordered[label] = body
        ordered.update(added.get(label, {}))
    return ordered
