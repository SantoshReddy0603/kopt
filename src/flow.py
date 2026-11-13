from collections import deque
from heapq import heappop, heappush

from .ir import parents, reads, targets, writes


def has_short_exit(blocks, limit):
    entry = next(iter(blocks))
    pending = [(0, entry)]
    distances = {entry: 0}
    remaining = 8192
    while pending and remaining:
        distance, label = heappop(pending)
        if distance != distances[label]:
            continue
        remaining -= 1
        body = blocks[label]
        distance += len(body)
        if distance > limit:
            continue
        successors = targets(body[-1])
        if not successors:
            return True
        for dest in successors:
            if distance < distances.get(dest, limit + 1):
                distances[dest] = distance
                heappush(pending, (distance, dest))
    return False


def clean(blocks):
    entry = next(iter(blocks))
    redirects = {}
    for label in blocks:
        current = label
        path = {}
        while current not in redirects and current not in path:
            path[current] = None
            body = blocks[current]
            if len(body) != 1 or body[0][0] != "jmp":
                break
            current = body[0][1][0]
        final = redirects.get(current, current)
        redirects.update((item, final) for item in path)

    reachable = {}
    todo = [entry]
    while todo:
        label = todo.pop()
        if label in reachable:
            continue
        body = blocks[label]
        op, args = body[-1]
        if op == "jmp":
            tail = ("jmp", (redirects[args[0]],))
        elif op == "br":
            left, right = redirects[args[1]], redirects[args[2]]
            tail = ("jmp", (left,)) if left == right else ("br", (args[0], left, right))
        else:
            tail = body[-1]
        reachable[label] = body if tail == body[-1] else body[:-1] + [tail]
        todo.extend(targets(tail))

    inbound = parents(reachable)
    ordered = [label for label in blocks if label in reachable]
    following = dict(zip(ordered, ordered[1:]))
    next_block = {}
    merged = set()
    for label, body in reachable.items():
        op, args = body[-1]
        if op == "jmp":
            dest = args[0]
            if dest == following.get(label) and len(inbound[dest]) == 1:
                next_block[label] = dest
                merged.add(dest)

    result = {}
    for label in blocks:
        if label not in reachable or label in merged:
            continue
        current = label
        combined = []
        while current in next_block:
            combined.extend(reachable[current][:-1])
            current = next_block[current]
        combined.extend(reachable[current])
        result[label] = combined
    return result


def live_out(blocks):
    uses = {}
    defs = {}
    for label, body in blocks.items():
        used, defined = set(), set()
        for op, args in body:
            used.update(reg for reg in reads(op, args) if reg not in defined)
            if writes(op):
                defined.add(args[0])
        uses[label], defs[label] = used, defined

    inbound = parents(blocks)
    outgoing = {label: set() for label in blocks}
    incoming = {label: set() for label in blocks}
    queue = deque(reversed(blocks))
    pending = set(blocks)
    budget = 16 * sum(map(len, blocks.values())) + 4096
    total = 0
    while queue:
        label = queue.popleft()
        pending.remove(label)
        successors = targets(blocks[label][-1])
        live = set().union(*(incoming[dest] for dest in successors))
        budget -= len(live) + len(uses[label]) + 1
        if budget < 0:
            return None
        updated = uses[label] | (live - defs[label])
        outgoing[label] = live
        if updated != incoming[label]:
            total += len(updated) - len(incoming[label])
            if total > 1000000:
                return None
            incoming[label] = updated
            for parent in inbound[label]:
                if parent not in pending:
                    pending.add(parent)
                    queue.append(parent)
    return outgoing
