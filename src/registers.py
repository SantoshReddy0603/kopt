"""Global register renaming for machines with fewer registers than names.

The timing model only charges reload penalties when the program names more
registers than the machine has. Two registers can share a name when neither is
written while the other still holds a value that is read later. Parameters keep
their names, and every register that may be read before it is written on some
path is live at entry together with the parameters, so its implicit zero is
kept intact.
"""
from .flow import live_out
from .ir import reads, targets, writes

WORK_LIMIT = 400_000


def _interference(blocks, params, live):
    order = {}
    for param in params:
        order.setdefault(param, len(order))
    for body in blocks.values():
        for op, args in body:
            for arg in args:
                if isinstance(arg, int) and arg < 0:
                    order.setdefault(arg, len(order))
    edges = {register: set() for register in order}
    moves = {register: [] for register in order}
    work = 0
    entry = next(iter(blocks))
    for label, body in blocks.items():
        current = set(live[label])
        for op, args in reversed(body):
            if writes(op):
                destination = args[0]
                source = args[1] if op == "mov" and isinstance(args[1], int) and args[1] < 0 else None
                work += len(current)
                if work > WORK_LIMIT:
                    return None
                for register in current:
                    if register != destination and register != source:
                        edges[destination].add(register)
                        edges[register].add(destination)
                if source is not None and source != destination:
                    moves[destination].append(source)
                    moves[source].append(destination)
                current.discard(destination)
            current.update(reads(op, args))
        if label == entry:
            start = sorted(current | set(params), key=order.get)
            work += len(start) ** 2
            if work > WORK_LIMIT:
                return None
            for index, first in enumerate(start):
                for second in start[index + 1:]:
                    edges[first].add(second)
                    edges[second].add(first)
    return order, edges, moves


def _elimination(order, edges, fixed):
    """Smallest-last order of the registers that are not parameters."""
    degree = {register: len(edges[register]) for register in order if register not in fixed}
    buckets = {}
    for register in sorted(degree, key=order.get):
        buckets.setdefault(degree[register], {})[register] = None
    removed = set()
    stack = []
    while len(stack) < len(degree):
        low = min(key for key, bucket in buckets.items() if bucket)
        register = next(iter(buckets[low]))
        del buckets[low][register]
        removed.add(register)
        stack.append(register)
        for neighbour in edges[register]:
            if neighbour in degree and neighbour not in removed:
                old = degree[neighbour]
                del buckets[old][neighbour]
                degree[neighbour] = old - 1
                buckets.setdefault(old - 1, {})[neighbour] = None
    return reversed(stack)


def color(blocks, params, machine):
    names = set(params)
    for body in blocks.values():
        for op, args in body:
            names.update(arg for arg in args if isinstance(arg, int) and arg < 0)
    if len(names) <= machine["registers"] or not blocks:
        return blocks
    live = live_out(blocks)
    if live is None:
        return blocks
    graph = _interference(blocks, params, live)
    if graph is None:
        return blocks
    order, edges, moves = graph
    colors = {}
    for index, param in enumerate(dict.fromkeys(params)):
        colors[param] = index
    count = len(colors)
    for register in _elimination(order, edges, colors):
        taken = {colors[other] for other in edges[register] if other in colors}
        preferred = [colors[other] for other in moves[register]
                     if other in colors and colors[other] not in taken]
        if preferred:
            colors[register] = min(preferred)
            continue
        choice = 0
        while choice in taken:
            choice += 1
        colors[register] = choice
        count = max(count, choice + 1)
    if count >= len(names):
        return blocks
    names_by_color = {}
    for register in sorted(colors, key=order.get):
        names_by_color.setdefault(colors[register], register)
    rename = {register: names_by_color[colors[register]] for register in colors}
    result = {}
    for label, body in blocks.items():
        renamed = []
        for op, args in body:
            args = tuple(rename.get(arg, arg) if isinstance(arg, int) and arg < 0 else arg for arg in args)
            if op == "mov" and args[0] == args[1]:
                continue
            renamed.append((op, args))
        if not renamed or targets(renamed[-1]) != targets(body[-1]):
            return blocks
        result[label] = renamed
    return result
