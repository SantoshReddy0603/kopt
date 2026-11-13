from collections import Counter, OrderedDict, defaultdict
from heapq import heappop, heappush

from .ir import is_reg, reads, targets, writes
from .kirlib import LATKEY, MEM_OPS, MEM_WORDS


EFFECTS = frozenset((*MEM_OPS, 'div', 'mod', 'trap'))
BLOCK_LIMIT = 320
WORK_LIMIT = 16_384
RENAME_LIMIT = 200
RENAME_BLOCK = 160
PRESSURE_BLOCK = 40
PRESSURE_LIMIT = 96
PRESSURE_LOOPS = 40
PRESSURE_WIDTH = 16


def _accesses(instruction):
    op, operands = instruction
    destination = operands[0] if writes(op) else None
    return destination, tuple(reads(op, operands))


def _operations(block):
    result = []
    for instruction in block:
        op = instruction[0]
        destination, sources = _accesses(instruction)
        touched = tuple(dict.fromkeys(sources if destination is None else (*sources, destination)))
        result.append((op, destination, sources, touched))
    return result


class _Clock:
    def __init__(self, machine, offset, params):
        self.width = machine['issue_width']
        self.ports = machine['mem_ports']
        self.capacity = machine['registers']
        self.reload = machine['reg_spill_penalty']
        self.cache = machine['icache_size']
        self.fetch = machine['icache_penalty']
        self.latency = {op: machine[key] for op, key in LATKEY.items()}
        self.offset = offset
        self.previous = 0
        self.slots = 0
        self.memory_slots = 0
        self.ready = {}
        self.resident = OrderedDict((register, None) for register in params[-self.capacity:])
        self.seen = set(params)

    def _earliest(self, op, touched, delay):
        cycle = self.previous + delay
        for register in touched:
            cycle = max(cycle, self.ready.get(register, 0))
        if cycle == self.previous and (
            self.slots >= self.width or (op in MEM_OPS and self.memory_slots >= self.ports)
        ):
            cycle += 1
        return cycle

    def predict(self, operation):
        op, _, _, touched = operation
        delay = self.fetch if self.offset >= self.cache else 0
        if self.reload:
            delay += self.reload * sum(
                register in self.seen and register not in self.resident
                for register in touched
            )
        return self._earliest(op, touched, delay)

    def issue(self, operation):
        op, destination, _, touched = operation
        delay = self.fetch if self.offset >= self.cache else 0
        for register in touched:
            if register in self.resident:
                self.resident.move_to_end(register)
                continue
            if register in self.seen:
                delay += self.reload
            if len(self.resident) == self.capacity:
                self.resident.popitem(last=False)
            self.resident[register] = None
            self.seen.add(register)
        cycle = self._earliest(op, touched, delay)
        if cycle > self.previous:
            self.slots = 0
            self.memory_slots = 0
        self.slots += 1
        self.memory_slots += op in MEM_OPS
        self.previous = cycle
        self.offset += 1
        if destination is not None:
            self.ready[destination] = cycle + self.latency[op]
        return cycle


def _score(block, machine, offset, params, external):
    clock = _Clock(machine, offset, params)
    effects = []
    for operation in _operations(block):
        cycle = clock.issue(operation)
        if operation[0] in EFFECTS:
            effects.append(cycle)
    return (*sorted(effects), clock.previous, *(clock.ready.get(register, 0) for register in external))


def _steady(block, machine, offset, params):
    """Finish time of two back-to-back runs of a loop body.

    A single run hides latency that the next iteration waits for, such as a
    multiply on a carried register.
    """
    clock = _Clock(machine, offset, params)
    operations = _operations(block)
    for operation in operations:
        clock.issue(operation)
    clock.previous += 1 + machine['taken_branch_penalty']
    clock.slots = clock.memory_slots = 0
    clock.offset = offset
    for operation in operations:
        clock.issue(operation)
    return (clock.previous, *sorted(clock.ready.values())[-1:])


def _improves(candidate, current):
    return candidate != current and all(new <= old for new, old in zip(candidate, current))


def _sooner(candidate, current):
    """The candidate block finishes strictly earlier (latest event, then total)."""
    return (max(candidate), sum(candidate)) < (max(current), sum(current))


def _fixed_address(instruction):
    """The literal address of a load or store that cannot trap, else None."""
    op, operands = instruction
    if op not in ("load", "store"):
        return None
    address = operands[1] if op == "load" else operands[0]
    if is_reg(address) or not 0 <= address < MEM_WORDS:
        return None
    return address


def _dependencies(operations, block):
    writers = {}
    readers = defaultdict(list)
    successors = [[] for _ in operations]
    incoming = []
    previous_effect = None
    since = []
    for index, (op, destination, sources, _) in enumerate(operations):
        dependencies = {writers[register] for register in sources if register in writers}
        if destination is not None:
            dependencies.update(readers.pop(destination, ()))
            if destination in writers:
                dependencies.add(writers[destination])
            writers[destination] = index
        for register in dict.fromkeys(sources):
            if register != destination:
                readers[register].append(index)
        address = _fixed_address(block[index])
        if address is not None:
            if previous_effect is not None:
                dependencies.add(previous_effect)
            dependencies.update(other for other, where, stored in since
                                if where == address and (stored or op == "store"))
            since.append((index, address, op == "store"))
        elif op in EFFECTS:
            if previous_effect is not None:
                dependencies.add(previous_effect)
            dependencies.update(other for other, _, _ in since)
            since = []
            previous_effect = index
        for before in sorted(dependencies):
            successors[before].append(index)
        incoming.append(len(dependencies))
    return successors, incoming


def _schedule(block, machine, offset, params, limit=None, inbound=False):
    """List-schedule the block body.

    With a limit, an instruction that would raise the number of live values
    to the limit or above waits while another instruction is ready that
    does not. With `inbound`, registers read before any write in the block
    count as live from the start.
    """
    operations = _operations(block[:-1])
    successors, incoming = _dependencies(operations, block)
    pending = defaultdict(int)
    for _, _, sources, _ in operations:
        for register in set(sources):
            pending[register] += 1
    live = set()
    if inbound:
        written = set()
        for _, destination, sources, _ in operations:
            live.update(r for r in sources if r not in written)
            if destination is not None:
                written.add(destination)
    clock = _Clock(machine, offset, params)
    critical = [0] * len(operations)
    for index in reversed(range(len(operations))):
        critical[index] = clock.latency.get(operations[index][0], 1) + max(
            (critical[after] for after in successors[index]), default=0
        )
    available = {index for index, count in enumerate(incoming) if count == 0}
    result = []
    while available:
        if limit is None:
            index = min(available, key=lambda i: (clock.predict(operations[i]), -critical[i], i))
        else:
            def growth(i):
                _, destination, sources, _ = operations[i]
                freed = sum(1 for r in set(sources) if r in live and pending[r] == 1)
                added = destination is not None and destination not in live and pending[destination] > 0
                return added - freed
            index = min(available, key=lambda i: (len(live) + growth(i) >= limit and growth(i) > 0,
                                                  clock.predict(operations[i]), -critical[i], i))
            _, destination, sources, _ = operations[index]
            for register in set(sources):
                pending[register] -= 1
                if pending[register] == 0:
                    live.discard(register)
            if destination is not None and pending[destination] > 0:
                live.add(destination)
        available.remove(index)
        clock.issue(operations[index])
        result.append(block[index])
        for after in successors[index]:
            incoming[after] -= 1
            if incoming[after] == 0:
                available.add(after)
    result.append(block[-1])
    return result


def _schedule_backward(block, machine):
    """List-schedule the block body from its end towards its start.

    Each instruction goes as late as its readers allow, so a value is
    produced about one latency before it is needed and few values are live
    at once. Among the instructions that may take a cycle, the one with the
    longest chain of work before it goes first (it cannot move much earlier).
    """
    operations = _operations(block[:-1])
    if not operations:
        return block
    successors, _ = _dependencies(operations, block)
    latency = {op: machine[key] for op, key in LATKEY.items()}
    predecessors = [[] for _ in operations]
    for index, after in enumerate(successors):
        for other in after:
            predecessors[other].append(index)
    depth = [0] * len(operations)
    for index in range(len(operations)):
        depth[index] = max((depth[before] + latency.get(operations[before][0], 1)
                            for before in predecessors[index]), default=0)
    remaining = [len(after) for after in successors]
    earliest = [0] * len(operations)     # in cycles counted back from the end
    ready = {index for index, count in enumerate(remaining) if count == 0}
    width, ports = machine['issue_width'], machine['mem_ports']
    order = []
    cycle = 0
    while ready:
        slots = memory = 0
        placed = []
        for index in sorted(ready, key=lambda i: (-depth[i], -i)):
            if earliest[index] > cycle or slots >= width:
                continue
            memory_op = operations[index][0] in MEM_OPS
            if memory_op and memory >= ports:
                continue
            slots += 1
            memory += memory_op
            placed.append(index)
        if not placed:
            cycle = min(earliest[index] for index in ready)
            continue
        # Instructions placed in the same cycle keep their relative order.
        for index in sorted(placed, reverse=True):
            ready.remove(index)
            order.append(index)
            sources = operations[index][2]
            for before in predecessors[index]:
                # A reader waits the producer's latency; other orderings
                # only need the producer somewhere earlier.
                produced = operations[before][1]
                gap = latency.get(operations[before][0], 1) if produced is not None and produced in sources else 0
                earliest[before] = max(earliest[before], cycle + gap)
                remaining[before] -= 1
                if remaining[before] == 0:
                    ready.add(before)
        cycle += 1
    if len(order) != len(operations):
        return block
    return [block[index] for index in reversed(order)] + [block[-1]]


def _copy_clock(clock):
    other = object.__new__(_Clock)
    other.__dict__.update(clock.__dict__)
    other.ready = dict(clock.ready)
    other.resident = OrderedDict(clock.resident)
    other.seen = set(clock.seen)
    return other


def _pressure_order(block, machine, offset, params, private, donors, width):
    """Order and name the block body together, for machines that reload.

    A beam search over the ready instructions keeps the `width` partial
    orders that issue their last instruction earliest under the reload
    model. A local value (one definition, not read before it, not live
    after the block) takes the name of the most recently used register
    whose value is dead and that is still resident, so the write evicts
    nothing; otherwise it keeps its own name. Donors are registers that
    arrive with a value, are never written in the block and are dead after
    their last read in it. Returns None when nothing qualifies.
    """
    body = block[:-1]
    operations = _operations(body)
    if not operations:
        return None
    successors, incoming = _dependencies(operations, block)
    definitions = Counter(destination for _, destination, _, _ in operations if destination is not None)
    early = set()
    defined = set()
    for _, destination, sources, _ in operations:
        early.update(r for r in sources if r not in defined)
        if destination is not None:
            defined.add(destination)
    local = {r for r in private if definitions[r] == 1 and r not in early}
    donors = {r for r in donors if r not in defined}
    if not local:
        return None
    left = Counter()
    for _, _, sources, _ in operations:
        left.update(set(sources))
    left.update(set(_accesses(block[-1])[1]))
    # A state is (clock, waiting counts with -1 once issued, order, names,
    # remaining reads, free names). Children are ranked by their predicted
    # issue cycle and only the kept ones are built.
    beam = [(_Clock(machine, offset, params), tuple(incoming), (), {}, left, ())]
    for _ in operations:
        previous = beam
        choices = []
        for number, (clock, waiting, order, names, remaining, dead) in enumerate(previous):
            latest = max(clock.ready.values(), default=0)
            for index, count in enumerate(waiting):
                if count:
                    continue
                op, destination, sources, _ = operations[index]
                physical = tuple(names.get(r, r) for r in sources)
                free = dead + tuple(names.get(r, r) for r in dict.fromkeys(sources)
                                    if remaining[r] == 1 and (r in local or r in donors))
                target = destination
                if destination in local:
                    resident = [r for r in clock.resident if r in free]
                    if resident:
                        target = resident[-1]
                touched = tuple(dict.fromkeys(physical if target is None else (*physical, target)))
                operation = (op, target, physical, touched)
                cycle = clock.predict(operation)
                done = max(latest, cycle + clock.latency.get(op, 1) if target is not None else 0)
                choices.append((max(cycle, clock.previous), done, number, index, operation, free))
        choices.sort(key=lambda choice: choice[:4])
        beam = []
        kept = set()
        for _, _, number, index, operation, free in choices:
            clock, waiting, order, names, remaining, _ = previous[number]
            target = operation[1]
            counts = Counter(remaining)
            counts.subtract(set(operations[index][2]))
            if target in free:
                free = tuple(r for r in free if r != target)
            after = _copy_clock(clock)
            after.issue(operation)
            key = (frozenset(i for i, _ in order) | {index}, tuple(after.resident))
            if key in kept:
                continue
            kept.add(key)
            pending = list(waiting)
            pending[index] = -1
            for later in successors[index]:
                pending[later] -= 1
            destination = operations[index][1]
            new_names = names
            if destination in local:
                new_names = dict(names)
                new_names[destination] = target
            beam.append((after, tuple(pending), order + ((index, target),), new_names, counts, free))
            if len(beam) == width:
                break
    _, _, order, names, _, _ = beam[0]

    def rename(op, operands, target):
        start = 1 if writes(op) else 0
        head = (target,) if start else ()
        return (op, head + tuple(names.get(a, a) if is_reg(a) else a for a in operands[start:]))

    result = [rename(*body[index], target) for index, target in order]
    op, operands = block[-1]
    result.append((op, tuple(names.get(a, a) if is_reg(a) else a for a in operands)))
    return result


def _schedule_pressure(block, machine, offset, params, private, donors=(), width=PRESSURE_WIDTH):
    """The best pressure order from three starting orders.

    The beam breaks ties by position, so it is run on the block as given and
    on its forward and backward list schedules.
    """
    best = None
    for seed in (block, _schedule(block, machine, offset, params), _schedule_backward(block, machine)):
        candidate = _pressure_order(seed, machine, offset, params, private, donors, width)
        if candidate is None:
            return None
        cost = _score(candidate, machine, offset, params, ())
        if best is None or cost < best[0]:
            best = (cost, candidate)
    return best[1]


def _local_intervals(block, private):
    starts = {}
    ends = {}
    blocked = set()
    for index, instruction in enumerate(block):
        destination, sources = _accesses(instruction)
        for register in sources:
            if register in private:
                if register not in starts:
                    blocked.add(register)
                ends[register] = index
        if destination in private:
            if destination in starts:
                blocked.add(destination)
            starts[destination] = index
    return [
        (start, ends.get(register, start), register)
        for register, start in starts.items()
        if register not in blocked
    ]


def _reuse(block, private, eager=False, donors=()):
    """Share names between non-overlapping local values.

    Eagerly, a value may also take the name of one read for the last time by
    the instruction that defines it (as in mac r, r, y, t). Donors are
    registers that arrive with a value, are never written and are dead after
    their last read in this block; their names become free after that read.
    """
    replacements = {}
    active = []
    free = []
    if donors:
        last = {}
        for index, instruction in enumerate(block):
            destination, sources = _accesses(instruction)
            for register in sources:
                if register in donors:
                    last[register] = index
            if destination in donors:
                return _reuse(block, private, eager)
        for register in sorted(donors):
            heappush(active, (last.get(register, -1), register))
    for start, end, register in _local_intervals(block, private):
        while active and (active[0][0] < start or eager and active[0][0] == start):
            dead_at, reusable = heappop(active)
            heappush(free, (-dead_at, reusable))
        target = heappop(free)[1] if free else register
        heappush(active, (end, target))
        if target != register:
            replacements[register] = target
    if not replacements:
        return block
    return [
        (op, tuple(replacements.get(operand, operand) for operand in operands))
        for op, operands in block
    ]


def _rename(block, fresh):
    """Give every definition except the last one of each register a new name.

    Only the final value of a register can leave the block, so the earlier
    ones are local and renaming them removes false dependencies for the
    scheduler. Returns the new block and the names it used.
    """
    last = {}
    for index, instruction in enumerate(block):
        destination, _ = _accesses(instruction)
        if destination is not None:
            last[destination] = index
    current = {}
    result = []
    names = set()
    for index, (op, operands) in enumerate(block):
        destination = operands[0] if writes(op) else None
        start = 1 if destination is not None else 0
        new = list(operands[:start]) + [current.get(a, a) if is_reg(a) else a for a in operands[start:]]
        if destination is not None:
            if last[destination] == index:
                current.pop(destination, None)
            else:
                current[destination] = fresh
                names.add(fresh)
                new[0] = fresh
                fresh -= 1
        result.append((op, tuple(new)))
    return result, names


def _successors(block):
    return tuple(dict.fromkeys(targets(block[-1])))


def _region_order(labels, edges, first, last, profile):
    links = {
        source: target for source, target in zip(labels, labels[1:])
        if target in edges[source]
    }
    tails = {}
    heads = {}
    start = labels[0]
    for index, label in enumerate(labels):
        if label not in links:
            tails[start] = label
            heads[label] = start
            if index + 1 < len(labels):
                start = labels[index + 1]
    proposed = [(source, target) for source in heads for target in edges[source] if target in tails]
    if profile:
        proposed.sort(key=lambda edge: -profile.get(edge, 0))
    added = 0
    for source, target in proposed:
        start = heads.get(source)
        end = tails.get(target)
        if start is None or end is None or start == target or source == last or target == first:
            continue
        if start == first and end == last and len(tails) > 2:
            continue
        links[source] = target
        added += 1
        del heads[source]
        del tails[target]
        tails[start] = end
        heads[end] = start
    order = []
    delayed = heads.get(last)
    starts = [label for label in labels if label in tails and label != delayed]
    if delayed is not None:
        starts.append(delayed)
    for label in starts:
        while label is not None:
            order.append(label)
            label = links.get(label)
    return order, added


def _regions(labels, blocks, machine):
    if not machine['icache_penalty']:
        return [labels]
    limit = machine['icache_size']
    offset = 0
    for index, label in enumerate(labels):
        end = offset + len(blocks[label])
        if offset < limit < end:
            return [labels[:index], [label], labels[index + 1:]]
        if end == limit:
            return [labels[:index + 1], labels[index + 1:]]
        offset = end
    return [labels]


def _profile_weight(order, edges, profile):
    return sum(
        profile.get((source, target), 0)
        for source, target in zip(order, order[1:])
        if target in edges[source]
    )


def _cold_last(blocks):
    """Move blocks that end in `trap` and only follow a branch to the end.

    A trap ends the program, so its branch is taken at most once, while the
    other side of the branch may run many times and can then fall through.
    """
    labels = list(blocks)
    incoming = defaultdict(list)
    for label in labels:
        for target in _successors(blocks[label]):
            incoming[target].append(label)
    cold = []
    for index in range(1, len(labels)):
        label, before = labels[index], labels[index - 1]
        body = blocks[label]
        if body[-1][0] != "trap" or incoming[label] != [before] or blocks[before][-1][0] != "br":
            continue
        cold.append(label)
    if not cold:
        return blocks
    order = [label for label in labels if label not in cold] + cold
    return {label: blocks[label] for label in order}


def _layout(blocks, machine, profile):
    fits = sum(map(len, blocks.values())) <= machine['icache_size']
    if machine['taken_branch_penalty'] and (fits or not machine['icache_penalty']):
        blocks = _cold_last(blocks)
    labels = list(blocks)
    if len(labels) < 3:
        return blocks
    edges = {label: _successors(block) for label, block in blocks.items()}
    order = []
    for region in _regions(labels, blocks, machine):
        if len(region) < 2:
            order.extend(region)
            continue
        start = len(order)
        stop = start + len(region)
        fixed_first = region[0] if start == 0 or region[0] in edges[labels[start - 1]] else None
        fixed_last = region[-1] if stop < len(labels) and labels[stop] in edges[region[-1]] else None
        unweighted, added = _region_order(region, edges, fixed_first, fixed_last, {})
        if profile and added:
            weighted, hinted_added = _region_order(region, edges, fixed_first, fixed_last, profile)
            hotter = hinted_added == added and (
                _profile_weight(weighted, edges, profile) > _profile_weight(unweighted, edges, profile)
            )
            if hinted_added > added or hotter:
                unweighted, added = weighted, hinted_added
        order.extend(unweighted if added else region)
    return {label: blocks[label] for label in order}


def _owners(blocks, params):
    owners = dict.fromkeys(params)
    for label, block in blocks.items():
        for instruction in block:
            destination, sources = _accesses(instruction)
            registers = sources if destination is None else (*sources, destination)
            for register in registers:
                if owners.setdefault(register, label) != label:
                    owners[register] = None
    return owners


def _names(blocks, params):
    return set(params) | {r for body in blocks.values() for _, operands in body for r in operands if is_reg(r)}


def tune_pair(blocks, params, machine, profile, layout=True):
    """(guarded, spread): tuned blocks with and without the register-count guard.

    Register spills are only modelled once the program names more registers
    than the machine has, so the guarded result never lets renaming cross
    that line. The spread one may, which sometimes still pays off.
    """
    spread = tune(blocks, params, machine, profile, layout=layout, rename=None)
    if len(_names(spread, params)) > max(len(_names(blocks, params)), machine['registers']):
        return tune(blocks, params, machine, profile, layout=layout, rename="bounded"), spread
    return spread, spread


def tune(blocks, params, machine, profile, *, schedule=True, reuse=True, layout=True, rename=True, loose=False):
    if rename is True and schedule and reuse:
        return tune_pair(blocks, params, machine, profile, layout)[0]
    if not (schedule or reuse) or not any(3 <= len(block) <= BLOCK_LIMIT for block in blocks.values()):
        return _layout(blocks, machine, profile) if layout else blocks
    result = {}
    offset = 0
    budget = WORK_LIMIT
    renaming = RENAME_LIMIT if sum(map(len, blocks.values())) <= RENAME_LIMIT else 0
    # The beam search for reload-aware orders is the costliest schedule, so
    # it is kept to a few blocks: loop bodies, and in the loose placement
    # any block.
    pressure = PRESSURE_LIMIT if loose else PRESSURE_LOOPS
    owners = _owners(blocks, params)
    fresh = min((register for register in owners if register is not None), default=0) - 1
    # In "bounded" mode a renamed block is kept only while the whole program
    # still names no more registers than max(K, its original count), so the
    # machine's spill tracking is never switched on by renaming.
    bounded = rename == "bounded"
    users = Counter()
    for body in blocks.values():
        users.update({r for _, operands in body for r in operands if is_reg(r)})
    allowed = max(machine['registers'], len(set(users) | set(params)))
    # Parameters read only in the entry block, never written and never
    # needed again (nothing jumps back to the entry) can hand their names on.
    entry = next(iter(blocks), None)
    solo = set()
    if entry is not None and not any(entry in targets(body[-1]) for body in blocks.values() if body):
        solo = {p for p in params if owners.get(p, entry) in (None, entry)}
        for label, body in blocks.items():
            for instruction in body:
                destination, sources = _accesses(instruction)
                solo.discard(destination)
                if label != entry:
                    solo.difference_update(sources)
    for label, block in blocks.items():
        selected = block
        donors = solo if label == entry else ()
        if 3 <= len(block) <= min(BLOCK_LIMIT, budget):
            budget -= len(block)
            written = {destination for _, destination, _, _ in _operations(block) if destination is not None}
            private = {register for register in written if owners[register] == label}
            external = tuple(sorted(written - private))
            initial = params if offset == 0 else ()
            cost = _score(block, machine, offset, initial, external)
            if schedule:
                candidate = _schedule(block, machine, offset, initial)
                candidate_cost = _score(candidate, machine, offset, initial, external)
                if _improves(candidate_cost, cost):
                    selected, cost = candidate, candidate_cost
            if reuse:
                candidate = _reuse(selected, private, loose, donors)
                candidate_cost = _score(candidate, machine, offset, initial, external)
                if candidate is not selected and _improves(candidate_cost, cost):
                    selected, cost = candidate, candidate_cost
            # Renamed schedules cost several extra passes, so they are tried
            # on loop bodies and, up to a small total, on other blocks.
            looped = block[-1][0] in ("br", "jmp") and label in block[-1][1]
            room = looped or len(block) <= renaming
            if rename is not False and schedule and reuse and room and 6 <= len(block) <= RENAME_BLOCK and ~fresh + len(block) <= 999999:
                if not looped:
                    renaming -= len(block)
                renamed, names = _rename(block, fresh)
                capacity = machine['registers']
                named = {r for _, operands in renamed for r in operands if is_reg(r)}
                if names or len(named) > capacity or loose and pressure:
                    chosen = None
                    limits = [None]
                    depth = 7 if bounded else 4
                    if len(named) > capacity and budget >= depth * len(block):
                        budget -= depth * len(block)
                        limits += [limit for limit in range(capacity, capacity - depth, -1) if limit >= 2]
                    mine = {r for _, operands in block for r in operands if is_reg(r)}
                    base = len({r for r, n in users.items() if n > 0} | set(params)) - sum(
                        1 for r in mine if users[r] == 1 and r not in params)

                    def fits(instructions, room):
                        new = {r for _, operands in instructions for r in operands if is_reg(r)}
                        extra = sum(1 for r in new if users[r] - (r in mine) <= 0 and r not in params)
                        return base + extra <= room
                    def measure(instructions):
                        if looped:
                            return _steady(instructions, machine, offset, initial)
                        return _score(instructions, machine, offset, initial, external)
                    current = measure(selected)
                    orders = limits + ["backward"]
                    if len(named) > capacity and machine['reg_spill_penalty'] and (looped or loose) and len(block) <= min(PRESSURE_BLOCK, pressure):
                        pressure -= len(block)
                        orders.append("pressure")
                    for limit in orders:
                        if limit == "pressure":
                            ordered = _schedule_pressure(renamed, machine, offset, initial, private | names, donors)
                            if ordered is None:
                                continue
                        elif limit == "backward":
                            ordered = _schedule_backward(renamed, machine)
                        else:
                            ordered = _schedule(renamed, machine, offset, initial, limit, loose)
                        tight = None
                        for eager in ((False, True) if limit == "backward" or loose else (False,)):
                            # The pressure order already chose its names.
                            candidate = ordered if limit == "pressure" else _reuse(ordered, private | names, eager, donors)
                            if candidate == tight:
                                continue
                            tight = candidate
                            if bounded and not fits(candidate, allowed):
                                continue
                            # The backward order is judged on a block alone,
                            # so it may never make the program name more
                            # registers than max(K, its original count).
                            if limit in ("backward", "pressure") and not loose and not fits(candidate, allowed):
                                continue
                            candidate_cost = measure(candidate)
                            if _sooner(candidate_cost, current):
                                selected, current, chosen = candidate, candidate_cost, True
                    if chosen:
                        fresh -= len(names)
                        users.subtract(mine)
                        users.update({r for _, operands in selected for r in operands if is_reg(r)})
        result[label] = selected
        offset += len(block)
    return _layout(result, machine, profile) if layout else result
