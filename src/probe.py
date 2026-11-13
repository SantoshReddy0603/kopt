from .kirlib import MASK, SCRATCH_BASE, compile_program, execute, time_trace


PROBE_STEPS = 3200


def inputs(nparams):
    words = ([3, 0, 5, MASK, 1, 2, 7, 4, 6] * 8)[:64]
    memory = [(0, words), (3071, [12, MASK]), (4095, [3])]
    ports = {port: [1, 9, 0, MASK] for port in range(8)}
    for value in (0, 1, 2, 7, 31, 63, 4095, 4096, 0x80000000, MASK):
        yield [value] * nparams, memory, ports
    yield list(range(nparams)), [], {}
    yield [1 << i for i in range(nparams)], memory, ports


def baseline_runs(program, machine):
    compiled = compile_program(program)
    finished = []
    for args, memory, ports in inputs(len(program.params)):
        result = execute(compiled, args, memory, ports, PROBE_STEPS, want_trace=True)
        if result.status != "TIMEOUT":
            cycles = time_trace(compiled, result.trace, machine)
            finished.append((args, memory, ports, result, cycles))
    return finished


def candidate_cycles(program, machine, baseline, veto=True, work=None):
    compiled = compile_program(program)
    total = 0
    for args, memory, ports, before, cycles in baseline:
        limit = min(before.steps * 20 + 1000, 2 * PROBE_STEPS)
        after = execute(compiled, args, memory, ports, limit, want_trace=True)
        if work is not None:
            work[0] += after.steps
        if (after.status != before.status or after.value != before.value
                or after.vlog != before.vlog or after.mem[:SCRATCH_BASE] != before.mem[:SCRATCH_BASE]):
            return None
        measured = time_trace(compiled, after.trace, machine)
        # A run that traps may pay one taken branch more: moving the trap
        # block out of the fall-through path is what speeds up the others.
        allowance = machine['taken_branch_penalty'] if before.status == 'TRAP' else 0
        if veto and measured > cycles + allowance:
            return None
        total += measured
    return total
