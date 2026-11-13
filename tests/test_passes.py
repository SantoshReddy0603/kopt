"""Direct checks of the specializing passes against the reference interpreter."""
from itertools import product
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from _reference import Interpreter, difference
from src import affine, backend, bits, divide, finite, modular, paths, popcount, ranges
from src.selects import simplify_selects
from src.ir import from_program, render
from src.kirlib import SCRATCH_BASE, parse_program
from src.loops import optimize_loops, rotate
from src.rewrite import prune
from tests.test_optimizer import MACHINE, Inputs

EDGES = (0, 1, 2, 3, 7, 100, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFE, 0xFFFFFFFF)


def parameters(program):
    return tuple(~param for param in program.params)


class PassTests(unittest.TestCase):
    def same(self, source, blocks, inputs, steps=20000, skip_long=False):
        original = parse_program(source)
        rendered = parse_program(render(original.name, parameters(original), blocks))
        left, right = Interpreter(original), Interpreter(rendered)
        for inp in inputs:
            before = left.execute(inp, steps)
            if skip_long and before.status == "TIMEOUT":
                continue
            self.assertNotEqual(before.status, "TIMEOUT")
            after = right.execute(inp, 20 * steps)
            if before.memory[SCRATCH_BASE:] != after.memory[SCRATCH_BASE:]:
                before.memory[SCRATCH_BASE:] = after.memory[SCRATCH_BASE:]
            self.assertIsNone(difference(before, after), inp)

    def test_paths_with_memory_traps_and_volatile_order(self):
        source = """func walk(v0, v1)
entry:
  const v2, 0
  load v3, 5
  store 6, v3
  store 3100, v0
  vstore 1, v0
  eq v4, v1, 0
  br v4, zero, other
zero:
  load v5, 3100
  add v6, v5, v3
  store 6, v6
  ret v6
other:
  div v7, v0, v1
  store 7, v7
  lt v8, v7, 4
  br v8, small, big
small:
  load v9, 4096
  ret v9
big:
  load v10, 6
  vstore 2, v10
  ret v7
"""
        program = parse_program(source)
        blocks = paths.specialize(program, parameters(program))
        self.assertIsNotNone(blocks)
        text = render(program.name, parameters(program), blocks)
        self.assertNotIn("3100", text)
        inputs = [Inputs([a, b], [(0, [9, 8, 7, 6, 5, 4, 3, 2])]) for a, b in product(EDGES, EDGES)]
        self.same(source, blocks, inputs)

    def test_paths_rejects_unknown_addresses(self):
        source = """func indexed(v0)
entry:
  load v1, v0
  ret v1
"""
        program = parse_program(source)
        self.assertIsNone(paths.specialize(program, parameters(program)))

    def test_finite_polynomial_table(self):
        source = """func poly(v0, v1)
entry:
  and v2, v0, 7
  const v3, 0
  const v4, 0
  jmp head
head:
  lt v5, v4, v2
  br v5, body, done
body:
  mul v6, v4, v4
  mul v6, v6, v1
  add v3, v3, v6
  add v3, v3, v1
  add v4, v4, 1
  jmp head
done:
  ret v3
"""
        program = parse_program(source)
        blocks = from_program(program)
        found = list(finite.specialize(program, blocks, parameters(program)))
        self.assertTrue(found)
        inputs = [Inputs([a, b]) for a, b in product(range(16), EDGES)]
        for candidate in found:
            self.same(source, candidate, inputs)

    def test_bit_vector_popcount(self):
        source = """func ones(v0)
entry:
  const v1, 0
  const v2, 0
  jmp head
head:
  lt v3, v2, 32
  br v3, body, done
body:
  shr v4, v0, v2
  and v4, v4, 1
  add v1, v1, v4
  add v2, v2, 1
  jmp head
done:
  ret v1
"""
        program = parse_program(source)
        blocks = bits.specialize(program, parameters(program))
        self.assertIsNotNone(blocks)
        rng = random.Random(5)
        inputs = [Inputs([value]) for value in EDGES + tuple(rng.getrandbits(32) for _ in range(64))]
        self.same(source, blocks, inputs)

    def test_ranges_remove_only_proven_dead_loads(self):
        source = """func loads(v0)
entry:
  and v1, v0, 4095
  load v2, v1
  and v3, v0, 8191
  load v4, v3
  ret 0
"""
        program = parse_program(source)
        blocks = from_program(program)
        self.assertEqual(ranges.safe_loads(blocks), {"entry": {1}})
        pruned = prune(blocks, parameters(program))
        loads = [op for body in pruned.values() for op, _ in body if op == "load"]
        self.assertEqual(loads, ["load"])
        self.same(source, pruned, [Inputs([value]) for value in EDGES + (4095, 4096, 8191)])

    def test_grouped_loop_steps_and_limits(self):
        template = """func strided(v0, v1, v2)
entry:
  mov v3, v0
  const v5, 0
  jmp head
head:
  lt v4, v3, v1
  br v4, body, done
body:
  add v6, v3, 512
  and v6, v6, 1023
  store v6, v3
  add v5, v5, 1
  add v3, v3, {step}
  jmp head
done:
  ret v5
"""
        machine = dict(MACHINE, registers=16, icache_size=256)
        starts = (0, 1, 5, 0x7FFFFFF0, 0x80000000, 0xFFFFFFF0)
        limits = (0, 3, 9, 40, 0x7FFFFFFF, 0x80000002, 0xFFFFFFFF)
        for step in ("v2", "3"):
            source = template.format(step=step)
            program = parse_program(source)
            params = parameters(program)
            blocks = optimize_loops(from_program(program), params, machine, {})
            self.assertTrue(any(label.endswith("_group") for label in blocks), step)
            self.assertIsNone(optimize_loops(from_program(program), params, machine, {},
                                             frozenset({"group"})).get("body_group"))
            self.assertFalse(any(label.endswith("_group") for label in optimize_loops(
                from_program(program), params, machine, {}, frozenset({"group"}))))
            steps = (1, 2, 3, 7, 0x10000000, 0x7FFFFFFF, 0xFFFFFFFF, 0) if step == "v2" else (3,)
            inputs = [Inputs(list(values)) for values in product(starts, limits, steps)]
            self.same(source, blocks, inputs, steps=20000, skip_long=True)

    def test_deferred_remainders_only_when_sums_stay_small(self):
        template = """func checksum(v0)
entry:
  and v0, v0, {mask}
  const v1, 1
  const v2, 0
  const v3, 0
  jmp head
head:
  lt v4, v3, v0
  br v4, body, done
body:
  load v5, v3
  and v5, v5, {byte}
  add v1, v1, v5
  mod v1, v1, {modulus}
  add v2, v2, v1
  mod v2, v2, {modulus}
  {extra}
  add v3, v3, 1
  jmp head
done:
  shl v6, v2, 16
  or v6, v6, v1
  ret v6
"""
        def deferred(**fields):
            values = dict(mask=63, byte=255, modulus=65521, extra="")
            values.update(fields)
            source = template.format(**values)
            program = parse_program(source)
            rotated = rotate(from_program(program))
            changed = modular.defer(rotated)
            return source, changed, changed is not rotated

        rng = random.Random(7)
        memory = [[(0, [rng.randrange(1 << 32) for _ in range(64)])] for _ in range(4)]
        memory.append([(0, [0xFFFFFFFF] * 64)])
        for fields in ({}, {"modulus": 255}, {"modulus": 2}):
            source, blocks, changed = deferred(**fields)
            self.assertTrue(changed, fields)
            loop = blocks["body"] if "body" in blocks else next(b for b in blocks.values() if b[-1][0] == "br")
            self.assertFalse(any(op == "mod" for op, _ in loop))
            inputs = [Inputs([value], segments) for value in EDGES + (5, 62, 63) for segments in memory]
            self.same(source, blocks, inputs)
        for fields in ({"mask": 4095, "byte": 0xFFFFFF},
                       {"mask": 0xFFFF},
                       {"byte": 0xFFFFFFFF},
                       {"extra": "lt v7, v1, 9\n  store 100, v7"}):
            self.assertFalse(deferred(**fields)[2], fields)

    def test_scheduler_reorders_only_distinct_literal_addresses(self):
        block = [("load", (~0, 0)), ("store", (8, ~0)), ("load", (~1, 8)), ("store", (9, ~1)),
                 ("load", (~2, ~5)), ("store", (10, ~2)), ("ret", (0,))]
        operations = backend._operations(block[:-1])
        successors, _ = backend._dependencies(operations, block)
        self.assertIn(2, successors[1])
        self.assertNotIn(2, successors[0])
        self.assertIn(4, successors[0])
        self.assertIn(4, successors[3])
        self.assertIn(5, successors[4])

    def test_select_rules_keep_values(self):
        source = """func count(v0, v1, v2)
entry:
  and v3, v0, 255
  eq v4, v3, 0
  eq v5, v1, 0
  add v6, v2, 1
  select v7, v5, v6, v2
  select v8, v4, v2, v7
  select v9, v4, 0, v5
  eq v10, v4, 0
  eq v11, v10, 0
  select v12, v5, 0, 1
  select v13, v5, 1, v4
  add v14, v8, v9
  add v14, v14, v11
  add v14, v14, v12
  add v14, v14, v13
  ret v14
"""
        program = parse_program(source)
        blocks = simplify_selects(from_program(program), parameters(program))
        text = render(program.name, parameters(program), blocks)
        self.assertEqual(text.count("select"), 0, text)
        inputs = [Inputs([a, b, c]) for a, b, c in product(EDGES, (0, 1, 5, 0xFFFFFFFF), (0, 9, 0xFFFFFFFF))]
        self.same(source, blocks, inputs)

    def test_select_rules_need_boolean_operands(self):
        # v1 may be any value and v3 is rewritten after its test, so these
        # selects must stay.
        source = """func keep(v0, v1)
entry:
  eq v2, v0, 0
  select v3, v2, 0, v1
  eq v4, v3, 0
  add v3, v3, 2
  select v5, v4, v3, 7
  ret v5
"""
        program = parse_program(source)
        blocks = simplify_selects(from_program(program), parameters(program))
        inputs = [Inputs([a, b]) for a, b in product(EDGES, EDGES)]
        self.same(source, blocks, inputs)
        self.assertIn("select", render(program.name, parameters(program), blocks))

    def test_bit_test_lookup_trees(self):
        source = """func classify(v0, v1)
entry:
  and v2, v0, 12
  and v3, v1, 1
  eq v4, v2, 0
  br v4, low, high
low:
  br v3, one, two
one:
  ret 17
two:
  ret 3
high:
  lt v5, v2, 8
  br v5, mid, top
mid:
  add v6, v3, 40
  ret v6
top:
  ret 5
"""
        program = parse_program(source)
        blocks = from_program(program)
        params = parameters(program)
        masks = finite.prove(blocks, params)
        values = [0] * 8
        keys = finite.spread(blocks, params)
        self.assertEqual(len(keys), 8)
        self.assertEqual(len({(a & 12, b & 1) for a, b in keys}), 8)
        reference = Interpreter(program)
        order = finite._key_bits(params, masks)
        for args in keys:
            key = sum(1 << j for j, (source, bit) in enumerate(order) if args[params.index(source)] >> bit & 1)
            values[key] = reference.execute(Inputs(args), 1000).value
        inputs = [Inputs([a, b]) for a, b in product(EDGES + (4, 8, 12, 13), EDGES)]
        built = 0
        for levels, early, inner in product(range(3), (False, True), (False, True)):
            tree = finite.build_tests(params, masks, values, levels, early, inner)
            if tree is not None:
                built += 1
                self.same(source, tree, inputs)
        self.assertTrue(built)
        for candidate in finite.specialize(program, blocks, params):
            self.same(source, candidate, inputs)

    def test_bounded_renaming_keeps_register_count(self):
        source = """func sums(v0)
entry:
  const v1, 0
  const v2, 0
  jmp body
body:
  load v3, v2
  and v4, v3, 7
  add v1, v1, v4
  load v3, v2
  shr v4, v3, 3
  add v1, v1, v4
  load v3, v2
  mul v4, v3, 3
  add v1, v1, v4
  add v2, v2, 1
  lt v5, v2, v0
  br v5, body, done
done:
  ret v1
"""
        program = parse_program(source)
        params = parameters(program)
        blocks = from_program(program)
        machine = dict(MACHINE, registers=6, issue_width=4, mem_ports=2)
        before = backend._names(blocks, params)
        tuned = backend.tune(blocks, params, machine, None, rename="bounded")
        self.assertLessEqual(len(backend._names(tuned, params)), max(6, len(before)))
        inputs = [Inputs([n], [(0, list(range(5, 40)))]) for n in (0, 1, 2, 7, 20)]
        self.same(source, tuned, inputs)


    def test_clear_lowest_and_shift_out_loops(self):
        sources = ["""func ones(v0, v1)
entry:
  const v2, 0
  eq v3, v0, 0
  br v3, done, body
body:
  sub v4, v0, 1
  and v0, v0, v4
  add v2, v2, 3
  eq v5, v0, 0
  br v5, done, body
done:
  add v2, v2, v0
  ret v2
""", """func ones(v0, v1)
entry:
  mov v2, v1
  jmp body
body:
  sub v4, v0, 1
  and v0, v0, v4
  add v2, v2, 1
  br v0, body, done
done:
  ret v2
""", """func ones(v0, v1)
entry:
  const v2, 0
  jmp body
body:
  and v4, v0, 1
  add v2, v2, v4
  shr v0, v0, 1
  br v0, body, done
done:
  mul v2, v2, v1
  ret v2
"""]
        rng = random.Random(9)
        values = EDGES + tuple(rng.getrandbits(32) for _ in range(40))
        inputs = [Inputs([a, b]) for a in values for b in (0, 5)]
        for source in sources:
            program = parse_program(source)
            blocks = from_program(program)
            replaced = popcount.replace_loops(blocks, parameters(program), MACHINE)
            self.assertIsNot(replaced, blocks)
            self.assertFalse(any(label in body[-1][1] for label, body in replaced.items()
                                 if body[-1][0] == "br"))
            self.same(source, replaced, inputs)

    def test_bit_affine_rebuild(self):
        source = """func crc(v0, v1)
entry:
  xor v2, v0, 4294967295
  and v3, v2, 1
  eq v4, v3, 0
  shr v5, v2, 1
  xor v6, v5, 3988292384
  select v2, v4, v5, v6
  and v3, v2, 1
  eq v4, v3, 1
  shr v5, v2, 1
  xor v6, v5, 3988292384
  select v2, v4, v6, v5
  shl v7, v1, 3
  and v7, v7, 4080
  xor v2, v2, v7
  shr v8, v1, 30
  xor v2, v2, v8
  ret v2
"""
        program = parse_program(source)
        rebuilt = affine.synthesize(from_program(program), parameters(program))
        self.assertIsNotNone(rebuilt)
        rng = random.Random(3)
        inputs = [Inputs([a, b]) for a, b in product(EDGES, EDGES)]
        inputs += [Inputs([rng.getrandbits(32), rng.getrandbits(32)]) for _ in range(100)]
        self.same(source, rebuilt, inputs)
        carry = source.replace("xor v2, v2, v8", "add v2, v2, v8")
        program = parse_program(carry)
        self.assertIsNone(affine.synthesize(from_program(program), parameters(program)))

    def test_power_of_two_division(self):
        source = """func q(v0, v1)
entry:
  div v2, v0, 8
  mod v3, v1, 16
  div v4, v1, 1073741824
  mod v5, v0, 2
  and v6, v1, 255
  div v6, v6, 4
  add v2, v2, v3
  xor v2, v2, v4
  shl v5, v5, 7
  xor v2, v2, v5
  add v2, v2, v6
  ret v2
"""
        program = parse_program(source)
        expanded = divide.expand(from_program(program), parameters(program), MACHINE)
        self.assertFalse(any(op in ("div", "mod") for body in expanded.values() for op, _ in body))
        rng = random.Random(4)
        inputs = [Inputs([a, b]) for a, b in product(EDGES, EDGES)]
        inputs += [Inputs([rng.getrandbits(32), rng.getrandbits(32)]) for _ in range(200)]
        self.same(source, expanded, inputs)


if __name__ == "__main__":
    unittest.main()
