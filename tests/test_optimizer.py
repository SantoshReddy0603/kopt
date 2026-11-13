from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from _reference import Interpreter, difference
from src.cli import optimize_source
from src.kirlib import (check_machine_features, compile_program, execute, load_machine,
                        load_program, parse_program, time_trace)


MACHINE = {
    "issue_width": 2, "mem_ports": 1, "registers": 4,
    "reg_spill_penalty": 7, "taken_branch_penalty": 4,
    "icache_size": 16, "icache_penalty": 2, "lat_alu": 1,
    "lat_mul": 3, "lat_div": 12, "lat_load": 4, "lat_mac": 3,
    "lat_addsh": 1, "fused_mac": 0, "fused_addsh": 0,
}


@dataclass
class Inputs:
    args: list
    segments: list = field(default_factory=list)
    vin: dict = field(default_factory=dict)


class OptimizerTests(unittest.TestCase):
    def check_program(self, source, inputs, profile=None):
        original = parse_program(source)
        reference = Interpreter(original)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            program = directory / "in.kir"
            machine = directory / "machine.cfg"
            program.write_text(source, encoding="utf-8")
            machine.write_text("".join("{} = {}\n".format(key, MACHINE[key])
                                       for key in sorted(MACHINE)), encoding="utf-8")
            command = [sys.executable, "-B", str(ROOT / "kopt"), str(program),
                       "--machine", str(machine), "-o"]
            if profile is not None:
                profile_path = directory / "sample.prof"
                profile_path.write_text(profile, encoding="utf-8")
                profile_args = ["--profile", str(profile_path)]
            else:
                profile_args = []
            results = []
            for hash_seed in (1, 191):
                cwd = directory / str(hash_seed)
                cwd.mkdir()
                output = cwd / "out.kir"
                result = subprocess.run(command + [str(output)] + profile_args, cwd=cwd,
                                        env=dict(os.environ, PYTHONHASHSEED=str(hash_seed)),
                                        capture_output=True, encoding="utf-8", timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(list(cwd.iterdir()), [output])
                results.append(output.read_bytes())
            self.assertEqual(results[0], results[1])
            optimized = load_program(str(output))
            check_machine_features(optimized, load_machine(str(machine)))
            self.assertEqual(len(optimized.params), len(original.params))
            candidate = Interpreter(optimized)
        for inp in inputs:
            with self.subTest(args=inp.args):
                left = reference.execute(inp, 1000)
                self.assertNotEqual(left.status, "TIMEOUT")
                right = candidate.execute(inp, 20 * left.steps + 1000)
                self.assertIsNone(difference(left, right))

    def test_rotated_counted_loop_with_effects_and_remainder(self):
        self.check_program(
            """func counted(v0, v1)
entry:
  const v2, 3
  const v3, 0
  jmp head
head:
  lt v4, v2, 23
  br v4, body, done
body:
  div v5, v0, v1
  add v3, v3, v5
  store v2, v3
  vstore 1, v2
  add v2, v2, 1
  jmp head
done:
  load v6, 10
  add v6, v6, v2
  ret v6
""",
            [Inputs([dividend, divisor]) for dividend in (0, 7, 4294967295)
             for divisor in (0, 1, 3, 4294967295)],
        )

    def test_rotation_with_two_latches_and_unknown_start(self):
        self.check_program(
            """func latches(v0, v1)
entry:
  and v2, v0, 15
  jmp head
head:
  lt v3, v2, 12
  br v3, body, done
body:
  eq v4, v1, 0
  br v4, skip, work
work:
  div v5, 100, v1
  add v6, v6, v5
  add v2, v2, 1
  jmp head
skip:
  add v6, v6, 3
  add v2, v2, 2
  jmp head
done:
  ret v6
""",
            [Inputs([start, divisor]) for start in (0, 5, 11, 12, 15)
             for divisor in (0, 1, 7)],
        )

    def test_coloring_keeps_parameters_and_zero_registers(self):
        self.check_program(
            """func pressure(v7, v3)
entry:
  add v10, v7, 1
  add v11, v3, 2
  mul v12, v10, v11
  add v13, v12, v20
  const v14, 0
  jmp loop
loop:
  add v15, v13, v14
  xor v16, v15, v21
  add v13, v16, v7
  add v14, v14, 1
  lt v17, v14, v3
  br v17, loop, out
out:
  add v18, v13, v22
  sub v19, v18, v3
  ret v19
""",
            [Inputs([first, count]) for first in (0, 1, 4294967295) for count in (0, 1, 2, 9)],
        )

    def test_select_conversion_keeps_guarded_loads_and_loop_maximum(self):
        self.check_program(
            """func guarded(v1, v2, v3)
entry:
  const v4, 0
  const v5, 0
  lt v9, v4, v2
  br v9, head, done
head:
  add v6, v1, v4
  load v7, v6
  lt v8, v5, v7
  br v8, bigger, next
bigger:
  load v5, v6
  jmp next
next:
  add v4, v4, 1
  lt v9, v4, v2
  br v9, head, done
done:
  lt v10, v3, 4096
  br v10, read, skip
read:
  load v11, v3
  add v5, v5, v11
  jmp skip
skip:
  ret v5
""",
            [Inputs([base, count, address], [(base, [3, 9, 1, 4294967295, 9, 12, 0, 5])])
             for base in (0, 4088) for count in (0, 1, 5, 8) for address in (2, 4096, 4294967295)],
        )

    def test_select_conversion_rejects_unproven_load_address(self):
        from src.ifconvert import convert
        from src.ir import from_program
        source = """func f(v1, v2)
entry:
  br v1, read, out
read:
  load v2, v2
  jmp out
out:
  ret v2
"""
        program = parse_program(source)
        blocks = from_program(program)
        self.assertEqual(convert(blocks, [~1, ~2]), blocks)

    def test_join_and_zero_initialized_register(self):
        self.check_program(
            """func joins(v999999, v1)
entry:
  br v999999, set, skip
set:
  mov v2, v1
  jmp join
skip:
  add v1, v1, 1
  jmp join
join:
  div v3, v2, v1
  ret v3
""",
            [Inputs([condition, value]) for condition in (0, 1)
             for value in (0, 1, 2147483648, 4294967295)],
            "edge entry skip 0\nedge entry set 1000000\n",
        )

    def test_visible_effects_and_trap_order(self):
        self.check_program(
            """func effects(v0, v1, v2)
entry:
  store 3071, v2
  vload v3, v2
  vstore v2, v3
  div v4, v1, v0
  vload v5, v2
  vstore v2, v5
  load v6, v1
  ret v5
""",
            [Inputs([divisor, address, 21], [(0, [5])], {21: [67, 79]})
             for divisor in (0, 1, 4294967295) for address in (0, 4095, 4096, 4294967295)],
        )

    def test_store_forwarding_and_unknown_alias(self):
        self.check_program(
            """func aliases(v0, v1)
entry:
  store 3072, v1
  load v2, 3072
  load v3, 4095
  store v0, v2
  load v4, 4095
  sub v5, v4, v3
  ret v5
""",
            [Inputs([address, 41], [(4095, [123])]) for address in (0, 3072, 4095, 4096)],
        )

    def test_mod_shift_fusion_and_mutable_operands(self):
        self.check_program(
            """func arithmetic(v1, v0)
entry:
  mul v2, v0, 7
  add v2, v2, v1
  shl v3, v2, 33
  add v2, v2, v3
  mod v4, v2, v1
  div v1, v0, v1
  add v0, v1, v4
  ret v0
""",
            [Inputs([divisor, value]) for divisor in (0, 2, 4294967295, 4294967294)
             for value in (7, 4294967289, 2147483648)],
        )

    def test_large_tail_keeps_short_path_timing_guard(self):
        source = """func pressure(v0,v1,v2,v3,v4,v5,v6,v7)
entry:
  mov v5, v2
  vstore 4, v2
  mov v5, v5
  sub v5, v0, v4
  add v7, v0, v1
  mov v1, v6
  vstore 4, v0
  mov v0, v5
  sub v3, v6, v4
  mov v6, v0
  add v3, v7, v3
  vstore 6, v6
  ret v3
unused:
""" + "  vstore 0, 0\n" * 4096 + "  ret 0\n"
        machine = dict(MACHINE, reg_spill_penalty=5000, icache_penalty=0)
        for text in (source, source.replace("entry:\n", "entry:\n  br v7, active, unused\nactive:\n", 1)):
            original = parse_program(text)
            output = optimize_source(text.encode("utf-8"), original, machine, {})
            programs = (original, parse_program(output.decode("utf-8")))
            cycles = []
            for program in programs:
                compiled = compile_program(program)
                result = execute(compiled, list(range(1, 9)), [], {}, 100, want_trace=True)
                self.assertEqual(result.status, "RET")
                cycles.append(time_trace(compiled, result.trace, machine))
            self.assertLessEqual(cycles[1], cycles[0])

    def test_double_negation_wraps(self):
        source = """func negate_twice(v0)
entry:
  sub v1, 0, v0
  sub v2, 0, v1
  ret v2
"""
        original = parse_program(source)
        output = optimize_source(source.encode("utf-8"), original, MACHINE, {})
        program = parse_program(output.decode("utf-8"))
        self.assertEqual(len(program.instrs), 1)
        self.assertEqual(program.instrs[0].args, [("r", 0)])


class CommandLineTests(unittest.TestCase):
    SOURCE = """func f(v1)
entry:
  mul v2, v1, 8
  add v3, v2, 0
  ret v3
"""

    def run_kopt(self, profile_bytes=None, extra_env=None):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "in.kir").write_text(self.SOURCE)
            (directory / "m.cfg").write_text("".join("{} = {}\n".format(key, MACHINE[key])
                                                     for key in sorted(MACHINE)))
            command = [sys.executable, "-B", str(ROOT / "kopt"), "in.kir", "--machine", "m.cfg",
                       "-o", "out.kir"]
            if profile_bytes is not None:
                (directory / "p.prof").write_bytes(profile_bytes)
                command += ["--profile", "p.prof"]
            result = subprocess.run(command, cwd=directory, capture_output=True,
                                    env=dict(os.environ, **(extra_env or {})))
            output = directory / "out.kir"
            return result.returncode, output.read_text() if output.is_file() else None

    def test_unreadable_profile_is_ignored(self):
        code, output = self.run_kopt(b"edge entry entry \xff\xfe\x00\x81 garbage")
        self.assertEqual(code, 0)
        self.assertIsNotNone(output)
        parse_program(output)

    def test_missing_profile_file_is_ignored(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "in.kir").write_text(self.SOURCE)
            (directory / "m.cfg").write_text("".join("{} = {}\n".format(key, MACHINE[key])
                                                     for key in sorted(MACHINE)))
            result = subprocess.run([sys.executable, "-B", str(ROOT / "kopt"), "in.kir",
                                     "--machine", "m.cfg", "--profile", "nope.prof",
                                     "-o", "out.kir"], cwd=directory, capture_output=True)
            self.assertEqual(result.returncode, 0)
            self.assertTrue((directory / "out.kir").is_file())

    def test_failure_falls_back_to_valid_output(self):
        from src import cli
        saved = cli.optimize_source
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "in.kir").write_text(self.SOURCE)
            (directory / "m.cfg").write_text("".join("{} = {}\n".format(key, MACHINE[key])
                                                     for key in sorted(MACHINE)))
            argv = [str(directory / "in.kir"), "--machine", str(directory / "m.cfg"),
                    "-o", str(directory / "out.kir")]

            def broken(*args, **kwargs):
                raise RuntimeError("boom")

            try:
                cli.optimize_source = broken
                self.assertEqual(cli.main(argv), 0)
                self.assertEqual((directory / "out.kir").read_text(), self.SOURCE)
            finally:
                cli.optimize_source = saved


if __name__ == "__main__":
    unittest.main()
