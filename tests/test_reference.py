from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from _reference import Interpreter, difference
from difftest import simulator_result


def execute(ops, args=(), segments=(), vin=None, labels=None, limit=100):
    instructions = [SimpleNamespace(op=op, args=operands) for op, operands in ops]
    program = SimpleNamespace(params=list(range(len(args))), instrs=instructions,
                              labels=labels or {})
    inputs = SimpleNamespace(args=args, segments=segments, vin=vin or {})
    return Interpreter(program).execute(inputs, limit)


class ReferenceTests(unittest.TestCase):
    def test_wrap_signed_division_shifts_and_register_reads(self):
        result = execute([
            ("div", (("r", 2), ("i", 1 << 31), ("i", 4294967295))),
            ("mod", (("r", 3), ("r", 0), ("r", 1))),
            ("shr", (("r", 2), ("r", 2), ("i", 63))),
            ("shl", (("r", 4), ("i", 1), ("i", 33))),
            ("mac", (("r", 4), ("r", 2), ("r", 4), ("r", 3))),
            ("ret", (("r", 4),)),
        ], args=(4294967289, 2))
        self.assertEqual((result.status, result.value), ("RET", 1))

    def test_store_and_port_activity_precede_address_trap(self):
        result = execute([
            ("vload", (("r", 0), ("i", 1))),
            ("store", (("i", 3071), ("r", 0))),
            ("vstore", (("i", 9), ("r", 0))),
            ("store", (("i", 4095), ("i", 99))),
            ("load", (("r", 1), ("i", 4096))),
            ("ret", (("r", 1),)),
        ], vin={1: [44]})
        self.assertEqual((result.status, result.value), ("TRAP", "bad_address"))
        self.assertEqual(result.events, [("L", 1, 44), ("S", 9, 44)])
        self.assertEqual(len(result.memory), 3072)
        self.assertEqual(result.memory[-1], 44)

    def test_loop_and_explicit_trap(self):
        ops = [
            ("sub", (("r", 0), ("r", 0), ("i", 1))),
            ("br", (("r", 0), ("l", "loop"), ("l", "done"))),
            ("trap", (("i", 4294967295),)),
        ]
        labels = {"loop": 0, "done": 2}
        result = execute(ops, args=(3,), labels=labels)
        self.assertEqual((result.status, result.value, result.steps),
                         ("TRAP", "explicit(4294967295)", 7))
        self.assertEqual(execute(ops, args=(3,), labels=labels, limit=6).status, "TIMEOUT")

    def test_volatile_difference_detected_with_same_return(self):
        ops = [("vload", (("r", 0), ("i", 17))), ("ret", (("i", 0),))]
        left = execute(ops, vin={17: [1]})
        right = execute(ops, vin={17: [0]})
        self.assertIn("volatile event 0", difference(left, right))

    def test_ksim_verbose_format(self):
        result = simulator_result(
            "status: TRAP explicit(5)\nret: -\nvlog_entries: 1\n"
            "steps: 3\nvlog[0]: vstore port=8 value=5\nmem[3071] = 6\n"
        )
        self.assertEqual((result.status, result.value, result.steps),
                         ("TRAP", "explicit(5)", 3))
        self.assertEqual(result.events, [("S", 8, 5)])
        self.assertEqual(result.memory[-1], 6)


if __name__ == "__main__":
    unittest.main()
