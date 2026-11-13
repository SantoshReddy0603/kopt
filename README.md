# kopt

kopt optimizes Kiln IR for the COMP10 Oxygen Debt challenge. It reads a KIR program and a machine configuration, then writes a KIR program with the same observable behavior. Python 3.8 or newer and its standard library are sufficient.

## Build and run

~~~sh
./build.sh
./kopt input.kir --machine machine.cfg -o output.kir
./kopt input.kir --machine machine.cfg --profile branches.prof -o output.kir
~~~

The executable can be invoked by its absolute path from any working directory. On Windows, invoke it with Python using the same arguments. Diagnostics go to stderr. A successful run writes only the path supplied to `-o`.

## Check the output

Set `KIT` to the absolute path of the organizer's `Compilers/COMP10` folder in a separate checkout. The following example uses its public input and machine.

~~~sh
./kopt "$KIT/bench/sum_sq.kir" --machine "$KIT/machines/machine_default.cfg" -o /tmp/kopt-sum.kir
python3 -B "$KIT/kvalidate" /tmp/kopt-sum.kir --against "$KIT/bench/sum_sq.kir" --machine "$KIT/machines/machine_default.cfg"
python3 -B "$KIT/ksim" /tmp/kopt-sum.kir --input "$KIT/bench/sum_sq.in" --machine "$KIT/machines/machine_default.cfg"
~~~

## Reproduce the tests

~~~sh
python3 -B -m unittest discover -s tests
python3 -B tools/difftest.py --kit "$KIT" --cross-check-only
python3 -B tools/evaluate.py --kit "$KIT"
python3 -B tools/difftest.py --kit "$KIT" --inputs 200
tools/determinism.sh --kit "$KIT"
python3 -B tools/stress.py --kit "$KIT"
python3 -B tools/fuzz.py --kit "$KIT" --cases 2000 --jobs 4
~~~

The evaluator reports cycles, validation results and runtime per case. The differential tester compares return or trap, volatile events and memory against an independent interpreter. The determinism check varies the Python hash seed. The fuzzer runs mutated and generated programs on random machines with malformed profiles. `--match` and `--machine` narrow a run.

## Public results

All 100 public programs on each of the five public machines, checked with the official `kvalidate` and timed with `ksim`. The ratio is the geometric mean of output cycles over original cycles; smaller is better.

| Machine | Output / original cycles | Faster | Slower | Validation failures |
|---|---:|---:|---:|---:|
| Default | 0.126269 | 99/100 | 0 | 0 |
| Few registers | 0.093430 | 98/100 | 0 | 0 |
| No fused instructions | 0.140811 | 99/100 | 0 | 0 |
| Slow memory | 0.145666 | 99/100 | 0 | 0 |
| Wide and fast | 0.186580 | 99/100 | 0 | 0 |
| **All configurations** | **0.135185** | **494/500** | **0** | **0** |

The independent interpreter found zero differences in 134,765 executions, and all 500 repeat runs produced byte-identical output. The slowest public program takes about 2.5 seconds; generated 100,000- and 500,000-instruction programs take 3.0 and 13.7 seconds. The fuzzer found no failures in 2,000 cases.

## Layout

| Path | Contents |
|---|---|
| `kopt`, `build.sh` | Entry point and build check |
| `src/` | Parser, passes, scheduler and candidate search |
| `tests/` | Unit tests |
| `tools/` | Evaluation, differential testing, determinism, stress and fuzz scripts |
| `DESIGN.md` | Design write-up: passes, safety conditions, machine use, profiles, known problems |
| `REPORT.md` | Results, verification and per-program cycle counts |
