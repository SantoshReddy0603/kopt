# kopt: Report

kopt is an optimizer for Kiln IR (COMP10, Oxygen Debt). It reads a KIR program, a machine configuration and an optional edge profile, and writes a program with the same observable behaviour that runs in fewer simulated cycles on that machine. It is written in Python 3.8+ with the standard library only.

## Results

All 100 public programs were run on each of the five public machines with their supplied inputs and profiles. Each output was checked with the official `kvalidate` and timed with the official `ksim`. The ratio is output cycles divided by original cycles (geometric mean); smaller is better.

| Machine | Output / original cycles | Faster | Slower | Validation failures |
|---|---:|---:|---:|---:|
| Default | 0.126269 | 99/100 | 0 | 0 |
| Few registers | 0.093430 | 98/100 | 0 | 0 |
| No fused instructions | 0.140811 | 99/100 | 0 | 0 |
| Slow memory | 0.145666 | 99/100 | 0 | 0 |
| Wide and fast | 0.186580 | 99/100 | 0 | 0 |
| **All configurations** | **0.135185** | **494/500** | **0** | **0** |

On average the output needs about 13.5% of the original cycles (a 7.4× speed-up). No output is slower than its input. The six cases with unchanged cycles are `div_trap` on every machine (its trapping division must stay where it is) and `poly_mac` on the few-registers machine.

## Verification

| Check | Result |
|---|---|
| Official validator, 500 program and machine pairs | 0 failures |
| Independent interpreter, 200 extra inputs per pair (edge, random, trapping) | 0 differences in 134,765 executions |
| Independent interpreter against `ksim` on the supplied samples | 100/100 agree |
| Repeat runs with different Python hash seeds | 500/500 byte-identical outputs |
| Fuzzing: mutated and generated programs, random machines, malformed and wrong profiles | 2,000/2,000 cases correct, none slower |
| Generated 100,000-instruction program | 3.0 s |
| Generated 500,000-instruction program | 13.7 s |
| Slowest public program (WSL 2, Ubuntu) | about 2.5 s; the search stops at a 3 s budget |
| Unit tests | 34 passing |

## How it works

1. **Parse and normalise.** The program becomes a dictionary of blocks with explicit edges. Registers and literals have disjoint encodings, so every pass can tell them apart without lookups.
2. **Machine-independent rewriting.** Jump threading, unreachable-block removal and merging. Constant and copy propagation, algebraic simplification, value numbering and memory forwarding. Strength reduction and fusion into `mac` and `addsh`. Collapsing of linear add and multiply chains, common expressions across edges, and liveness-based dead-code removal.
3. **Control flow and loops.** Small branch arms become `select`, and 0/1 selects become plain `lt`/`eq`/`and`/`or`/`add`. Loops are rotated into self loops. Invariant code is hoisted and induction variables are strength-reduced. Counted loops are unrolled within the cache estimate, or run four iterations under one guarded test. A remainder inside a counted loop is taken once at the exit. Bit-counting loops become a branch-free population count.
4. **Arithmetic.** Serial multiply-add (Horner) chains are split into 2 or 4 independent chains when registers allow. Signed division and remainder by powers of two become shifts and masks when division is slow.
5. **Specialisation.** These rewrites are exact for every input, and each is kept only if it is faster:
   - small input domains become lookup tables or bit-test trees;
   - polynomial results get closed forms;
   - bit permutations and population counts become bit-vector closed forms;
   - straight-line CRC and parity code is rebuilt as a balanced xor tree over input bits;
   - small programs are replaced by a symbolic-execution decision tree.
6. **Machine-specific placement.** Each block is list-scheduled forward and backward, and block-local renaming removes false dependencies. On machines with an LRU reload penalty, a beam search orders and names register-heavy blocks. Global interference-graph colouring is used when the program names more registers than the machine has. Block layout keeps hot fall-through edges and moves trap blocks off the hot path.
7. **Self-checking selection.** Several pipeline variants are built, each with different placements. Every candidate is parsed again and checked against the machine's feature set. It is then run on bounded synthetic inputs through the real timing model and compared with the original on the full visible state. A candidate is accepted only if it is faster overall and not slower on any run that finishes within the probe budget. If nothing qualifies, the original is written unchanged. A time budget writes the best accepted result so far.

## Correctness rules

The observable behaviour is the return value or trap kind and code, the ordered volatile log, and memory words 0–3071 at termination.

- A divide with a possibly-zero divisor and a first load from a possibly-invalid address stay in place.
- Scheduling never reorders memory operations, volatile events or trapping instructions. The only exception is two ordinary accesses at different literal addresses below 4096, which can neither alias nor trap.
- Profiles only guide layout and the order in which loops are considered, so a wrong profile can cost speed but never correctness.
- Output does not depend on hash order, register names or program fingerprints, so repeat runs are byte-identical.
- Every pass has a work limit, so very large programs finish well inside the time limit.

`DESIGN.md` has the safety condition and work bound for each pass.

## Known limitations

- Hoisting, unrolling and induction rewrites need a single-block loop, possibly after rotation or `select` conversion.
- Loops over data-dependent memory, such as sorting, sieves and hashing input arrays, keep their loop structure.
- Register renaming minimises the number of names but does not choose which values to spill when the program still needs more registers than the machine has.
- Candidate selection uses bounded synthetic inputs, so cycle counts on other inputs can differ when control flow or register residency differs.

## Per-program cycles

Original → output cycles for every public program on each machine.

| Program | Default | Few registers | No fused | Slow memory | Wide and fast |
|---|---:|---:|---:|---:|---:|
| `addsh_index` | 746 → 124 | 1157 → 160 | 746 → 145 | 859 → 252 | 201 → 94 |
| `adler32` | 1659 → 120 | 2408 → 351 | 1915 → 168 | 1697 → 231 | 621 → 69 |
| `alias_hazard` | 22 → 13 | 48 → 13 | 22 → 13 | 40 → 29 | 12 → 8 |
| `alias_stores` | 1454 → 225 | 3134 → 514 | 1454 → 225 | 1644 → 481 | 1311 → 175 |
| `array_max` | 159 → 82 | 579 → 137 | 159 → 90 | 271 → 127 | 171 → 64 |
| `array_min` | 148 → 74 | 532 → 124 | 148 → 82 | 260 → 111 | 153 → 56 |
| `array_sum` | 108 → 63 | 499 → 80 | 108 → 63 | 188 → 97 | 109 → 45 |
| `binary_search` | 212 → 60 | 368 → 120 | 212 → 60 | 227 → 100 | 98 → 59 |
| `bit_reverse` | 1665 → 15 | 2857 → 15 | 1665 → 15 | 1665 → 15 | 392 → 15 |
| `bounds_trap` | 2875 → 712 | 4847 → 776 | 2875 → 712 | 3324 → 1224 | 1158 → 650 |
| `branchy_select` | 1767 → 206 | 2747 → 845 | 1767 → 265 | 1936 → 398 | 623 → 127 |
| `bubble_sort` | 3198 → 833 | 6304 → 1799 | 3198 → 833 | 3860 → 1377 | 1121 → 777 |
| `checked_add_trap` | 7 → 3 | 7 → 3 | 7 → 3 | 7 → 3 | 9 → 3 |
| `collatz` | 8033 → 20 | 15245 → 20 | 8033 → 27 | 8035 → 20 | 3154 → 24 |
| `common_subexpr` | 43 → 8 | 155 → 8 | 43 → 9 | 57 → 12 | 22 → 6 |
| `const_chain` | 39 → 1 | 74 → 1 | 43 → 1 | 45 → 1 | 32 → 1 |
| `constant_branch` | 959 → 7 | 1658 → 7 | 959 → 7 | 959 → 9 | 330 → 6 |
| `conv_mac` | 1363 → 288 | 2640 → 655 | 1363 → 294 | 1706 → 454 | 1264 → 206 |
| `count_above` | 394 → 104 | 1025 → 334 | 394 → 129 | 474 → 156 | 213 → 51 |
| `count_divisors` | 2776 → 13 | 5582 → 15 | 3264 → 23 | 2776 → 13 | 1715 → 14 |
| `crc_bitwise` | 1675 → 290 | 4700 → 426 | 1676 → 418 | 1739 → 362 | 1286 → 288 |
| `dead_div_trap` | 15 → 2 | 22 → 2 | 23 → 2 | 15 → 2 | 15 → 2 |
| `dead_load_trap` | 12 → 2 | 12 → 2 | 12 → 3 | 24 → 2 | 8 → 2 |
| `dead_mod_in_loop` | 258 → 13 | 463 → 13 | 306 → 13 | 258 → 13 | 129 → 14 |
| `dead_overwritten_stores` | 1085 → 92 | 1803 → 132 | 1085 → 132 | 1095 → 100 | 253 → 57 |
| `dfa_dispatch` | 1419 → 420 | 2504 → 769 | 1651 → 420 | 1588 → 612 | 770 → 506 |
| `digit_sum` | 388 → 104 | 639 → 104 | 484 → 152 | 388 → 104 | 221 → 113 |
| `div_trap` | 14 → 14 | 14 → 14 | 22 → 22 | 14 → 14 | 14 → 14 |
| `dot_pairs` | 626 → 66 | 1039 → 140 | 626 → 98 | 687 → 132 | 170 → 38 |
| `dot_product` | 366 → 56 | 635 → 138 | 366 → 90 | 425 → 122 | 113 → 36 |
| `factorial_mod` | 429 → 12 | 1393 → 13 | 589 → 14 | 469 → 12 | 450 → 12 |
| `fib` | 981 → 15 | 2233 → 22 | 981 → 18 | 981 → 15 | 310 → 16 |
| `fletcher16` | 1023 → 92 | 1484 → 251 | 1183 → 114 | 1049 → 171 | 393 → 67 |
| `fnv_hash` | 469 → 82 | 902 → 111 | 469 → 111 | 578 → 119 | 165 → 62 |
| `gcd` | 69 → 52 | 226 → 52 | 93 → 76 | 69 → 52 | 76 → 56 |
| `gray_code_sum` | 1506 → 13 | 2975 → 17 | 1506 → 15 | 1506 → 13 | 449 → 17 |
| `guarded_div` | 433 → 17 | 699 → 17 | 497 → 26 | 433 → 17 | 200 → 19 |
| `hamming_dist` | 699 → 186 | 1734 → 326 | 699 → 186 | 756 → 250 | 371 → 167 |
| `histogram` | 1568 → 402 | 2684 → 450 | 1568 → 421 | 1990 → 882 | 459 → 266 |
| `horner` | 330 → 20 | 755 → 33 | 330 → 21 | 387 → 42 | 107 → 12 |
| `index_trap` | 11 → 2 | 11 → 2 | 11 → 2 | 21 → 2 | 7 → 2 |
| `insertion_sort` | 1818 → 493 | 3245 → 863 | 1818 → 493 | 2206 → 1013 | 794 → 478 |
| `irreducible_loop` | 81 → 1 | 81 → 1 | 81 → 1 | 81 → 1 | 88 → 1 |
| `is_prime` | 68 → 19 | 141 → 28 | 84 → 30 | 70 → 19 | 52 → 20 |
| `isqrt` | 670 → 144 | 1197 → 337 | 670 → 144 | 670 → 168 | 243 → 152 |
| `jump_chain` | 59 → 7 | 59 → 7 | 59 → 7 | 59 → 9 | 69 → 6 |
| `junk_dead` | 2685 → 14 | 5021 → 14 | 2725 → 15 | 2836 → 12 | 2480 → 11 |
| `lcm_pair` | 108 → 86 | 343 → 86 | 148 → 126 | 110 → 88 | 114 → 91 |
| `log_then_divide` | 16 → 15 | 16 → 15 | 24 → 23 | 16 → 15 | 14 → 13 |
| `loop_invariant` | 2448 → 13 | 4083 → 13 | 2448 → 16 | 2452 → 21 | 520 → 9 |
| `many_live` | 137 → 27 | 278 → 94 | 144 → 27 | 145 → 30 | 93 → 13 |
| `matmul4` | 8196 → 363 | 14121 → 1178 | 8650 → 557 | 8335 → 664 | 6951 → 264 |
| `matrix_add4` | 2002 → 57 | 3730 → 100 | 2027 → 95 | 2053 → 107 | 1679 → 29 |
| `matvec8` | 457 → 35 | 782 → 64 | 457 → 75 | 516 → 64 | 153 → 23 |
| `merge_sorted` | 1097 → 328 | 2033 → 668 | 1185 → 340 | 1236 → 576 | 517 → 321 |
| `mini_interp` | 656 → 145 | 1163 → 421 | 710 → 145 | 774 → 209 | 406 → 148 |
| `multi_exit` | 487 → 13 | 2462 → 19 | 487 → 15 | 567 → 13 | 488 → 16 |
| `nested_invariant` | 6899 → 10 | 11614 → 10 | 6899 → 11 | 6901 → 15 | 2074 → 8 |
| `paren_depth` | 1030 → 275 | 1589 → 689 | 1115 → 276 | 1143 → 403 | 403 → 294 |
| `parity_fold` | 17 → 12 | 17 → 12 | 17 → 12 | 17 → 12 | 17 → 12 |
| `partial_redundancy` | 19 → 6 | 40 → 6 | 19 → 7 | 23 → 8 | 19 → 5 |
| `poly_mac` | 15 → 10 | 36 → 36 | 15 → 13 | 19 → 16 | 10 → 7 |
| `popcount` | 262 → 14 | 2111 → 14 | 262 → 14 | 262 → 16 | 328 → 13 |
| `pow_fast` | 779 → 137 | 1243 → 137 | 885 → 201 | 783 → 157 | 464 → 131 |
| `powmod` | 634 → 116 | 958 → 156 | 736 → 172 | 638 → 134 | 452 → 106 |
| `prefix_sum` | 91 → 71 | 435 → 78 | 91 → 78 | 147 → 127 | 86 → 62 |
| `redundant_loads` | 780 → 101 | 1599 → 132 | 780 → 133 | 1037 → 137 | 214 → 61 |
| `reverse_array` | 71 → 45 | 250 → 53 | 71 → 45 | 135 → 77 | 59 → 35 |
| `rle_encode` | 601 → 201 | 1179 → 533 | 608 → 201 | 706 → 305 | 244 → 209 |
| `rotate_array` | 117 → 85 | 484 → 108 | 117 → 85 | 197 → 157 | 112 → 73 |
| `rotate_mix` | 440 → 37 | 984 → 37 | 440 → 49 | 440 → 37 | 113 → 31 |
| `row_sums` | 311 → 21 | 609 → 43 | 311 → 30 | 370 → 37 | 129 → 14 |
| `run_count` | 386 → 103 | 1018 → 290 | 386 → 123 | 533 → 207 | 216 → 71 |
| `sem_bit_mix` | 7 → 6 | 43 → 6 | 7 → 6 | 7 → 6 | 7 → 5 |
| `sem_div_mod` | 61 → 17 | 78 → 17 | 93 → 27 | 63 → 17 | 56 → 17 |
| `sem_mul_wrap` | 32 → 14 | 49 → 19 | 40 → 14 | 38 → 18 | 25 → 12 |
| `sem_select_abs` | 39 → 13 | 64 → 17 | 55 → 13 | 39 → 13 | 35 → 9 |
| `sem_shifts` | 7 → 6 | 35 → 7 | 7 → 6 | 7 → 6 | 7 → 5 |
| `sem_signed_cmp` | 22 → 4 | 42 → 4 | 22 → 6 | 24 → 4 | 12 → 4 |
| `sensor_drain` | 294 → 44 | 471 → 75 | 294 → 47 | 385 → 57 | 81 → 40 |
| `sensor_sum` | 353 → 53 | 560 → 147 | 353 → 54 | 446 → 101 | 99 → 44 |
| `sensor_threshold` | 405 → 111 | 653 → 235 | 405 → 111 | 522 → 175 | 150 → 118 |
| `sieve` | 3662 → 1972 | 12906 → 2492 | 3662 → 2134 | 4439 → 2772 | 2908 → 2132 |
| `spill_storm` | 3576 → 12 | 6767 → 19 | 4099 → 15 | 3576 → 12 | 3256 → 12 |
| `stencil3` | 1401 → 235 | 2526 → 449 | 1401 → 257 | 1699 → 403 | 368 → 210 |
| `store_then_div_trap` | 16 → 14 | 16 → 14 | 24 → 22 | 16 → 14 | 16 → 13 |
| `strength_ladder` | 5258 → 13 | 10240 → 19 | 5978 → 15 | 5266 → 13 | 5095 → 19 |
| `sum_cubes_mod` | 1752 → 15 | 3022 → 22 | 1992 → 18 | 1756 → 15 | 760 → 13 |
| `sum_sq` | 2690 → 14 | 4544 → 14 | 2690 → 15 | 2696 → 13 | 690 → 11 |
| `swap_via_memory` | 23 → 16 | 67 → 16 | 23 → 17 | 55 → 34 | 15 → 9 |
| `transpose8` | 5468 → 194 | 10282 → 210 | 5468 → 226 | 5472 → 236 | 1643 → 162 |
| `unreachable_blocks` | 8 → 3 | 8 → 3 | 8 → 5 | 10 → 3 | 9 → 3 |
| `unrolled_crc12` | 146 → 24 | 149 → 41 | 168 → 33 | 146 → 24 | 168 → 12 |
| `unrolled_poly` | 227 → 44 | 579 → 104 | 260 → 104 | 419 → 86 | 179 → 20 |
| `vec_add` | 285 → 72 | 565 → 89 | 285 → 74 | 349 → 136 | 97 → 48 |
| `vending_fsm` | 1639 → 460 | 2714 → 939 | 1751 → 556 | 1810 → 654 | 951 → 559 |
| `vlog_every_other` | 127 → 21 | 566 → 21 | 127 → 21 | 147 → 21 | 144 → 18 |
| `wide_expression` | 51 → 17 | 185 → 100 | 51 → 16 | 55 → 18 | 14 → 8 |
| `word_count` | 604 → 169 | 1505 → 432 | 604 → 202 | 755 → 329 | 374 → 77 |
| `xorshift` | 1078 → 143 | 1814 → 180 | 1078 → 161 | 1078 → 143 | 309 → 147 |
