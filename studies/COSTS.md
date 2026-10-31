# Expected compute

Measured on 2026-10-31 on the 64-core development machine while other jobs held it at full
load, so per-core rates may improve on an idle machine. One torch thread, CPU only. These are
planning figures from short measurements. Rerun them on the merged engine before a final run.

## Measured rates

| Operation | Rate |
|---|---|
| PPO training, [256, 256], 16 lanes | 1,561 environment steps per core-second |
| Evaluation through the frozen normaliser and policy | 990 steps per core-second |
| Fixed action replay, timing null | 5,375 steps per core-second |
| `score_run` on 32,640 returns | 1.5 seconds |
| Realism tier panel, certified | 0.017 seconds per seed |
| GARCH panel, certified | 0.008 seconds per seed |
| LOB exchangeability or mark-walk episode | 0.02 seconds per seed |
| Importing `sharpearena` and SB3 | about 145 seconds per worker process |

The import line is why a smoke run looks slow. It is paid once per worker, and most of it is
TensorFlow pulled in from the system site packages. A venv without system packages should
remove most of it, which is unmeasured.

## Realism study

About 0.26 core-seconds per seed across all arms, over 512 seeds, is 0.04 core-hours of work.
With imports on 8 workers the full study is under 1 core-hour.

## Learner study

One 2,000,000-step run:

| Part | Core-seconds |
|---|---|
| Training | 1,281 |
| Ten checkpoints on dev and 16 seen seeds | 97 |
| Final and best readouts | 66 |
| Zero-shot transfer on two tiers | 62 |
| Timing null, 200 draws over 32 seeds | 143 |
| Kernel scoring | 6 |
| Total | about 1,655, or 0.46 core-hours |

Protocol version 4 runs four cells in the main stage: A and D always, B and C when A passes
its capacity check. Counts below assume all four.

| Stage | Runs | Core-hours at 2M steps | Core-hours at 8M steps |
|---|---|---|---|
| Pilot (cell A only) | 20 | 9 | 31 |
| Main, 8 bands, four cells | 64 | 29 | 98 |
| Main, 16 bands, four cells | 128 | 59 | 196 |
| Main, 32 bands (cap), four cells | 256 | 118 | 392 |
| Controls and imports, per stage | | under 3 | under 3 |

Expected total: about 41 core-hours at 8 bands, and at most about 130 at the cap without
escalation. The worst case, the cap with the one permitted escalation, is about 425. If A fails
its capacity check, only A and D run and the main stage halves. Cell B trains a smaller network,
cell C evaluates more seen seeds and cell D adds two transfer tiers; none is measured
separately, and the table counts all at cell A's rate. The rates above were measured on Hard and
were not remeasured on `martingale_v1` or Calm. On 32
workers the capped main stage takes about 4 hours, or about 12 with escalation.

The realism study gains one tier, `martingale_v1`, and stays under 1 core-hour.

Disk use is small: job files are JSON summaries, and no model is retained, only its digest.
