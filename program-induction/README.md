# Program Induction (Fleet)

This folder contains a Fleet-based program induction prototype for dot-pattern images.

## What it does

- Defines a turtle-style DSL with primitives:
  - `left`
  - `right`
  - `up`
  - `down`
  - `penup`
  - `pendown`
  - `jump(location)` where `location` is one of:
    - `center`, `top_left`, `top`, `top_right`
    - `left_edge`, `right_edge`
    - `bottom_left`, `bottom`, `bottom_right`
  - `jumpxy(x,y)` where `x,y` are index terminals `0..9,a..f` (0..15), mapped onto a configurable grid (`--jump-grid` in `{4,8,16}`)
  - `seq(a,b)`
  - `repeat2(a)` ... `repeat8(a)`
- Uses MCMC inference to search over programs.
- Scores programs by rendering to a binary image with either:
  - `bernoulli` (pixelwise noise, `--epsilon`)
  - `dt` (distance-transform shape term, `--dt-weight`)
  - `bernoulli+dt` (sum of both; default)
- Returns top programs sorted by posterior quality.
- Supports prior predictive tests (`--mode prior-test`).

## Build

```bash
cd /Users/maxs/counting-psychophysics/program-induction
make
```

## Input image format (plain text)

- File contains rows of `0`/`1`.
- `1` means filled pixel; `0` means empty.
- Optional comments: lines starting with `#` are ignored.
- Whitespace in lines is ignored.

Example (`10x10`):

```text
0000000000
0000110000
0000110000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
```

## Run inference

```bash
cd /Users/maxs/counting-psychophysics/program-induction
./main \
  --mode infer \
  --target-image /path/to/image.txt \
  --width 200 --height 200 \
  --steps 20000 --chains 4 --time 120s \
  --max-temp 12 \
  --step-size 6 \
  --noise-model bernoulli+dt --epsilon 0.05 --dt-weight 12 \
  --jump-grid 16 \
  --top 100 --show-top 20
```

## Run prior predictive tests

```bash
cd /Users/maxs/counting-psychophysics/program-induction
./main \
  --mode prior-test \
  --width 200 --height 200 \
  --prior-tests 10 \
  --prior-steps 3000 \
  --steps 20000 --chains 4 --time 120s
```

## Optional exports

Use `--export-prefix /tmp/run1` to write PGM images:
- `/tmp/run1_target.pgm`
- `/tmp/run1_best.pgm`
- In prior-test mode: `/tmp/run1_target_<i>.pgm`
