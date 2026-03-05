#!/usr/bin/env python3
import argparse
import concurrent.futures
import html
import os
import pathlib
import subprocess
import time
from typing import List, Tuple
from collections import deque

ROOT = pathlib.Path('/Users/maxs/counting-psychophysics/program-induction')
OUT = ROOT / 'gallery'
TARGETS = OUT / 'targets'
RUNS = OUT / 'runs'
SVG = OUT / 'svg'

W = 80
H = 80
STEPS = 1000
CHAINS = 4
TOP = 12
SHOW_TOP = 8
TIME = '15s'
MAX_WORKERS = max(2, min(8, (os.cpu_count() or 4)))
STEP_SIZE = 6
JUMP_GRID = 16


def blank() -> List[List[int]]:
    return [[0 for _ in range(W)] for _ in range(H)]


def draw_line(img, x0, y0, x1, y1):
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < W and 0 <= y0 < H:
            img[y0][x0] = 1
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def jump_idx(px: int, extent: int) -> int:
    if extent <= 1:
        return 0
    idx = int(round((px * (JUMP_GRID - 1)) / float(extent - 1)))
    return max(0, min(JUMP_GRID - 1, idx))


def resolve_jump_idx(idx: int, extent: int) -> float:
    if extent <= 1:
        return 0.0
    i = max(0, min(JUMP_GRID - 1, idx))
    return (float(i) / float(JUMP_GRID - 1)) * float(extent - 1)


def line_ops_by_steps(dx_steps: int, dy_steps: int) -> List[Tuple[str, int]]:
    ops: List[Tuple[str, int]] = []

    def emit(name: str):
        if ops and ops[-1][0] == name:
            ops[-1] = (name, ops[-1][1] + 1)
        else:
            ops.append((name, 1))

    ax = abs(dx_steps)
    ay = abs(dy_steps)
    sx = 1 if dx_steps >= 0 else -1
    sy = 1 if dy_steps >= 0 else -1

    if ax == 0 and ay == 0:
        return ops

    if ax >= ay:
        err = ax // 2
        for _ in range(ax):
            emit('right' if sx > 0 else 'left')
            err -= ay
            if err < 0 and ay > 0:
                emit('down' if sy > 0 else 'up')
                err += ax
    else:
        err = ay // 2
        for _ in range(ay):
            emit('down' if sy > 0 else 'up')
            err -= ax
            if err < 0 and ax > 0:
                emit('right' if sx > 0 else 'left')
                err += ay

    return ops


def line_segment_ops(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[str, int]]:
    dx_steps = int(round((x1 - x0) / float(STEP_SIZE)))
    dy_steps = int(round((y1 - y0) / float(STEP_SIZE)))
    return line_ops_by_steps(dx_steps, dy_steps)


def render_primitives(program: List[Tuple]) -> List[List[int]]:
    img = blank()
    x = (W - 1) / 2.0
    y = (H - 1) / 2.0
    pen_down = True

    for ins in program:
        op = ins[0]
        if op in ('left', 'right', 'up', 'down'):
            n = ins[1] if len(ins) > 1 else 1
            for _ in range(max(1, int(n))):
                nx, ny = x, y
                if op == 'left':
                    nx = x - STEP_SIZE
                elif op == 'right':
                    nx = x + STEP_SIZE
                elif op == 'up':
                    ny = y - STEP_SIZE
                elif op == 'down':
                    ny = y + STEP_SIZE
                if pen_down:
                    draw_line(img, int(round(x)), int(round(y)), int(round(nx)), int(round(ny)))
                x, y = nx, ny
        elif op == 'penup':
            pen_down = False
        elif op == 'pendown':
            pen_down = True
        elif op == 'jumpxy':
            x_idx = int(ins[1])
            y_idx = int(ins[2])
            x = resolve_jump_idx(x_idx, W)
            y = resolve_jump_idx(y_idx, H)
        else:
            raise RuntimeError(f'Unknown primitive op: {op}')

    return img


def polyline_program(points: List[Tuple[int, int]], close: bool = False) -> List[Tuple]:
    if not points:
        return []

    prog: List[Tuple] = []
    x0, y0 = points[0]
    prog.append(('penup',))
    prog.append(('jumpxy', jump_idx(x0, W), jump_idx(y0, H)))
    prog.append(('pendown',))

    seg_points = points + [points[0]] if close else points
    for (xa, ya), (xb, yb) in zip(seg_points[:-1], seg_points[1:]):
        prog.extend(line_segment_ops(xa, ya, xb, yb))

    return prog


def write_txt(path: pathlib.Path, img: List[List[int]]):
    with path.open('w') as f:
        for row in img:
            f.write(''.join('1' if v else '0' for v in row) + '\n')


def read_txt(path: pathlib.Path) -> List[List[int]]:
    rows = []
    for line in path.read_text().splitlines():
        line = ''.join(c for c in line if c in '01')
        if line:
            rows.append([1 if c == '1' else 0 for c in line])
    return rows


def read_pgm(path: pathlib.Path) -> List[List[int]]:
    tokens = []
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith('#'):
            continue
        tokens.extend(s.split())
    if tokens[0] != 'P2':
        raise RuntimeError(f'Unsupported PGM format in {path}')
    w = int(tokens[1])
    h = int(tokens[2])
    maxv = int(tokens[3])
    vals = list(map(int, tokens[4:4 + w * h]))
    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            v = vals[y * w + x]
            row.append(1 if v < maxv // 2 else 0)
        rows.append(row)
    return rows


def to_svg(rows: List[List[int]], out_path: pathlib.Path, scale: int = 4):
    h = len(rows)
    w = len(rows[0]) if h else 0
    with out_path.open('w') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w*scale}" height="{h*scale}" viewBox="0 0 {w} {h}" shape-rendering="crispEdges">\n')
        f.write('<rect width="100%" height="100%" fill="white"/>\n')
        for y, row in enumerate(rows):
            for x, v in enumerate(row):
                if v:
                    f.write(f'<rect x="{x}" y="{y}" width="1" height="1" fill="black"/>\n')
        f.write('</svg>\n')


def fmt_num(v: str) -> str:
    try:
        x = float(v)
        ax = abs(x)
        if ax == 0:
            return '0'
        if ax >= 1000:
            return f'{x:.0f}'
        if ax >= 100:
            return f'{x:.1f}'
        if ax >= 1:
            return f'{x:.2f}'
        if ax >= 0.01:
            return f'{x:.3f}'
        return f'{x:.2e}'
    except ValueError:
        return v


def pattern_specs() -> List[Tuple[str, List[List[int]]]]:
    out = []

    programs: List[Tuple[str, List[Tuple]]] = [
        ('cross', polyline_program([(10, 40), (70, 40)]) + polyline_program([(40, 10), (40, 70)])),
        ('square_frame', polyline_program([(18, 18), (62, 18), (62, 62), (18, 62)], close=True)),
        ('x_diagonals', polyline_program([(8, 8), (72, 72)]) + polyline_program([(8, 72), (72, 8)])),
        ('zigzag', polyline_program([(8, 65), (20, 15), (32, 65), (44, 15), (56, 65), (68, 15)])),
        ('triangle', polyline_program([(10, 10), (70, 10), (40, 70), (10, 10)])),
        ('square_plus',
            polyline_program([(12, 12), (68, 12), (68, 68), (12, 68)], close=True) +
            polyline_program([(12, 40), (68, 40)]) +
            polyline_program([(40, 12), (40, 68)])),
        ('star_simple',
            polyline_program([(12, 12), (68, 68)]) +
            polyline_program([(12, 68), (68, 12)]) +
            polyline_program([(40, 8), (40, 72)]) +
            polyline_program([(8, 40), (72, 40)])),
        ('horizontal_stripes',
            polyline_program([(12, 12), (68, 12)]) +
            polyline_program([(12, 24), (68, 24)]) +
            polyline_program([(12, 36), (68, 36)]) +
            polyline_program([(12, 48), (68, 48)]) +
            polyline_program([(12, 60), (68, 60)])),
        ('vertical_stripes',
            polyline_program([(12, 12), (12, 68)]) +
            polyline_program([(24, 12), (24, 68)]) +
            polyline_program([(36, 12), (36, 68)]) +
            polyline_program([(48, 12), (48, 68)]) +
            polyline_program([(60, 12), (60, 68)])),
        ('spiral_box', polyline_program([(10, 10), (70, 10), (70, 70), (22, 70), (22, 22), (58, 22), (58, 58), (34, 58), (34, 34), (46, 34), (46, 46)])),
    ]

    for name, prog in programs:
        out.append((name, render_primitives(prog)))

    return out


def run_one(
    name: str,
    txt_path: pathlib.Path,
    mode: str,
    steps: int,
    chains: int,
    top: int,
    show_top: int,
    time_limit: str,
    enum_steps: int,
    enum_method: str,
    noise_model: str,
    epsilon: float,
    dt_weight: float,
    f1_weight: float,
):
    run_dir = RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / 'infer.log'
    prefix = run_dir / 'out'

    cmd = [
        str(ROOT / 'main'),
        '--mode', mode,
        '--target-image', str(txt_path),
        '--width', str(W),
        '--height', str(H),
        '--top', str(top),
        '--show-top', str(show_top),
        '--export-prefix', str(prefix),
        '--noise-model', noise_model,
        '--epsilon', str(epsilon),
        '--dt-weight', str(dt_weight),
        '--f1-weight', str(f1_weight),
        '--jump-grid', '16',
        '--step-size', '6',
    ]
    if mode == 'infer':
        cmd.extend(['--steps', str(steps), '--chains', str(chains), '--time', time_limit])
    else:
        cmd.extend(['--enum-steps', str(enum_steps), '--enum-method', enum_method])

    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    tail = deque(maxlen=200)
    rankings = []
    assert proc.stdout is not None
    with log_path.open('w') as logf:
        for line in proc.stdout:
            logf.write(line)
            tail.append(line)

            s = line.strip()
            if s.startswith('# progress'):
                print(f'[{name}] {s[2:]}', flush=True)

            if s and not s.startswith('#'):
                parts = s.split('\t')
                if len(parts) >= 7 and parts[0].isdigit():
                    rankings.append({
                        'rank': parts[0],
                        'posterior': parts[1],
                        'prior': parts[2],
                        'likelihood': parts[3],
                        'weight': parts[4],
                        'mismatches': parts[5],
                        'program': parts[6],
                    })

    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd, output=''.join(tail))

    return rankings


def main():
    parser = argparse.ArgumentParser(description='Generate Fleet program-induction gallery in parallel.')
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--chains', type=int, default=CHAINS)
    parser.add_argument('--top', type=int, default=TOP)
    parser.add_argument('--show-top', type=int, default=SHOW_TOP)
    parser.add_argument('--time', type=str, default=TIME)
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--mode', choices=['infer', 'enumerate'], default='infer')
    parser.add_argument('--enum-steps', type=int, default=3000)
    parser.add_argument('--enum-method', choices=['basic', 'partial', 'full'], default='full')
    parser.add_argument('--noise-model', choices=['bernoulli', 'dt', 'f1', 'bernoulli+dt', 'f1+dt'], default='bernoulli+dt')
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--dt-weight', type=float, default=12.0)
    parser.add_argument('--f1-weight', type=float, default=220.0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    TARGETS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    SVG.mkdir(parents=True, exist_ok=True)

    specs = pattern_specs()
    for name, img in specs:
        write_txt(TARGETS / f'{name}.txt', img)

    results = {}
    t0 = time.time()
    total_jobs = len(specs)
    done_jobs = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        fut_to_name = {}
        for name, _ in specs:
            txt_path = TARGETS / f'{name}.txt'
            fut = ex.submit(
                run_one,
                name,
                txt_path,
                args.mode,
                args.steps,
                args.chains,
                args.top,
                args.show_top,
                args.time,
                args.enum_steps,
                args.enum_method,
                args.noise_model,
                args.epsilon,
                args.dt_weight,
                args.f1_weight,
            )
            fut_to_name[fut] = name
        for fut in concurrent.futures.as_completed(fut_to_name):
            name = fut_to_name[fut]
            results[name] = fut.result()
            done_jobs += 1
            elapsed = max(1e-9, time.time() - t0)
            rate = done_jobs / elapsed
            remain = total_jobs - done_jobs
            eta = remain / max(1e-9, rate)
            pct = int((done_jobs / max(1, total_jobs)) * 100)
            print(
                f'[gallery] {done_jobs}/{total_jobs} ({pct}%) '
                f'elapsed={elapsed:.1f}s eta={eta:.1f}s done={name}',
                flush=True,
            )

    rows = []
    for name, _ in specs:
        txt_path = TARGETS / f'{name}.txt'
        rankings = results[name]

        tgt_svg = SVG / f'{name}_target.svg'
        to_svg(read_txt(txt_path), tgt_svg)

        best_pgm = RUNS / name / 'out_best.pgm'
        best_svg = SVG / f'{name}_best.svg'
        if best_pgm.exists():
            to_svg(read_pgm(best_pgm), best_svg)

        best_mismatch = None
        if rankings:
            try:
                best_mismatch = int(rankings[0]['mismatches'])
            except ValueError:
                best_mismatch = None

        rows.append((name, tgt_svg, best_svg if best_pgm.exists() else None, rankings, best_mismatch))

    rows.sort(key=lambda x: (10**9 if x[4] is None else x[4], x[0]))

    html_lines = []
    html_lines.append('<!doctype html><html><head><meta charset="utf-8">')
    html_lines.append('<title>Program Induction Gallery</title>')
    html_lines.append('<style>')
    html_lines.append(':root{--paper:#f3efe6;--ink:#1f2a27;--card:#fffdfa;--muted:#6c756f;--line:#d7d0c1;--accent:#1e6f5c;--rank:#274754;--codebg:#f7f9fb;}')
    html_lines.append('*{box-sizing:border-box;}')
    html_lines.append('body{font-family:"Iosevka","JetBrains Mono","SF Mono","Menlo",monospace;margin:28px;background:radial-gradient(circle at 0 0,#fdf8ef 0,#f3efe6 45%,#eee9de 100%);color:var(--ink);}')
    html_lines.append('h1{margin:0;font-size:30px;letter-spacing:0.2px;}')
    html_lines.append('.sub{margin:8px 0 0 0;color:var(--muted);font-size:14px;}')
    html_lines.append('.meta{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 22px 0;}')
    html_lines.append('.chip{background:#fff;border:1px solid var(--line);border-radius:999px;padding:4px 10px;font-size:12px;color:#3a433f;}')
    html_lines.append('.grid{display:grid;grid-template-columns:1fr;gap:0;}')
    html_lines.append('.example{padding:16px 0 20px 0;border-top:1px solid var(--line);}')
    html_lines.append('.example:first-child{border-top:0;padding-top:4px;}')
    html_lines.append('.card-head{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:10px;}')
    html_lines.append('.title{font-size:18px;font-weight:700;}')
    html_lines.append('.score{font-size:12px;color:#173e35;background:#e7f2ee;border:1px solid #bfdcd2;border-radius:999px;padding:4px 10px;}')
    html_lines.append('.layout{display:grid;grid-template-columns:minmax(420px,560px) 1fr;gap:16px;align-items:start;}')
    html_lines.append('.imgs{display:grid;grid-template-columns:1fr 1fr;gap:10px;}')
    html_lines.append('.img-card{border:1px solid var(--line);border-radius:10px;padding:8px;background:#fff;}')
    html_lines.append('.img-card figcaption{font-size:11px;color:var(--muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:0.4px;}')
    html_lines.append('.img-card img{width:100%;height:auto;min-height:220px;border:1px solid #aeb6af;image-rendering:pixelated;background:white;}')
    html_lines.append('.prog-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;}')
    html_lines.append('.prog-col{padding:0 0 0 10px;border-left:2px solid #dde4ea;}')
    html_lines.append('.prog-meta{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:11px;margin-bottom:7px;color:#3d4743;}')
    html_lines.append('.badge{display:inline-block;background:var(--rank);color:white;border-radius:999px;padding:2px 8px;font-size:11px;}')
    html_lines.append('.kv{background:#f3f6f8;border:1px solid #d9e0e5;border-radius:999px;padding:2px 8px;}')
    html_lines.append('pre{margin:0;white-space:pre-wrap;word-break:break-word;font-size:12px;line-height:1.3;background:var(--codebg);border:1px solid #dde4ea;border-radius:8px;padding:8px;min-height:78px;}')
    html_lines.append('@media (max-width: 980px){.layout{grid-template-columns:1fr;}.imgs{grid-template-columns:1fr 1fr;}}')
    html_lines.append('@media (max-width: 680px){.imgs{grid-template-columns:1fr;}}')
    html_lines.append('</style></head><body>')
    html_lines.append('<h1>Fleet Program Induction Gallery</h1>')
    html_lines.append('<p class="sub">Target images and ranked imperative programs inferred by Fleet.</p>')
    html_lines.append('<div class="meta">')
    html_lines.append(f'<span class="chip">canvas {W}x{H}</span>')
    html_lines.append(f'<span class="chip">steps {args.steps}</span>')
    html_lines.append(f'<span class="chip">chains {args.chains}</span>')
    html_lines.append(f'<span class="chip">time {args.time}</span>')
    html_lines.append(f'<span class="chip">noise {args.noise_model}</span>')
    html_lines.append(f'<span class="chip">dt-w {args.dt_weight:g}</span>')
    if 'f1' in args.noise_model:
        html_lines.append(f'<span class="chip">f1-w {args.f1_weight:g}</span>')
    html_lines.append(f'<span class="chip">top {args.top}</span>')
    html_lines.append(f'<span class="chip">workers {max(1, args.workers)}</span>')
    html_lines.append(f'<span class="chip">mode {args.mode}</span>')
    if args.mode == 'enumerate':
        html_lines.append(f'<span class="chip">enum {args.enum_method}</span>')
        html_lines.append(f'<span class="chip">enum-steps {args.enum_steps}</span>')
    html_lines.append(f'<span class="chip">targets {len(rows)}</span>')
    html_lines.append('</div>')
    html_lines.append('<div class="grid">')

    for name, tgt_svg, best_svg, rankings, best_mismatch in rows:
        html_lines.append('<section class="example">')
        mismatch_label = 'n/a' if best_mismatch is None else str(best_mismatch)
        html_lines.append('<div class="card-head">')
        html_lines.append(f'<div class="title">{html.escape(name)}</div>')
        html_lines.append(f'<div class="score">best mismatch: {mismatch_label}</div>')
        html_lines.append('</div>')
        html_lines.append('<div class="layout">')
        html_lines.append('<div class="imgs">')
        html_lines.append(f'<figure class="img-card"><figcaption>target</figcaption><img src="{tgt_svg.relative_to(OUT)}"></figure>')
        if best_svg:
            html_lines.append(f'<figure class="img-card"><figcaption>best predicted</figcaption><img src="{best_svg.relative_to(OUT)}"></figure>')
        else:
            html_lines.append('<figure class="img-card"><figcaption>best predicted</figcaption><div style="font-size:12px;color:#666;padding:12px;">no rendered output</div></figure>')
        html_lines.append('</div>')
        html_lines.append('<div class="prog-grid">')
        for r in rankings:
            program_text = html.escape(r['program'].replace('\\n', '\n'))
            html_lines.append('<div class="prog-col">')
            html_lines.append('<div class="prog-meta">')
            html_lines.append(f'<span class="badge">rank {html.escape(r["rank"])}</span>')
            html_lines.append(f'<span class="kv">post {html.escape(fmt_num(r["posterior"]))}</span>')
            html_lines.append(f'<span class="kv">w {html.escape(fmt_num(r["weight"]))}</span>')
            html_lines.append(f'<span class="kv">mismatch {html.escape(fmt_num(r["mismatches"]))}</span>')
            html_lines.append('</div>')
            html_lines.append(f'<pre>{program_text}</pre>')
            html_lines.append('</div>')
        html_lines.append('</div>')
        html_lines.append('</section>')

    html_lines.append('</div></body></html>')

    (OUT / 'index.html').write_text('\n'.join(html_lines))
    print(str(OUT / 'index.html'))


if __name__ == '__main__':
    main()
