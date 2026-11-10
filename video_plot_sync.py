"""動画の下に，現在位置を赤線で示したプロットを縦に結合した動画を作る．

CSV は 1 行に index と値の列．index は動画のフレーム番号（--index-unit sec なら秒）．
既定では 1 列目を index，2 列目を値として 1 本描く．--cols で複数の列を選ぶと複数の系列を
1 枚の図に描き，--right で指定した列は右の y 軸に描く（スケールの違う系列用）．
--sub R,C を付けると，列を重ねずに 1 列ずつ subplot(R, C, i) に並べる（赤線は全パネルで同期）．
ヘッダ行は列名として使い，数値に変換できない行は読み飛ばす．fps とフレーム数は省略すると ffprobe で調べる．

例:
    python video_plot_sync.py --video in.mp4 --csv data.csv -o out.mp4
    python video_plot_sync.py -i in.mp4 -c scores.csv --cols p_happy happy_neutral --right happy_neutral -o out.mp4
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from collections import namedtuple

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_rate(s):
    """'30000/1001' のような ffprobe の分数表記を float にする．取れなければ None．"""
    try:
        num, _, den = s.partition('/')
        r = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None
    return r if r > 0 else None


def ffprobe_stream(path, extra_args=(), entries='avg_frame_rate,r_frame_rate,nb_frames,duration'):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', *extra_args,
         '-show_entries', f'stream={entries}:format=duration', '-of', 'json', path],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    if not info.get('streams'):
        raise SystemExit(f'no video stream in {path}')
    return info['streams'][0], info.get('format', {})


def _positive(v, cast):
    try:
        v = cast(v)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def probe_video(path, fps=None, frames=None):
    """fps とフレーム数を ffprobe で調べる（引数で与えられた方はそのまま使う）．

    fps: avg_frame_rate（なければ r_frame_rate）．
    フレーム数: メタデータの nb_frames → パケットを数える → duration × fps の順に試す．
    可変フレームレートを検出したら警告し，fps を自動検出していれば「フレーム数 / 長さ」の
    実測平均 fps に置き換える（mkv などはヘッダの avg_frame_rate が当てにならないため）．
    """
    st, fmt = ffprobe_stream(path)
    avg = parse_rate(st.get('avg_frame_rate', ''))
    r = parse_rate(st.get('r_frame_rate', ''))
    duration = _positive(st.get('duration'), float) or _positive(fmt.get('duration'), float)

    fps_given = fps is not None
    if fps is None:
        fps = avg or r
        if fps is None:
            raise SystemExit(f'cannot detect fps of {path}; specify --fps')

    counted = _positive(st.get('nb_frames'), int)
    if counted is None:
        # デコードせずにパケットを数える（ファイルは全部読むが速い）
        st2, _ = ffprobe_stream(path, ['-count_packets'], entries='nb_read_packets')
        counted = _positive(st2.get('nb_read_packets'), int)

    vfr = avg is not None and r is not None and abs(avg - r) / r > 0.01
    if counted and duration:
        measured = counted / duration
        if abs(measured - fps) / fps > 0.01:
            vfr = True
            if not fps_given:
                fps = measured
    if vfr:
        print(f'warning: {path} looks variable frame rate; using {fps:.3f} fps. '
              'Frame index -> time mapping may drift (convert to CFR or give --fps)',
              file=sys.stderr)

    if frames is None:
        frames = counted or (int(round(duration * fps)) if duration else None)
        if frames is None:
            raise SystemExit(f'cannot detect frame count of {path}; specify --frames')
    return fps, frames


def _is_number(s):
    try:
        float(s)
    except ValueError:
        return False
    return True


def read_table(path):
    """CSV を (列名のリスト or None, 行のリスト) で返す．
    先頭の行に数値でないセルがあれば，それを列名（ヘッダ）とみなす（ファイル名などの文字列の列があってもよい）．
    ヘッダの行も rows に残すが，選んだ列が数値でなければ load_series で飛ばされる．"""
    with open(path, newline='') as f:
        rows = [r for r in csv.reader(f) if r]
    header = None
    if rows and not all(_is_number(c.strip()) for c in rows[0]):
        header = [c.strip() for c in rows[0]]
    return header, rows


def column_index(spec, header, path):
    """列の指定（列名，または 0 始まりの列番号）を列番号にする．列名を優先する．"""
    if header is not None and spec in header:
        return header.index(spec)
    try:
        i = int(spec)
    except ValueError:
        raise SystemExit(f'error: no column "{spec}" in {path} (columns: {header})')
    if header is not None and not 0 <= i < len(header):
        raise SystemExit(f'error: no column {i} in {path} (columns: {header})')
    return i


def load_series(path, index_col, col):
    """index 列と値の列を読む．どちらかが数値でない行（ヘッダ，空欄など）は飛ばす．"""
    header, rows = read_table(path)
    ix = column_index(index_col, header, path)
    iy = column_index(col, header, path)
    xs, ys = [], []
    for row in rows:
        if len(row) <= max(ix, iy):
            continue
        try:
            x, y = float(row[ix]), float(row[iy])
        except ValueError:
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        raise ValueError(f'no numeric rows of columns ({index_col}, {col}) in {path}')
    x = np.array(xs)
    y = np.array(ys)
    order = np.argsort(x, kind='stable')
    name = header[iy] if header is not None and iy < len(header) else f'column {iy}'
    return x[order], y[order], name


def load_csv(path):
    """(index, value) の組を読む（1 列目と 2 列目）．数値に変換できない行（ヘッダなど）は飛ばす．"""
    x, y, _ = load_series(path, '0', '1')
    return x, y


def window_bounds(x, win):
    """ソート済みの x について，各点を中心とした開区間 (t - win/2, t + win/2) に入る
    添字の範囲 [lo, hi) を二分探索でまとめて求める．"""
    h = win / 2
    lo = np.searchsorted(x, x - h, side='right')  # x > t - h となる最初の添字
    hi = np.searchsorted(x, x + h, side='left')   # x >= t + h となる最初の添字
    return lo, hi


def smooth_mean(x, y, win):
    """移動平均．累積和を使うので O(N)．NaN は無視する．"""
    lo, hi = window_bounds(x, win)
    valid = ~np.isnan(y)
    csum = np.concatenate(([0.0], np.cumsum(np.where(valid, y, 0.0))))
    ccnt = np.concatenate(([0], np.cumsum(valid)))
    with np.errstate(invalid='ignore', divide='ignore'):
        return (csum[hi] - csum[lo]) / (ccnt[hi] - ccnt[lo])


def smooth_median(x, y, win, chunk=8192):
    """移動中央値．窓を (点数, 最大窓幅) の 2 次元配列に並べ，はみ出しを NaN にして
    nanmedian を一度にかける．メモリを抑えるため chunk 点ずつ処理する．"""
    lo, hi = window_bounds(x, win)
    width = int((hi - lo).max())
    offsets = np.arange(width)
    ypad = np.append(y, np.nan)  # 窓の外を指す添字はここ（NaN）に向ける
    out = np.empty(len(y))
    for s in range(0, len(y), chunk):
        idx = lo[s:s + chunk, None] + offsets
        idx = np.where(idx < hi[s:s + chunk, None], idx, len(y))
        out[s:s + chunk] = np.nanmedian(ypad[idx], axis=1)
    return out


def y_limits(ys, ymin, ymax, invert):
    """系列の値（のリスト）から y 軸の範囲を決める．ymin / ymax が与えられればそれを使う．"""
    allv = np.concatenate([np.asarray(y, dtype=float) for y in ys])
    if ymin is not None or ymax is not None:
        ylo = ymin if ymin is not None else np.nanmin(allv)
        yhi = ymax if ymax is not None else np.nanmax(allv)
    else:
        ylo, yhi = np.nanmin(allv), np.nanmax(allv)
        margin = (yhi - ylo) * 0.05 or 1.0
        ylo, yhi = ylo - margin, yhi + margin
    return (yhi, ylo) if invert else (ylo, yhi)


# x[秒], y, 凡例の名前, 'left' / 'right'（y 軸），パネルの番号（--sub），CSV の番号，列名
Series = namedtuple('Series', 'x y label side panel csv name')


def plot_line(ax, x, y, gap, **kw):
    """--gap 秒以上データが抜けているところは線を切って描く．"""
    if gap > 0:
        breaks = np.where(np.diff(x) > gap)[0] + 1
        x, y = np.insert(x, breaks, np.nan), np.insert(y, breaks, np.nan)
    return ax.plot(x, y, **kw)[0]


def make_figure(args, series):
    """series: [Series]．side は 'left' か 'right'（右の y 軸）．
    系列が 1 本だけなら従来どおり青で描き，凡例は付けない．"""
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height), dpi=args.dpi)
    ax.grid()
    ax.set_xlabel('seconds', fontsize=12)
    if args.title:
        ax.set_title(args.title)

    left = [k for k, s in enumerate(series) if s[3] == 'left']
    right = [k for k, s in enumerate(series) if s[3] == 'right']
    if not left:
        raise SystemExit('error: all the series are on the right axis; plot at least one on the left')
    ax2 = ax.twinx() if right else None

    colors = (args.colors or []) + plt.rcParams['axes.prop_cycle'].by_key()['color']
    if len(series) == 1 and not args.colors:
        colors = ['blue']
    handles = []
    for k, s in enumerate(series):
        target = ax if s.side == 'left' else ax2
        handles.append(plot_line(target, s.x, s.y, args.gap, color=colors[k % len(colors)],
                                 label=s.label))

    ylo, yhi = y_limits([series[k][1] for k in left], args.ymin, args.ymax, args.invert_y)
    ax.set_ylim(ylo, yhi)
    if ax2 is not None:
        ax2.set_ylim(*y_limits([series[k][1] for k in right], args.ymin_right, args.ymax_right,
                               args.invert_y_right))

    # y 軸ラベル: 指定がなければ，その軸の系列が 1 本なら系列名（色も系列に合わせる）
    def label_axis(a, text, ks, default):
        own = len(ks) == 1 and len(series) > 1  # 系列が複数で，この軸には 1 本だけ
        if text is None:
            text = series[ks[0]][2] if own else default
        if text:
            kw = {'color': handles[ks[0]].get_color()} if own else {}
            a.set_ylabel(text, fontsize=12, **kw)
    label_axis(ax, args.ylabel, left, 'value')
    if ax2 is not None:
        label_axis(ax2, args.ylabel_right, right, '')
    if args.zero_line:
        ax.axhline(0, color='gray', lw=0.8, ls=':')
        if ax2 is not None:
            ax2.axhline(0, color='gray', lw=0.8, ls=':')

    if len(series) > 1 and args.legend != 'none':
        ax.legend(handles=handles, loc=args.legend, fontsize=10)

    # 赤線は左の軸に描く（右の軸があっても，縦線なので高さは左の範囲で足りる）
    line_red, = ax.plot([], [], color='red', lw=args.line_width, animated=True)
    fig.tight_layout()
    return fig, ax, [(ax, line_red, min(ylo, yhi), max(ylo, yhi))]


def make_subplots_figure(args, series):
    """--sub R,C: --cols の列ごとに subplot(R, C, i) に描く（左上から横に並べる）．
    CSV が複数なら，同じ列を各 CSV から同じパネルに重ねる（色は CSV ごと）．
    x 軸は共有し，赤線はパネルごとに描く．余ったパネルは消す．"""
    rows, ncols = args.sub
    npanel = max(s.panel for s in series) + 1
    fig, axes = plt.subplots(rows, ncols, sharex=True, squeeze=False,
                             figsize=(args.fig_width, args.fig_height), dpi=args.dpi)
    axes = axes.ravel()
    for a in axes[npanel:]:
        a.remove()
    axes = axes[:npanel]
    if args.title:
        fig.suptitle(args.title)

    ncsv = max(s.csv for s in series) + 1
    colors = (args.colors or []) + plt.rcParams['axes.prop_cycle'].by_key()['color']
    if not args.colors:
        colors = ['blue'] if ncsv == 1 else colors

    reds = []
    legend_handles = {}
    for p, a in enumerate(axes):
        ss = [s for s in series if s.panel == p]
        a.grid()
        for s in ss:
            # CSV が 1 つなら色はパネルごと（--colors を与えたとき），複数なら CSV ごと
            k = (p if ncsv == 1 else s.csv) % len(colors)
            h = plot_line(a, s.x, s.y, args.gap, color=colors[k], label=s.label)
            legend_handles.setdefault(s.csv, h)
        a.set_title(ss[0].name if ncsv > 1 else ss[0].label, fontsize=11)
        if args.ylabel:
            a.set_ylabel(args.ylabel, fontsize=11)
        ylo, yhi = y_limits([s.y for s in ss], args.ymin, args.ymax, args.invert_y)
        a.set_ylim(ylo, yhi)
        if args.zero_line:
            a.axhline(0, color='gray', lw=0.8, ls=':')
        # 一番下の段か，下のパネルが消えている列には x 軸の目盛りとラベルを付ける
        if p + ncols >= npanel:
            a.xaxis.set_tick_params(labelbottom=True)
            a.set_xlabel('seconds', fontsize=11)
        line_red, = a.plot([], [], color='red', lw=args.line_width, animated=True)
        reds.append((a, line_red, min(ylo, yhi), max(ylo, yhi)))

    # 凡例は CSV が複数のときだけ，左上のパネルに 1 つ（色は全パネル共通なので）
    if ncsv > 1 and args.legend != 'none':
        axes[0].legend(handles=[legend_handles[c] for c in sorted(legend_handles)],
                       loc=args.legend, fontsize=9)
    fig.tight_layout()
    return fig, axes[0], reds


def build_series(args, fps):
    """--csv・--cols・--right から [Series] を作り，平滑化をかける．
    --sub のときは，panel に --cols の何番目の列か（= 何番目のパネルか）を入れる．"""
    cols = args.cols or ['1']
    right = set(args.right or [])
    unknown = right - set(cols)
    if unknown:
        cols = cols + [c for c in args.right if c in unknown]  # --right だけに書いた列も描く
    if args.labels and len(args.labels) != len(args.csv) * len(cols):
        raise SystemExit(f'error: --labels needs {len(args.csv) * len(cols)} names '
                         f'(CSV files x columns), got {len(args.labels)}')
    if args.mean and args.median:
        raise SystemExit('--mean and --median are exclusive')

    if args.sub:
        if args.right:
            raise SystemExit('error: --right cannot be used with --sub')
        rows, ncols = args.sub
        if rows * ncols < len(cols):
            raise SystemExit(f'error: --sub {rows},{ncols} has {rows * ncols} panels '
                             f'but --cols has {len(cols)} columns')

    series = []
    for ci, path in enumerate(args.csv):
        for col in cols:
            x, y, name = load_series(path, args.index_col, col)
            if args.index_unit == 'frame':
                x = x / fps
            if args.mean:
                y = smooth_mean(x, y, args.window)
            elif args.median:
                y = smooth_median(x, y, args.window)
            stem = os.path.splitext(os.path.basename(path))[0]
            if args.sub:
                # パネルの名前は列名（パネルのタイトルにする），凡例は CSV ごと（ファイル名）
                label = stem if len(args.csv) > 1 else name
                series.append(Series(x, y, label, 'left', cols.index(col), ci, name))
                continue
            if len(args.csv) > 1:
                name = f'{stem}: {name}'
            series.append(Series(x, y, name, 'right' if col in right else 'left', 0, ci, name))
    if args.labels:
        series = [s._replace(label=label) for s, label in zip(series, args.labels)]
    return series


def run(args):
    if args.fps is not None and args.frames is not None:
        fps, frames = args.fps, args.frames
    else:
        fps, frames = probe_video(args.video, args.fps, args.frames)
    print(f'fps: {fps:.6g} ({"given" if args.fps else "detected"}), '
          f'frames: {frames} ({"given" if args.frames else "detected"})', file=sys.stderr)
    duration = frames / fps
    step = args.step if args.step > 0 else duration  # 1 画面に表示する秒数

    series = build_series(args, fps)
    fig, ax, reds = (make_subplots_figure if args.sub else make_figure)(args, series)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # 入力 0: 元動画，入力 1: プロットの rawvideo（stdin）を 1 回の ffmpeg で縮小して縦に結合する
    width = args.width
    threads = str(args.threads)
    filt = (f'[0:v]scale={width}:-2,setsar=1[top];'
            f'[1:v]scale={width}:-2,setsar=1[bottom];'
            f'[top][bottom]vstack=inputs=2:shortest=1[v]')
    # 上書きの確認は main() で済ませてある．-n は念のため（その間に作られたファイルも上書きしない）
    cmd = ['ffmpeg', '-y' if args.overwrite else '-n', '-loglevel', 'error', '-stats',
           '-filter_threads', threads, '-filter_complex_threads', threads,
           # -t: 元動画は必要な長さだけ読む（これがないと --frames で映像を短くしても
           #     音声が元動画の最後まで続き，ffmpeg が動画全体を読み続ける）
           '-threads', threads, '-t', f'{duration:.6f}', '-i', args.video,
           '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
           '-filter_complex', filt, '-map', '[v]']
    if not args.no_audio:
        cmd += ['-map', '0:a?', '-c:a', 'aac']
    cmd += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', str(args.crf),
            '-threads', threads, '-shortest', args.output]
    ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # 表示区間が変わるときだけ図全体を描いて背景として保存し，
    # フレームごとに背景を戻して赤線だけ描く
    current = None
    background = None
    try:
        for i in range(frames):
            t = i / fps
            j = int(t // step)
            if j != current:
                ax.set_xlim(j * step, (j + 1) * step)
                if args.xtick > 0:
                    ax.set_xticks(np.arange(j * step, (j + 1) * step + 1e-9, args.xtick))
                fig.canvas.draw()
                background = fig.canvas.copy_from_bbox(fig.bbox)
                current = j
            fig.canvas.restore_region(background)
            for a, line_red, ylo, yhi in reds:  # --sub ならパネルごとに 1 本
                line_red.set_data([t, t], [ylo, yhi])
                a.draw_artist(line_red)
            ffmpeg.stdin.write(fig.canvas.buffer_rgba())
    except BrokenPipeError:
        pass
    finally:
        try:
            ffmpeg.stdin.close()
        except BrokenPipeError:
            pass
    if ffmpeg.wait() != 0:
        raise SystemExit(f'ffmpeg failed: {args.output}')


def parse_grid(s):
    """--sub の 'R,C'（'RxC' も可）を (R, C) にする．"""
    try:
        r, c = (int(v) for v in s.lower().replace('x', ',').split(','))
    except ValueError:
        raise argparse.ArgumentTypeError(f'expected ROWS,COLS (e.g. 2,3), got "{s}"')
    if r < 1 or c < 1:
        raise argparse.ArgumentTypeError(f'ROWS and COLS must be >= 1, got "{s}"')
    return r, c


def check_output(video, output, overwrite):
    """重い処理を始める前に出力先を確かめる．
    入力と同じファイルなら常に止め，すでにあるファイルは --overwrite のときだけ上書きを許す．"""
    same = (os.path.realpath(video) == os.path.realpath(output)
            or (os.path.exists(output) and os.path.exists(video) and os.path.samefile(video, output)))
    if same:
        raise SystemExit(f'error: output is the same file as the input video: {output}')
    if os.path.exists(output) and not overwrite:
        raise SystemExit(f'error: {output} already exists (use --overwrite / -y to replace it)')


def main():
    p = argparse.ArgumentParser(description='動画とプロットを縦に結合し，プロットに現在位置の赤線を引く')
    p.add_argument('--video', '-i', required=True, help='入力動画')
    p.add_argument('--csv', '-c', required=True, nargs='+',
                   help='index と値の列の CSV．複数を与えると，それぞれから --cols の列を描く')
    p.add_argument('--index-col', default='0',
                   help='index の列（列名，または 0 始まりの列番号．default: 0）')
    p.add_argument('--cols', nargs='+', default=None,
                   help='描く値の列（列名，または 0 始まりの列番号）．複数なら 1 枚の図に重ねる（default: 1）')
    p.add_argument('--right', nargs='+', default=None,
                   help='右の y 軸に描く列（--cols と同じ書き方）．スケールの違う系列用')
    p.add_argument('--labels', nargs='+', default=None,
                   help='凡例の名前（系列の順: CSV ごとに --cols の順）．省略すると列名')
    p.add_argument('--sub', type=parse_grid, default=None, metavar='ROWS,COLS',
                   help='--cols の列を 1 つずつ subplot(ROWS, COLS, i) に並べて描く（例: --sub 2,3）．'
                        'CSV が複数なら同じ列を同じパネルに重ねる．--right とは併用できない')
    p.add_argument('--colors', nargs='+', default=None, help='系列の色（matplotlib の色名など）')
    p.add_argument('--legend', default='upper left',
                   help='凡例の位置（matplotlib の loc）．none なら付けない（系列が 1 本なら付けない）')
    p.add_argument('--fps', type=float, default=None,
                   help='動画の fps．省略すると ffprobe で調べる')
    p.add_argument('--frames', type=int, default=None,
                   help='出力するフレーム数．省略すると ffprobe で動画のフレーム数を調べる')
    p.add_argument('--output', '-o', default='output.mp4', help='出力動画')

    p.add_argument('--index-unit', choices=['frame', 'sec'], default='frame',
                   help='CSV の index の単位（default: frame）')
    p.add_argument('--step', type=float, default=0,
                   help='1 画面に表示する秒数．0 なら動画全体（default: 0）')
    p.add_argument('--xtick', type=float, default=0, help='x 軸目盛りの間隔 [秒]．0 なら自動')
    p.add_argument('--gap', type=float, default=0.5,
                   help='この秒数以上データが抜けている所で線を切る．0 なら全部つなぐ')
    p.add_argument('--mean', action='store_true', help='移動平均をかける')
    p.add_argument('--median', action='store_true', help='移動中央値をかける')
    p.add_argument('--window', type=float, default=1.0, help='平滑化の窓幅 [秒]')

    p.add_argument('--title', default='', help='プロットのタイトル')
    p.add_argument('--ylabel', default=None,
                   help='（左の）y 軸ラベル．省略すると value（左の系列が 1 本で系列が複数なら系列名）')
    p.add_argument('--ymin', type=float, default=None, help='（左の）y 軸の下限')
    p.add_argument('--ymax', type=float, default=None, help='（左の）y 軸の上限')
    p.add_argument('--invert-y', action='store_true', help='（左の）y 軸を上下反転する')
    p.add_argument('--ylabel-right', default=None,
                   help='右の y 軸ラベル．省略すると，右の系列が 1 本なら系列名')
    p.add_argument('--ymin-right', type=float, default=None, help='右の y 軸の下限')
    p.add_argument('--ymax-right', type=float, default=None, help='右の y 軸の上限')
    p.add_argument('--invert-y-right', action='store_true', help='右の y 軸を上下反転する')
    p.add_argument('--zero-line', action='store_true', help='y = 0 に点線を引く（左右の軸それぞれ）')
    p.add_argument('--line-width', type=float, default=3, help='赤線の太さ')

    p.add_argument('--width', type=int, default=800, help='出力動画の幅 [px]')
    p.add_argument('--fig-width', type=float, default=15, help='プロットの幅 [inch]')
    p.add_argument('--fig-height', type=float, default=None,
                   help='プロットの高さ [inch]（default: 4．--sub なら 2.5 × 行数 + 0.5）')
    p.add_argument('--dpi', type=int, default=100)
    p.add_argument('--crf', type=int, default=23, help='x264 の画質（小さいほど高画質）')
    p.add_argument('--no-audio', action='store_true', help='音声を含めない')
    p.add_argument('--threads', type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help='使うコア数（ffmpeg のスレッド数もこれにそろえる）．0 なら制限しない'
                        '（default: コア数の半分）')
    p.add_argument('--nice', type=int, default=10,
                   help='処理の優先度を下げる量（0〜19）．0 なら下げない（default: 10）')
    p.add_argument('--overwrite', '-y', action='store_true',
                   help='出力ファイルがすでにあれば上書きする（既定では止める）')
    args = p.parse_args()
    if args.legend.lower() == 'none':
        args.legend = 'none'
    if args.fig_height is None:
        args.fig_height = 2.5 * args.sub[0] + 0.5 if args.sub else 4
    check_output(args.video, args.output, args.overwrite)
    # 優先度と使うコアは ffmpeg（子プロセス）にも引き継がれる
    if args.nice > 0:
        os.nice(args.nice)
    if args.threads > 0 and hasattr(os, 'sched_setaffinity'):
        # -threads はデコード・フィルタ・エンコードそれぞれの上限で合計は超えるので，
        # 使えるコア自体を threads 個に絞って PC 全体が重くならないようにする
        cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, cpus[:args.threads])
    run(args)


if __name__ == '__main__':
    main()
