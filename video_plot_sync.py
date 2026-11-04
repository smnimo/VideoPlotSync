"""動画の下に，現在位置を赤線で示したプロットを縦に結合した動画を作る．

CSV は 1 行に (index, value) の組．index は動画のフレーム番号（--index-unit sec なら秒）．
ヘッダ行は自動で読み飛ばす．fps とフレーム数は省略すると ffprobe で調べる．

例:
    python video_plot_sync.py --video in.mp4 --csv data.csv -o out.mp4
"""
import argparse
import csv
import json
import subprocess
import sys

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


def load_csv(path):
    """(index, value) の組を読む．数値に変換できない行（ヘッダなど）は飛ばす．"""
    xs, ys = [], []
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                x, y = float(row[0]), float(row[1])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError(f'no numeric (index, value) rows in {path}')
    x = np.array(xs)
    y = np.array(ys)
    order = np.argsort(x, kind='stable')
    return x[order], y[order]


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


def make_figure(args, x, y):
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height), dpi=args.dpi)
    ax.grid()
    ax.set_xlabel('seconds', fontsize=12)
    if args.ylabel:
        ax.set_ylabel(args.ylabel, fontsize=12)
    if args.title:
        ax.set_title(args.title)

    if args.ymin is not None or args.ymax is not None:
        ylo = args.ymin if args.ymin is not None else np.nanmin(y)
        yhi = args.ymax if args.ymax is not None else np.nanmax(y)
    else:
        ylo, yhi = np.nanmin(y), np.nanmax(y)
        margin = (yhi - ylo) * 0.05 or 1.0
        ylo, yhi = ylo - margin, yhi + margin
    ax.set_ylim((yhi, ylo) if args.invert_y else (ylo, yhi))

    # --gap 秒以上データが抜けているところは線を切る
    if args.gap > 0:
        breaks = np.where(np.diff(x) > args.gap)[0] + 1
        ax.plot(np.insert(x, breaks, np.nan), np.insert(y, breaks, np.nan), color='blue')
    else:
        ax.plot(x, y, color='blue')

    line_red, = ax.plot([], [], color='red', lw=args.line_width, animated=True)
    fig.tight_layout()
    return fig, ax, line_red, (min(ylo, yhi), max(ylo, yhi))


def run(args):
    if args.fps is not None and args.frames is not None:
        fps, frames = args.fps, args.frames
    else:
        fps, frames = probe_video(args.video, args.fps, args.frames)
    print(f'fps: {fps:.6g} ({"given" if args.fps else "detected"}), '
          f'frames: {frames} ({"given" if args.frames else "detected"})', file=sys.stderr)
    duration = frames / fps
    step = args.step if args.step > 0 else duration  # 1 画面に表示する秒数

    x, y = load_csv(args.csv)
    if args.index_unit == 'frame':
        x = x / fps
    if args.mean and args.median:
        raise SystemExit('--mean and --median are exclusive')
    if args.mean:
        y = smooth_mean(x, y, args.window)
    elif args.median:
        y = smooth_median(x, y, args.window)

    fig, ax, line_red, (ylo, yhi) = make_figure(args, x, y)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # 入力 0: 元動画，入力 1: プロットの rawvideo（stdin）を 1 回の ffmpeg で縮小して縦に結合する
    width = args.width
    filt = (f'[0:v]scale={width}:-2,setsar=1[top];'
            f'[1:v]scale={width}:-2,setsar=1[bottom];'
            f'[top][bottom]vstack=inputs=2:shortest=1[v]')
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-stats',
           '-i', args.video,
           '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
           '-filter_complex', filt, '-map', '[v]']
    if not args.no_audio:
        cmd += ['-map', '0:a?', '-c:a', 'aac']
    cmd += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', str(args.crf), args.output]
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
            line_red.set_data([t, t], [ylo, yhi])
            ax.draw_artist(line_red)
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


def main():
    p = argparse.ArgumentParser(description='動画とプロットを縦に結合し，プロットに現在位置の赤線を引く')
    p.add_argument('--video', '-i', required=True, help='入力動画')
    p.add_argument('--csv', '-c', required=True, help='index,value の CSV')
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
    p.add_argument('--ylabel', default='value', help='y 軸ラベル')
    p.add_argument('--ymin', type=float, default=None)
    p.add_argument('--ymax', type=float, default=None)
    p.add_argument('--invert-y', action='store_true', help='y 軸を上下反転する')
    p.add_argument('--line-width', type=float, default=3, help='赤線の太さ')

    p.add_argument('--width', type=int, default=800, help='出力動画の幅 [px]')
    p.add_argument('--fig-width', type=float, default=15, help='プロットの幅 [inch]')
    p.add_argument('--fig-height', type=float, default=4, help='プロットの高さ [inch]')
    p.add_argument('--dpi', type=int, default=100)
    p.add_argument('--crf', type=int, default=23, help='x264 の画質（小さいほど高画質）')
    p.add_argument('--no-audio', action='store_true', help='音声を含めない')
    run(p.parse_args())


if __name__ == '__main__':
    main()
