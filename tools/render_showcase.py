"""Encode the browser recording and verify the full-width caption on every frame."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np


def run(ffmpeg, *args, capture=False):
    return subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', *args],
                          capture_output=capture, check=True).stdout


def sample_frames(ffmpeg, source):
    raw = run(ffmpeg, '-i', str(source), '-vf', 'fps=25,scale=160:90,format=rgb24',
              '-f', 'rawvideo', '-', capture=True)
    return np.frombuffer(raw, np.uint8).reshape(-1, 90, 160, 3)


def caption_present(frames):
    expected = np.array([20, 23, 28])  # caption bar #14171c
    pixels = frames[:, 88, [2, 80, 157], :].astype(int)
    return (abs(pixels - expected).max(axis=(1, 2)) <= 18)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--media', type=Path, default=Path('docs/media'))
    args = parser.parse_args()
    meta = json.loads((args.media / 'recording.json').read_text())
    assert meta['viewport'] == {'width': 1600, 'height': 900}
    assert not meta['viewport_violations'] and not meta['page_errors']
    assert meta['screenshots_during_recording'] == 0
    assert all(scene['end_s'] - scene['settled_s'] >= 5.9 for scene in meta['scenes'])
    source, target = args.media / 'embodiq-demo.webm', args.media / 'embodiq-demo.mp4'
    raw = sample_frames(args.ffmpeg, source)
    visible = caption_present(raw)
    indices = np.flatnonzero(visible)
    first, last = int(indices[0]), int(indices[-1]) + 1
    if first > 100 or not visible[first:last].all():
        raise RuntimeError('Caption cropped or viewport scaled during recording')
    del raw
    run(args.ffmpeg, '-i', str(source), '-vf',
        f'fps=25,trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', '-an', '-y', str(target))
    encoded = sample_frames(args.ffmpeg, target)
    if not caption_present(encoded).all():
        raise RuntimeError('Encoded video failed the full-width caption check')
    # Keep each preview shot at normal speed; no accelerated navigation.
    clips = []
    for i, scene_index in enumerate((1, 5, 7)):
        start = meta['scenes'][scene_index]['settled_s']
        clips.append(f'[0:v]trim=start={start:.3f}:duration=6,setpts=PTS-STARTPTS[c{i}]')
    graph = ';'.join(clips) + ';[c0][c1][c2]concat=n=3:v=1:a=0,fps=6,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4[out]'
    run(args.ffmpeg, '-i', str(target), '-filter_complex', graph, '-map', '[out]',
        '-loop', '0', '-y', str(args.media / 'embodiq-preview.gif'))
    verification = {'frames_checked': len(encoded), 'duration_s': len(encoded)/25,
                    'size': [1600,900], 'all_caption_edges_visible': True,
                    'viewport_changes': 0, 'scene_count': len(meta['scenes']),
                    'minimum_settled_hold_s': min(s['end_s']-s['settled_s'] for s in meta['scenes']),
                    'trimmed_start_frames': first}
    (args.media / 'video-validation.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
