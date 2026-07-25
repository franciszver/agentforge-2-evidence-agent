#!/usr/bin/env python3
"""Build the Clinical Co-Pilot demo mp4 + README hero gif (issue #148).

Reads the 9 already-captured OpenEMR Co-Pilot panel screenshots (see
docs/DEMO_SCRIPT.md's three measured beats), composites each into a
two-pane frame (full dashboard for context + a large, readable crop of the
Co-Pilot panel), overlays a caption bar, muxes a synthesized calm arpeggio
soundtrack, and emits:

  - an mp4 (silent-fallback if audio muxing fails) -- NOT committed, too
    large / regenerable
  - a gif (captioned, same frames) -- committed to docs/assets/demo.gif

Usage:
    python scripts/generate_demo_video.py \
        --frames-dir tmp \
        --out-dir docs/assets \
        --mp4-out docs/assets/demo.mp4

Dependencies: Pillow (already a project dependency elsewhere in this repo)
and imageio-ffmpeg (dev-only, NOT added to the repo's declared
dependencies -- install into your own environment: `pip install
imageio-ffmpeg`). Audio synthesis uses only the Python stdlib (wave, math,
array) -- no numpy.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import subprocess
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
CAPTIONS_FILE = SCRIPT_DIR / "demo_captions.json"

# Original frames are all 1592x853 (OpenEMR Co-Pilot panel, live capture).
# The panel is bottom-right anchored; its bottom/right edges sit flush with
# the page edge in every captured frame, but its top edge varies with
# content length. This crop box was verified against all 9 frames (top
# edges observed between y=357 and y=433) and includes a margin so no
# frame's panel content is clipped.
PANEL_CROP = (1216, 317, 1592, 853)  # left, top, right, bottom

CANVAS_W, CANVAS_H = 1600, 900
CAPTION_H = round(CANVAS_H * 0.12)  # ~12% bottom bar
CONTENT_H = CANVAS_H - CAPTION_H
LEFT_W = 800
RIGHT_W = CANVAS_W - LEFT_W
PADDING = 16

FRAME_HOLD_SECONDS = 3.5
FPS = 2  # static hold per frame; a couple of duplicated frames avoid some
         # players treating a 1-fps mp4 oddly, without bloating file size.

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_within(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / img.width, max_h / img.height)
    new_w = max(1, round(img.width * scale))
    new_h = max(1, round(img.height * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def build_composite(frame_path: Path) -> Image.Image:
    src = Image.open(frame_path).convert("RGB")

    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (245, 246, 248))
    draw = ImageDraw.Draw(canvas)

    # Left pane: downscaled full dashboard, for context ("this is a real
    # EMR"), vertically centered in the content area.
    left_target_w = LEFT_W - 2 * PADDING
    left_target_h = CONTENT_H - 2 * PADDING
    left_img = fit_within(src, left_target_w, left_target_h)
    left_x = (LEFT_W - left_img.width) // 2
    left_y = PADDING + (left_target_h - left_img.height) // 2
    canvas.paste(left_img, (left_x, left_y))

    # Right pane: the Co-Pilot panel crop, enlarged for legibility.
    panel = src.crop(PANEL_CROP)
    right_target_w = RIGHT_W - 2 * PADDING
    right_target_h = CONTENT_H - 2 * PADDING
    right_img = fit_within(panel, right_target_w, right_target_h)
    right_x = LEFT_W + PADDING + (right_target_w - right_img.width) // 2
    right_y = PADDING + (right_target_h - right_img.height) // 2
    canvas.paste(right_img, (right_x, right_y))

    # Divider between panes.
    draw.line([(LEFT_W, 0), (LEFT_W, CONTENT_H)], fill=(210, 212, 216), width=2)

    return canvas


def draw_caption(canvas: Image.Image, text: str) -> Image.Image:
    canvas = canvas.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")

    bar_top = CONTENT_H
    draw.rectangle(
        [(0, bar_top), (CANVAS_W, CANVAS_H)],
        fill=(15, 17, 20, 210),
    )

    max_text_w = CANVAS_W - 80
    size = 30
    font = load_font(size)
    while size > 12:
        font = load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        if text_w <= max_text_w:
            break
        size -= 2

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (CANVAS_W - text_w) // 2 - bbox[0]
    y = bar_top + (CAPTION_H - text_h) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=(240, 241, 245, 255))

    return canvas


def make_frames(frames_dir: Path) -> list[Image.Image]:
    captions = json.loads(CAPTIONS_FILE.read_text(encoding="utf-8"))["frames"]
    out = []
    for entry in captions:
        path = frames_dir / entry["file"]
        composite = build_composite(path)
        captioned = draw_caption(composite, entry["caption"])
        out.append(captioned)
    return out


# --- Audio: calm arpeggio soundtrack, stdlib only -------------------------

SAMPLE_RATE = 44100

# C - G - Am - F progression, arpeggiated.
NOTE_HZ = {
    "C3": 130.81, "E3": 164.81, "G3": 196.00, "C4": 261.63,
    "D3": 146.83, "G2": 98.00, "B3": 246.94,
    "A2": 110.00, "E4": 329.63,
    "F2": 87.31, "A3": 220.00,
}

CHORDS = [
    ["C3", "E3", "G3", "C4"],
    ["G2", "B3", "D3", "G3"],
    ["A2", "C4", "E4", "A3"],
    ["F2", "A3", "C4", "F2"],
]


def synth_note(freq: float, duration: float, volume: float) -> array.array:
    n = int(SAMPLE_RATE * duration)
    attack = max(1, int(n * 0.1))
    decay = max(1, int(n * 0.3))
    samples = array.array("h", [0] * n)
    for i in range(n):
        t = i / SAMPLE_RATE
        env = 1.0
        if i < attack:
            env = i / attack
        elif i > n - decay:
            env = max(0.0, (n - i) / decay)
        val = math.sin(2 * math.pi * freq * t) * volume * env
        # Defensive clamp before 16-bit packing.
        sample = int(max(-32767, min(32767, val * 32767)))
        samples[i] = sample
    return samples


def synth_soundtrack(total_seconds: float) -> array.array:
    total_samples = int(SAMPLE_RATE * total_seconds)
    mix = array.array("h", [0] * total_samples)

    note_dur = 0.6
    volume = 0.12  # low volume, headroom against clipping
    t = 0.0
    chord_i = 0
    while t < total_seconds:
        chord = CHORDS[chord_i % len(CHORDS)]
        for note_name in chord:
            if t >= total_seconds:
                break
            freq = NOTE_HZ[note_name]
            note = synth_note(freq, note_dur, volume)
            start = int(t * SAMPLE_RATE)
            for i, s in enumerate(note):
                idx = start + i
                if idx >= total_samples:
                    break
                mixed = mix[idx] + s
                mix[idx] = max(-32767, min(32767, mixed))
            t += note_dur
        chord_i += 1

    # Fade in/out sized to the video length (~0.8s or 5% of total, whichever
    # is smaller).
    fade_samples = min(int(SAMPLE_RATE * 0.8), total_samples // 20 or 1)
    for i in range(fade_samples):
        factor = i / fade_samples
        mix[i] = int(mix[i] * factor)
        j = total_samples - 1 - i
        mix[j] = int(mix[j] * factor)

    return mix


def write_wav(path: Path, samples: array.array) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())


# --- Video assembly via ffmpeg --------------------------------------------


def get_ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def build_video(frames: list[Image.Image], tmp: Path, ffmpeg: str) -> Path:
    for i, frame in enumerate(frames):
        # Ensure even dimensions (already guaranteed by CANVAS_W/H, but be
        # defensive if that ever changes).
        w, h = frame.size
        if w % 2 or h % 2:
            frame = frame.crop((0, 0, w - (w % 2), h - (h % 2)))
        frame.save(tmp / f"frame_{i:03d}.png")

    silent_mp4 = tmp / "silent.mp4"
    cmd = [
        ffmpeg, "-y",
        "-framerate", f"1/{FRAME_HOLD_SECONDS}",
        "-i", str(tmp / "frame_%03d.png"),
        "-vf", "format=yuv420p",
        "-r", "24",
        str(silent_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return silent_mp4


def mux_audio(silent_mp4: Path, wav_path: Path, final_mp4: Path, ffmpeg: str) -> bool:
    cmd = [
        ffmpeg, "-y",
        "-i", str(silent_mp4),
        "-i", str(wav_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(final_mp4),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and final_mp4.exists()


def build_gif(frames: list[Image.Image], gif_path: Path, max_width: int = 640) -> None:
    resized = [fit_within(f, max_width, CANVAS_H) for f in frames]
    duration_ms = int(FRAME_HOLD_SECONDS * 1000)
    resized[0].save(
        gif_path,
        save_all=True,
        append_images=resized[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, default=Path("tmp"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument(
        "--mp4-out", type=Path, default=Path("docs/assets/demo.mp4")
    )
    parser.add_argument("--gif-max-width", type=int, default=640)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.mp4_out.parent.mkdir(parents=True, exist_ok=True)

    frames = make_frames(args.frames_dir)
    total_seconds = len(frames) * FRAME_HOLD_SECONDS

    gif_path = args.out_dir / "demo.gif"
    build_gif(frames, gif_path, max_width=args.gif_max_width)
    print(f"gif written: {gif_path} ({gif_path.stat().st_size} bytes)")

    with TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        ffmpeg = get_ffmpeg()
        silent_mp4 = build_video(frames, tmp, ffmpeg)

        wav_path = tmp / "soundtrack.wav"
        samples = synth_soundtrack(total_seconds)
        write_wav(wav_path, samples)

        muxed = mux_audio(silent_mp4, wav_path, args.mp4_out, ffmpeg)
        if not muxed:
            print("audio mux failed -- keeping silent mp4")
            silent_mp4.replace(args.mp4_out)
        else:
            print(f"mp4 written (with audio): {args.mp4_out}")
    # TemporaryDirectory cleanup happens here; all handles above are closed
    # by their `with`/subprocess.run before this point.

    print(f"mp4 size: {args.mp4_out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
