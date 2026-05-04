"""Procedural ambient piano pad. CC0 (we synthesized it).

Generates a slow chord progression (i-iv-i-v in A minor at low tempo) with
piano-like attack envelopes, harmonic stack, and gentle vibrato. About 3
minutes long. Output: assets/audio/music.wav (then convert to mp3 via ffmpeg).

This is a placeholder. Drop in a different `music.mp3` to swap; the video
build picks it up automatically.

Usage:
    uv run python assets/audio/_synthesize_ambient.py
    ffmpeg -y -i assets/audio/music.wav -b:a 192k assets/audio/music.mp3
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

OUT = Path(__file__).resolve().parent / "music.wav"
SR = 44100
DURATION = 180.0  # 3 min

# A minor: i = A C E ; iv = D F A ; v = E G B ; back to i
# Octaves chosen low-mid for ambient feel. Hz of A2/A3.
CHORDS = [
    (110.0, 130.81, 164.81),   # A2 C3 E3
    (146.83, 174.61, 220.0),   # D3 F3 A3
    (110.0, 130.81, 164.81),
    (164.81, 196.0, 246.94),   # E3 G3 B3
]
CHORD_DUR = DURATION / len(CHORDS)


def _piano_note(freq: float, n_samples: int, sr: int = SR) -> np.ndarray:
    """One sustained piano-like tone with attack/release + harmonic stack."""
    t = np.linspace(0, n_samples / sr, n_samples, endpoint=False)
    # Slow attack, slow release across the whole chord.
    attack_s = min(2.5, n_samples / sr * 0.25)
    release_s = min(4.0, n_samples / sr * 0.4)
    env = np.ones_like(t)
    a = int(attack_s * sr)
    r = int(release_s * sr)
    env[:a] = np.linspace(0, 1, a)
    env[-r:] = np.linspace(1, 0, r)
    # Subtle vibrato (3 Hz, 0.6 % depth) for warmth
    vib = 1.0 + 0.006 * np.sin(2 * np.pi * 3.0 * t)
    base = np.sin(2 * np.pi * freq * vib * t)
    h2 = 0.45 * np.sin(2 * np.pi * 2 * freq * vib * t)
    h3 = 0.20 * np.sin(2 * np.pi * 3 * freq * vib * t)
    h4 = 0.08 * np.sin(2 * np.pi * 4 * freq * vib * t)
    return env * (base + h2 + h3 + h4)


def main() -> None:
    n_total = int(DURATION * SR)
    out = np.zeros(n_total, dtype=np.float32)
    n_chord = int(CHORD_DUR * SR)
    for i, freqs in enumerate(CHORDS):
        seg = np.zeros(n_chord, dtype=np.float32)
        for f in freqs:
            seg += _piano_note(f, n_chord) * 0.18
        # Cross-fade between chords by overlapping by 1.5 s
        crossfade_s = 1.5
        cf = int(crossfade_s * SR)
        start = i * n_chord
        end = start + n_chord
        if i > 0:
            # Fade-in over crossfade region
            seg[:cf] *= np.linspace(0, 1, cf)
        if i < len(CHORDS) - 1:
            seg[-cf:] *= np.linspace(1, 0, cf)
        s = max(0, start)
        e = min(n_total, end)
        out[s:e] += seg[: e - s]

    # Normalise and add light low-frequency rumble to fill the bottom octave
    rumble_t = np.linspace(0, DURATION, n_total)
    rumble = 0.04 * np.sin(2 * np.pi * 55.0 * rumble_t) * np.sin(2 * np.pi * 0.05 * rumble_t)
    out += rumble.astype(np.float32)
    peak = float(np.max(np.abs(out))) or 1.0
    out = out / peak * 0.7

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(OUT), SR, (out * 32767).astype(np.int16))
    print(f"wrote {OUT}  ({DURATION:.0f}s, {SR} Hz, mono)")


if __name__ == "__main__":
    main()
