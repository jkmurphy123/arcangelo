#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mido


LAYER_IDS = ["drone", "motif", "lead", "texture", "field"]


# ----------------------------
# Utilities
# ----------------------------

NOTE_TO_PC = {
    "C": 0, "C#": 1, "DB": 1,
    "D": 2, "D#": 3, "EB": 3,
    "E": 4,
    "F": 5, "F#": 6, "GB": 6,
    "G": 7, "G#": 8, "AB": 8,
    "A": 9, "A#": 10, "BB": 10,
    "B": 11,
}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def db_to_linear(db: float) -> float:
    # linear gain multiplier
    return 10 ** (db / 20.0)


def bpm_to_tempo_us(bpm: float) -> int:
    return mido.bpm2tempo(bpm)


def sec_to_ticks(sec: float, bpm: float, ppq: int) -> int:
    # ticks = seconds * (beats/sec) * ppq
    beats_per_sec = bpm / 60.0
    return int(round(sec * beats_per_sec * ppq))


def parse_time_signature(ts: str) -> Tuple[int, int]:
    if not ts or "/" not in ts:
        return (4, 4)
    a, b = ts.split("/", 1)
    try:
        return (int(a.strip()), int(b.strip()))
    except Exception:
        return (4, 4)


def parse_key_signature(s: str) -> str:
    # Your JSON: "C Major"
    # MIDI meta key_signature expects like "C" or "Cm" etc. mido accepts "C" "Am" etc.
    if not s:
        return "C"
    parts = s.strip().split()
    if not parts:
        return "C"
    tonic = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else "major"
    if mode.startswith("min"):
        return tonic + "m"
    return tonic


def tonic_pc_from_key_sig(key_sig: str) -> int:
    # key_sig may be "C", "Cm", "F#m" etc.
    ks = key_sig.strip()
    minor = ks.endswith("m")
    tonic = ks[:-1] if minor else ks
    tonic = tonic.upper()
    tonic = tonic.replace("♭", "B").replace("♯", "#")
    return NOTE_TO_PC.get(tonic, 0)


def midi_note(tonic: str, octave: int) -> int:
    # MIDI: C-1 = 0, C4 = 60
    pc = NOTE_TO_PC.get(tonic.upper(), 0)
    return (octave + 1) * 12 + pc


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def pick_range(rng: random.Random, a: Any, default_lo: float, default_hi: float) -> Tuple[float, float]:
    if isinstance(a, list) and len(a) == 2:
        try:
            return float(a[0]), float(a[1])
        except Exception:
            return default_lo, default_hi
    return default_lo, default_hi


def eval_piecewise(points: List[List[float]], t: float) -> float:
    """
    points: [[t0, v0], [t1, v1], ...] in seconds (absolute), linear interpolate.
    If t before first -> v0, after last -> v_last.
    """
    if not points:
        return 0.0
    pts = []
    for p in points:
        if isinstance(p, list) and len(p) == 2:
            try:
                pts.append((float(p[0]), float(p[1])))
            except Exception:
                pass
    if not pts:
        return 0.0
    pts.sort(key=lambda x: x[0])

    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]

    for i in range(len(pts) - 1):
        t0, v0 = pts[i]
        t1, v1 = pts[i + 1]
        if t0 <= t <= t1:
            if t1 == t0:
                return v1
            a = (t - t0) / (t1 - t0)
            return v0 + a * (v1 - v0)
    return pts[-1][1]


def grid_to_beats(grid: str) -> float:
    # "1/8" -> 0.5 beats in 4/4 where quarter = 1 beat.
    # beats = 4 / denom for "1/denom" if numerator is 1.
    if not grid or "/" not in grid:
        return 0.5
    num_s, den_s = grid.split("/", 1)
    try:
        num = int(num_s.strip())
        den = int(den_s.strip())
        if num <= 0 or den <= 0:
            return 0.5
        # whole note = 4 beats; 1/den is (4/den) beats if num=1
        return (4.0 * num) / den
    except Exception:
        return 0.5


# ----------------------------
# Normalized structures
# ----------------------------

@dataclass
class NoteEvent:
    t_on: float
    t_off: float
    pitch: int
    vel: int
    channel: int


@dataclass
class Movement:
    id: str
    label: str
    start_s: float
    duration_s: float
    tonal_center: str
    scale: str
    degree_pool: List[int]          # usually semitone offsets like [0,2,4,5,7,9,11]
    layers: Dict[str, Dict[str, Any]]


# ----------------------------
# Plan normalization
# ----------------------------

def normalize_foreground(v: Any) -> float:
    # Your JSON uses boolean. Map to something useful.
    if isinstance(v, bool):
        return 0.75 if v else 0.35
    try:
        return clamp(float(v), 0.0, 1.0)
    except Exception:
        return 0.35


def normalize_mix(mix: Dict[str, Any]) -> Dict[str, float]:
    gain_db = float(mix.get("gain_db", -18)) if isinstance(mix, dict) else -18.0
    fade_in_s = float(mix.get("fade_in_s", 0)) if isinstance(mix, dict) else 0.0
    fade_out_s = float(mix.get("fade_out_s", 0)) if isinstance(mix, dict) else 0.0
    pan = float(mix.get("pan", 0.0)) if isinstance(mix, dict) else 0.0
    return {
        "gain_db": gain_db,
        "fade_in_s": clamp(fade_in_s, 0.0, 60.0),
        "fade_out_s": clamp(fade_out_s, 0.0, 60.0),
        "pan": clamp(pan, -1.0, 1.0)
    }


def normalize_plan(plan: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Movement], List[str]]:
    warnings: List[str] = []

    bpm = float(plan.get("tempo_bpm", 60))
    length_s = float(plan.get("length_s", 240))
    key_sig = parse_key_signature(plan.get("key_signature", "C Major"))
    tonic_pc = tonic_pc_from_key_sig(key_sig)
    time_sig = plan.get("time_signature", "4/4")

    events = plan.get("events", [])
    if not isinstance(events, list):
        raise ValueError("plan.events must be a list")

    movements: List[Movement] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        harm = ev.get("harmony", {}) if isinstance(ev.get("harmony", {}), dict) else {}
        tonal_center = str(harm.get("tonal_center", "C"))
        scale = str(harm.get("scale", "major"))
        degree_pool = harm.get("degree_pool", [0, 2, 4, 5, 7, 9, 11])
        if not (isinstance(degree_pool, list) and all(isinstance(x, (int, float)) for x in degree_pool)):
            degree_pool = [0, 2, 4, 5, 7, 9, 11]
        degree_pool_int = [int(round(float(x))) for x in degree_pool]

        layers = ev.get("layers", {})
        if not isinstance(layers, dict):
            layers = {}

        # normalize each layer
        norm_layers: Dict[str, Dict[str, Any]] = {}
        for lid in LAYER_IDS:
            raw = layers.get(lid, {})
            if not isinstance(raw, dict):
                raw = {}
            state = raw.get("state", "off")
            fg = normalize_foreground(raw.get("foreground", False))
            mix = normalize_mix(raw.get("mix", {}) if isinstance(raw.get("mix", {}), dict) else {})
            gen = raw.get("gen", {}) if isinstance(raw.get("gen", {}), dict) else {}
            automation = raw.get("automation", {}) if isinstance(raw.get("automation", {}), dict) else {}

            # clamp weird pans from LLM (-2..2)
            if abs(mix["pan"]) > 1.0:
                warnings.append(f"{ev.get('id','?')}.{lid}.mix.pan out of range; clamped")

            norm_layers[lid] = {
                "state": state,
                "fg": fg,
                "mix": mix,
                "gen": gen,
                "automation": automation
            }

        m = Movement(
            id=str(ev.get("id", "")),
            label=str(ev.get("label", "")),
            start_s=float(ev.get("start_s", 0)),
            duration_s=float(ev.get("duration_s", 0)),
            tonal_center=tonal_center,
            scale=scale,
            degree_pool=degree_pool_int,
            layers=norm_layers
        )
        movements.append(m)

    # Basic continuity warnings
    movements.sort(key=lambda m: m.start_s)
    for i in range(len(movements) - 1):
        end_i = movements[i].start_s + movements[i].duration_s
        if movements[i + 1].start_s < end_i - 1e-6:
            warnings.append(f"movements overlap: {movements[i].id} and {movements[i+1].id}")
    # ensure last end within length
    if movements:
        end_last = movements[-1].start_s + movements[-1].duration_s
        if end_last > length_s + 1e-6:
            warnings.append("sum of movements exceeds plan.length_s; performer will still render all movements")

    meta = {
        "bpm": bpm,
        "length_s": length_s,
        "key_sig": key_sig,
        "time_sig": time_sig,
        "tonic_pc": tonic_pc
    }
    return meta, movements, warnings


# ----------------------------
# Generators
# ----------------------------

def gain_at(layer: Dict[str, Any], t_abs: float, movement_start: float) -> float:
    """
    layer.automation.gain_db in your JSON appears as [[t_seconds, gain_db], ...]
    where t_seconds is ABSOLUTE movement-local or absolute? In your sample it’s [0..duration].
    We treat it as movement-local seconds.
    """
    auto = layer.get("automation", {})
    pts = auto.get("gain_db", []) if isinstance(auto, dict) else []
    if isinstance(pts, list) and pts:
        v = eval_piecewise(pts, t_abs - movement_start)
    else:
        v = float(layer["mix"]["gain_db"])
    return float(v)


def velocity_from(layer: Dict[str, Any], t_abs: float, m_start: float, base_min: int, base_max: int) -> int:
    fg = float(layer.get("fg", 0.35))
    gdb = gain_at(layer, t_abs, m_start)
    gain = db_to_linear(gdb)

    # Foreground as a gentle multiplier (0.35..0.75 typical)
    fg_mult = 0.6 + 0.8 * fg  # 0.6..1.4

    vmin = int(round(base_min * fg_mult))
    vmax = int(round(base_max * fg_mult))

    # Apply gain to both ends
    vmin = int(round(vmin * gain))
    vmax = int(round(vmax * gain))

    vmin = max(1, min(126, vmin))
    vmax = max(vmin + 1, min(127, vmax))

    return random.randint(vmin, vmax)


def build_motif_pitch_resolver(m: Movement, motif_gen: Dict[str, Any]) -> Tuple[int, List[int]]:
    """
    Interpret your motif pattern_degrees as INDEXES into movement.degree_pool when plausible.

    Your sample:
      harmony.degree_pool = [0,2,4,5,7,9,11]
      pattern_degrees = [0,2,4,6]
    Treat as indexes => semitone offsets [0,4,7,11] => C E G B

    Also handle octave mapping:
      LLM octave=3 but notes_for_humans show C5-ish.
      We'll map midi_octave = octave + 2 (so 3 -> 5).
    """
    tonal = m.tonal_center.strip().upper().replace("♭", "B").replace("♯", "#")
    tonic_pc = NOTE_TO_PC.get(tonal, 0)

    octave_raw = motif_gen.get("octave", 4)
    try:
        octave_raw = int(octave_raw)
    except Exception:
        octave_raw = 4
    midi_octave = octave_raw + 2  # heuristic for your current LLM output

    tonic_pitch = (midi_octave + 1) * 12 + tonic_pc

    dp = m.degree_pool[:] if m.degree_pool else [0, 2, 4, 5, 7, 9, 11]
    # ensure ints
    dp = [int(x) for x in dp]

    return tonic_pitch, dp


def gen_drone(rng: random.Random, meta: Dict[str, Any], m: Movement, layer: Dict[str, Any], channel: int) -> List[NoteEvent]:
    gen = layer.get("gen", {})
    notes = gen.get("notes", [])
    if not (isinstance(notes, list) and notes and all(isinstance(n, (int, float)) for n in notes)):
        return []
    notes = [int(round(float(n))) for n in notes]

    hold_lo, hold_hi = pick_range(rng, gen.get("hold_s", [20, 40]), 20.0, 60.0)
    voices = int(safe_get(gen, ["voicing", "voices"], 1) or 1)
    spread = int(safe_get(gen, ["voicing", "spread_semitones"], 0) or 0)

    start = m.start_s
    end = m.start_s + m.duration_s
    t = start

    out: List[NoteEvent] = []

    # Build voiced pitches
    base_pitches = notes[:]
    while len(base_pitches) < voices:
        # duplicate and spread if requested
        src = base_pitches[len(base_pitches) % len(notes)]
        base_pitches.append(src + spread)

    while t < end - 1e-6:
        hold = rng.uniform(hold_lo, hold_hi)
        t_off = min(t + hold, end)
        for p in base_pitches[:voices]:
            vel = velocity_from(layer, t, m.start_s, base_min=20, base_max=60)
            out.append(NoteEvent(t_on=t, t_off=t_off, pitch=int(clamp(p, 0, 127)), vel=vel, channel=channel))
        t = t_off

    return out


def gen_motif(rng: random.Random, meta: Dict[str, Any], m: Movement, layer: Dict[str, Any], channel: int,
             enforce_cycle_variant: bool, default_variant_steps: List[int]) -> List[NoteEvent]:
    gen = layer.get("gen", {})
    grid = str(gen.get("grid", "1/16"))
    step_beats = grid_to_beats(grid)
    bpm = float(meta["bpm"])
    step_s = step_beats * (60.0 / bpm)

    motif_steps = int(gen.get("motif_steps", 8) or 8)
    pattern = gen.get("pattern_degrees", [])
    if not (isinstance(pattern, list) and len(pattern) >= 1):
        return []
    pattern = [int(round(float(x))) for x in pattern][:motif_steps]
    if len(pattern) < motif_steps:
        # pad with last
        pattern = pattern + [pattern[-1]] * (motif_steps - len(pattern))

    # interpret repeat_cycle
    rc = gen.get("repeat_cycle", {}) if isinstance(gen.get("repeat_cycle", {}), dict) else {}
    cycle_len = int(rc.get("length", 4) or 4)
    variant_on = rc.get("variant_on", [])
    if not (isinstance(variant_on, list) and variant_on):
        variant_on = []
    variant_on = [int(x) for x in variant_on if isinstance(x, (int, float))]

    var = gen.get("variant", {}) if isinstance(gen.get("variant", {}), dict) else {}
    var_mode = str(var.get("mode", "none"))
    substitute_steps = var.get("substitute_steps", [])
    if not (isinstance(substitute_steps, list) and substitute_steps):
        substitute_steps = []
    substitute_steps = [int(x) for x in substitute_steps if isinstance(x, (int, float))]

    # Gate
    gate = gen.get("gate", {}) if isinstance(gen.get("gate", {}), dict) else {}
    note_len_str = str(gate.get("note_len", grid))
    note_len_beats = grid_to_beats(note_len_str)
    note_len_s = note_len_beats * (60.0 / bpm)
    swing = float(gate.get("swing", 0.0) or 0.0)

    tonic_pitch, degree_pool = build_motif_pitch_resolver(m, gen)

    start = m.start_s
    end = m.start_s + m.duration_s

    # Determine if pattern values look like indexes into degree_pool
    # Heuristic: all 0 <= x < len(degree_pool)
    looks_indexed = all(0 <= x < len(degree_pool) for x in pattern)

    # Variant defaults: enforce AAA'B if requested even if LLM omitted
    apply_default_variant = enforce_cycle_variant and (variant_on == [] or var_mode == "none")

    out: List[NoteEvent] = []
    t = start
    rep = 1  # 1-based repetition count

    # Precompute alternate pool for default variant
    alt_indexes = list(range(len(degree_pool))) if looks_indexed else pattern[:]

    while t < end - 1e-6:
        # determine if this repetition is variant
        is_variant = False
        if apply_default_variant:
            # every 4th repetition
            is_variant = (rep % 4 == 0)
        else:
            # honor plan
            is_variant = (rep in variant_on)

        for step_i in range(motif_steps):
            t_step = t + step_i * step_s
            if t_step >= end:
                break

            idx = pattern[step_i]

            if is_variant:
                # Plan-specified variant
                if var_mode == "substitution" and substitute_steps:
                    if (step_i + 1) in substitute_steps:
                        pool = var.get("degree_pool", [])
                        if isinstance(pool, list) and pool:
                            # pool may be indexes or offsets; prefer index style if plausible
                            cand = [int(round(float(x))) for x in pool if isinstance(x, (int, float))]
                            if cand:
                                idx = rng.choice(cand)
                # Default variant: substitute specified steps (default last step)
                if apply_default_variant and (step_i + 1) in default_variant_steps:
                    # pick a different index if possible
                    choices = [x for x in alt_indexes if x != idx]
                    if choices:
                        idx = rng.choice(choices)

            # Convert idx to pitch
            if looks_indexed:
                semioff = degree_pool[idx % len(degree_pool)]
                pitch = tonic_pitch + semioff
            else:
                # treat pattern values as semitone offsets
                pitch = tonic_pitch + idx

            # Swing: delay off-beats slightly
            swing_offset = 0.0
            if swing != 0.0:
                if (step_i % 2) == 1:
                    swing_offset = swing * step_s

            t_on = t_step + swing_offset
            t_off = min(t_on + note_len_s, end)

            # velocity range from gen.velocity if present
            vel_block = gen.get("velocity", {})
            if isinstance(vel_block, dict) and "min" in vel_block and "max" in vel_block:
                base_min = int(vel_block.get("min", 40))
                base_max = int(vel_block.get("max", 80))
                base_min = int(clamp(base_min, 1, 126))
                base_max = int(clamp(base_max, base_min + 1, 127))
                # apply gain/fg
                vel = velocity_from(layer, t_on, m.start_s, base_min=base_min, base_max=base_max)
            else:
                vel = velocity_from(layer, t_on, m.start_s, base_min=30, base_max=75)

            out.append(NoteEvent(t_on=t_on, t_off=t_off, pitch=int(clamp(pitch, 0, 127)), vel=vel, channel=channel))

        # next repetition
        t += motif_steps * step_s
        rep += 1

    return out


def weighted_choice(rng: random.Random, items: List[int], weights: List[float]) -> int:
    s = sum(weights)
    if s <= 0:
        return rng.choice(items)
    r = rng.random() * s
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if r <= acc:
            return it
    return items[-1]


def gen_lead(rng: random.Random, meta: Dict[str, Any], m: Movement, layer: Dict[str, Any], channel: int) -> List[NoteEvent]:
    gen = layer.get("gen", {})
    dp = gen.get("degree_pool", [])
    if not (isinstance(dp, list) and dp and all(isinstance(x, (int, float)) for x in dp)):
        return []

    pool = [int(round(float(x))) for x in dp]
    # If pool looks like semitone offsets, lift to a tonic
    if all(0 <= x <= 11 for x in pool):
        tonic = m.tonal_center.strip().upper().replace("♭", "B").replace("♯", "#")
        tonic_pc = NOTE_TO_PC.get(tonic, 0)
        base = (6 + 1) * 12 + tonic_pc  # default C6-ish
        pool = [base + x for x in pool]

    dens_lo, dens_hi = pick_range(rng, gen.get("density_notes_per_min", [6, 18]), 6, 18)
    # Your LLM may emit 120..300; clamp to something playable unless you want chaos
    dens_lo = clamp(dens_lo, 1, 60)
    dens_hi = clamp(dens_hi, dens_lo, 80)

    phr_lo, phr_hi = pick_range(rng, gen.get("phrase_s", [6, 14]), 6, 14)
    phr_lo = clamp(phr_lo, 1, 60)
    phr_hi = clamp(phr_hi, phr_lo, 120)

    range_semi = int(gen.get("range_semitones", 12) or 12)
    range_semi = int(clamp(range_semi, 1, 36))

    step_bias = float(gen.get("step_bias", 0.7) or 0.7)
    step_bias = clamp(step_bias, 0.0, 1.0)

    tendency = str(gen.get("tendency", "none"))
    rests_prob = float(gen.get("rests_prob", 0.3) or 0.3)
    rests_prob = clamp(rests_prob, 0.0, 0.95)

    start = m.start_s
    end = m.start_s + m.duration_s

    out: List[NoteEvent] = []

    # Determine a center pitch and confine to range_semi around it
    center = int(sorted(pool)[len(pool) // 2])
    min_p = center - range_semi
    max_p = center + range_semi

    def nearest_in_pool(p: int) -> int:
        # pick nearest pitch in pool (or clamp)
        cand = [x for x in pool if min_p <= x <= max_p]
        if not cand:
            cand = pool[:]
        return min(cand, key=lambda x: abs(x - p))

    t = start
    last_pitch = nearest_in_pool(center)

    while t < end - 1e-6:
        phrase = rng.uniform(phr_lo, phr_hi)
        phrase_end = min(t + phrase, end)

        dens = rng.uniform(dens_lo, dens_hi)  # notes per minute
        expected_notes = dens * (phrase / 60.0)
        n_notes = max(1, int(round(expected_notes)))

        if n_notes <= 0:
            t = phrase_end
            continue

        # schedule onsets roughly evenly with jitter
        for i in range(n_notes):
            if rng.random() < rests_prob:
                continue
            a = (i + rng.random() * 0.6) / max(1, n_notes)
            t_on = t + a * phrase
            if t_on >= phrase_end:
                continue

            # random walk: prefer small steps
            step_options = [-7, -5, -3, -2, -1, 1, 2, 3, 5, 7]
            weights = []
            for s in step_options:
                w = 1.0 / (abs(s) + 0.5)
                weights.append(w)
            # step_bias pulls toward smaller steps (increase weight on small steps)
            weights = [w ** (1.0 + 3.0 * step_bias) for w in weights]

            step = weighted_choice(rng, step_options, weights)

            if tendency == "up":
                step = abs(step)
            elif tendency == "down":
                step = -abs(step)

            target = last_pitch + step
            pitch = nearest_in_pool(target)
            last_pitch = pitch

            # duration: short-ish relative to phrase
            dur = rng.uniform(0.15, 0.55) * min(2.0, phrase / 4.0)
            t_off = min(t_on + dur, phrase_end)

            vel = velocity_from(layer, t_on, m.start_s, base_min=35, base_max=95)
            out.append(NoteEvent(t_on=t_on, t_off=t_off, pitch=int(clamp(pitch, 0, 127)), vel=vel, channel=channel))

        t = phrase_end

    return out


def gen_texture(rng: random.Random, meta: Dict[str, Any], m: Movement, layer: Dict[str, Any], channel: int,
                texture_max_rate: float) -> List[NoteEvent]:
    gen = layer.get("gen", {})
    rate = gen.get("event_rate_per_min", [2, 10])
    rate_lo, rate_hi = pick_range(rng, rate, 2, 10)
    rate_lo = clamp(rate_lo, 0.0, texture_max_rate)
    rate_hi = clamp(rate_hi, rate_lo, texture_max_rate)

    dur_lo, dur_hi = pick_range(rng, gen.get("dur_s", [2, 10]), 2, 10)
    dur_lo = clamp(dur_lo, 0.05, 60)
    dur_hi = clamp(dur_hi, dur_lo, 60)

    dp = gen.get("degree_pool", [])
    if not (isinstance(dp, list) and dp and all(isinstance(x, (int, float)) for x in dp)):
        return []
    pool = [int(round(float(x))) for x in dp]

    start = m.start_s
    end = m.start_s + m.duration_s

    out: List[NoteEvent] = []

    # Poisson-ish scheduling: repeatedly sample inter-arrival from exponential distribution
    t = start
    while t < end - 1e-6:
        lam = rng.uniform(rate_lo, rate_hi) / 60.0  # events per second
        if lam <= 1e-6:
            break
        dt = rng.expovariate(lam)
        t_on = t + dt
        if t_on >= end:
            break
        pitch = rng.choice(pool)
        dur = rng.uniform(dur_lo, dur_hi)
        t_off = min(t_on + dur, end)

        vel = velocity_from(layer, t_on, m.start_s, base_min=12, base_max=55)
        out.append(NoteEvent(t_on=t_on, t_off=t_off, pitch=int(clamp(pitch, 0, 127)), vel=vel, channel=channel))
        t = t_on

    return out


def gen_field_cues(rng: random.Random, m: Movement, layer: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Field is not MIDI by nature. We'll emit a cue timeline if it is on.
    """
    gen = layer.get("gen", {})
    clips = gen.get("clips", [])
    if not (isinstance(clips, list) and clips):
        return []
    clips = [str(c) for c in clips]

    start_prob = float(gen.get("start_prob", 0.2) or 0.2)
    start_prob = clamp(start_prob, 0.0, 1.0)

    seg_lo, seg_hi = pick_range(rng, gen.get("segment_s", [10, 25]), 10, 25)
    seg_lo = clamp(seg_lo, 0.5, 180)
    seg_hi = clamp(seg_hi, seg_lo, 300)

    cues: List[Dict[str, Any]] = []
    t = m.start_s
    end = m.start_s + m.duration_s
    while t < end - 1e-6:
        # every ~10 seconds decide whether to spawn a cue
        window = 10.0
        if rng.random() < start_prob:
            clip = rng.choice(clips)
            dur = rng.uniform(seg_lo, seg_hi)
            cues.append({"t": round(t, 3), "clip": clip, "dur": round(min(dur, end - t), 3)})
            t += dur  # avoid stacking too much
        else:
            t += window
    return cues


# ----------------------------
# MIDI writing
# ----------------------------

def write_track_midi(note_events: List[NoteEvent], bpm: float, ppq: int, channel: int,
                     title: str, time_sig: str, key_sig: str) -> mido.MidiFile:
    mid = mido.MidiFile(ticks_per_beat=ppq)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)

    # Meta
    tr.append(mido.MetaMessage("track_name", name=title, time=0))
    tr.append(mido.MetaMessage("set_tempo", tempo=bpm_to_tempo_us(bpm), time=0))
    nn, dd = parse_time_signature(time_sig)
    tr.append(mido.MetaMessage("time_signature", numerator=nn, denominator=dd, time=0))
    tr.append(mido.MetaMessage("key_signature", key=key_sig, time=0))

    # Sort messages by absolute tick
    msgs: List[Tuple[int, mido.Message]] = []

    for e in note_events:
        on = sec_to_ticks(e.t_on, bpm, ppq)
        off = sec_to_ticks(e.t_off, bpm, ppq)
        on = max(0, on)
        off = max(on + 1, off)  # ensure > on
        pitch = int(clamp(e.pitch, 0, 127))
        vel = int(clamp(e.vel, 1, 127))

        msgs.append((on, mido.Message("note_on", note=pitch, velocity=vel, channel=channel)))
        msgs.append((off, mido.Message("note_off", note=pitch, velocity=0, channel=channel)))

    msgs.sort(key=lambda x: (x[0], 0 if x[1].type == "note_off" else 1))

    # Convert to delta times
    last = 0
    for tick, msg in msgs:
        dt = tick - last
        if dt < 0:
            dt = 0
        msg.time = dt
        tr.append(msg)
        last = tick

    return mid


def write_multitrack_midi(events_by_layer: Dict[str, List[NoteEvent]], bpm: float, ppq: int,
                          channels: Dict[str, int], time_sig: str, key_sig: str) -> mido.MidiFile:
    mid = mido.MidiFile(ticks_per_beat=ppq)

    # Put global meta in first track
    meta_tr = mido.MidiTrack()
    mid.tracks.append(meta_tr)
    meta_tr.append(mido.MetaMessage("track_name", name="META", time=0))
    meta_tr.append(mido.MetaMessage("set_tempo", tempo=bpm_to_tempo_us(bpm), time=0))
    nn, dd = parse_time_signature(time_sig)
    meta_tr.append(mido.MetaMessage("time_signature", numerator=nn, denominator=dd, time=0))
    meta_tr.append(mido.MetaMessage("key_signature", key=key_sig, time=0))

    for lid in LAYER_IDS:
        tr = mido.MidiTrack()
        mid.tracks.append(tr)
        ch = channels.get(lid, 0)
        tr.append(mido.MetaMessage("track_name", name=lid.upper(), time=0))

        msgs: List[Tuple[int, mido.Message]] = []
        for e in events_by_layer.get(lid, []):
            on = sec_to_ticks(e.t_on, bpm, ppq)
            off = sec_to_ticks(e.t_off, bpm, ppq)
            on = max(0, on)
            off = max(on + 1, off)
            pitch = int(clamp(e.pitch, 0, 127))
            vel = int(clamp(e.vel, 1, 127))

            msgs.append((on, mido.Message("note_on", note=pitch, velocity=vel, channel=ch)))
            msgs.append((off, mido.Message("note_off", note=pitch, velocity=0, channel=ch)))

        msgs.sort(key=lambda x: (x[0], 0 if x[1].type == "note_off" else 1))

        last = 0
        for tick, msg in msgs:
            dt = tick - last
            if dt < 0:
                dt = 0
            msg.time = dt
            tr.append(msg)
            last = tick

    return mid


# ----------------------------
# Main performer
# ----------------------------

def perform(plan: Dict[str, Any],
            ppq: int,
            channels: Dict[str, int],
            seed: int,
            enforce_cycle_variant: bool,
            default_variant_steps: List[int],
            texture_max_rate: float,
            debug_dir: Optional[str]) -> Tuple[mido.MidiFile, Dict[str, List[NoteEvent]], Dict[str, Any], List[str]]:
    rng = random.Random(seed)

    meta, movements, warnings = normalize_plan(plan)
    bpm = float(meta["bpm"])
    time_sig = str(meta["time_sig"])
    key_sig = meta["key_sig"]

    events_by_layer: Dict[str, List[NoteEvent]] = {lid: [] for lid in LAYER_IDS}
    field_cues: List[Dict[str, Any]] = []

    for m in movements:
        for lid in LAYER_IDS:
            layer = m.layers[lid]
            state = str(layer.get("state", "off"))
            if state == "off":
                continue
            # treat "auto" as on if fg is noticeable
            if state == "auto" and float(layer.get("fg", 0.0)) < 0.15:
                continue

            gen = layer.get("gen", {})
            gtype = str(gen.get("type", "")).strip().lower()

            ch = channels.get(lid, 0)

            if lid == "drone" or gtype == "drone":
                events_by_layer[lid].extend(gen_drone(rng, meta, m, layer, ch))
            elif lid == "motif" or gtype == "motif_loop":
                events_by_layer[lid].extend(
                    gen_motif(rng, meta, m, layer, ch, enforce_cycle_variant=enforce_cycle_variant,
                              default_variant_steps=default_variant_steps)
                )
            elif lid == "lead" or gtype == "wander":
                events_by_layer[lid].extend(gen_lead(rng, meta, m, layer, ch))
            elif lid == "texture" or gtype == "texture":
                events_by_layer[lid].extend(gen_texture(rng, meta, m, layer, ch, texture_max_rate=texture_max_rate))
            elif lid == "field" or gtype == "field":
                # not MIDI; cues only
                field_cues.extend(gen_field_cues(rng, m, layer))
            else:
                warnings.append(f"{m.id}.{lid}: unknown gen.type '{gtype}', skipping")

    # Debug dumps
    aux: Dict[str, Any] = {"field_cues": field_cues, "meta": meta}
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        for lid, evs in events_by_layer.items():
            path = os.path.join(debug_dir, f"{lid}_note_events.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([e.__dict__ for e in evs], f, indent=2)
        if field_cues:
            with open(os.path.join(debug_dir, "field_cues.json"), "w", encoding="utf-8") as f:
                json.dump(field_cues, f, indent=2)
        with open(os.path.join(debug_dir, "warnings.json"), "w", encoding="utf-8") as f:
            json.dump(warnings, f, indent=2)

    mid = write_multitrack_midi(events_by_layer, bpm=bpm, ppq=ppq, channels=channels, time_sig=time_sig, key_sig=key_sig)
    return mid, events_by_layer, aux, warnings


def parse_channels(s: str) -> Dict[str, int]:
    out = {lid: i for i, lid in enumerate(LAYER_IDS)}  # default 0..4
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k in LAYER_IDS:
            try:
                ch = int(v)
                # user likely uses 1..16; mido uses 0..15
                if 1 <= ch <= 16:
                    ch -= 1
                out[k] = int(clamp(ch, 0, 15))
            except Exception:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Perform an ambient plan JSON into MIDI (mido).")
    ap.add_argument("--plan", required=True, help="Path to generated plan JSON (LLM output).")
    ap.add_argument("--out", required=True, help="Output MIDI path (multi-track).")
    ap.add_argument("--separate", action="store_true", help="Also write separate MIDIs per layer.")
    ap.add_argument("--ppq", type=int, default=480, help="Ticks per beat (PPQ).")
    ap.add_argument("--channels", default="drone=1,motif=2,lead=3,texture=4,field=5",
                    help="Layer->MIDI channel mapping (1-16 or 0-15).")
    ap.add_argument("--seed", type=int, default=0, help="Seed for determinism (0 uses plan.seed if present, else random).")
    ap.add_argument("--enforce_cycle_variant", action="store_true",
                    help="Enforce 3x same + 4th variant motif if plan omits it.")
    ap.add_argument("--variant_steps", default="4", help="1-based motif step numbers to substitute on variant reps (e.g. '4' or '7,8').")
    ap.add_argument("--texture_max_rate", type=float, default=20.0, help="Cap texture events/min to avoid confetti.")
    ap.add_argument("--debug", default="", help="Debug output directory (event dumps, warnings).")
    args = ap.parse_args()

    with open(args.plan, "r", encoding="utf-8") as f:
        plan = json.load(f)

    channels = parse_channels(args.channels)

    # seed precedence: CLI -> plan.meta.seed -> random
    seed = args.seed
    if seed == 0:
        # your plan has no meta.seed; keep fallback behavior
        seed = int(plan.get("seed", 0) or 0)
    if seed == 0:
        seed = random.randint(1, 2_000_000_000)

    steps = []
    for part in args.variant_steps.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            steps.append(int(part))
        except Exception:
            pass
    if not steps:
        steps = [4]

    debug_dir = args.debug.strip() or None

    mid, events_by_layer, aux, warnings = perform(
        plan=plan,
        ppq=args.ppq,
        channels=channels,
        seed=seed,
        enforce_cycle_variant=args.enforce_cycle_variant,
        default_variant_steps=steps,
        texture_max_rate=args.texture_max_rate,
        debug_dir=debug_dir
    )

    # Write main MIDI
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    mid.save(out_path)

    # Optionally separate
    if args.separate:
        base, ext = os.path.splitext(out_path)
        bpm = float(plan.get("tempo_bpm", 60))
        time_sig = str(plan.get("time_signature", "4/4"))
        key_sig = parse_key_signature(plan.get("key_signature", "C Major"))
        for lid in LAYER_IDS:
            # Skip field layer MIDI unless it has actual note events (usually none)
            if lid == "field" and not events_by_layer.get(lid):
                continue
            ch = channels.get(lid, 0)
            mf = write_track_midi(
                note_events=events_by_layer.get(lid, []),
                bpm=bpm,
                ppq=args.ppq,
                channel=ch,
                title=lid.upper(),
                time_sig=time_sig,
                key_sig=key_sig
            )
            mf.save(f"{base}.{lid}{ext}")

    # Print summary
    print(f"Wrote: {out_path}")
    print(f"Seed: {seed}")
    for lid in LAYER_IDS:
        n = len(events_by_layer.get(lid, []))
        if n:
            print(f"- {lid}: {n} notes")
    if debug_dir:
        print(f"Debug: {debug_dir}")
    if warnings:
        print("Warnings:")
        for w in warnings[:20]:
            print(" - " + w)
        if len(warnings) > 20:
            print(f" - ... ({len(warnings)-20} more)")

    # If field cues exist, mention them
    if debug_dir and aux.get("field_cues"):
        print("Field cues written to debug/field_cues.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
