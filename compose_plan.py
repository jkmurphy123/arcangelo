#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from jsonschema import Draft202012Validator


FILL = "__FILL__"


def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def make_template(prompt: str, n_movements: int, total_s: int, seed: int) -> Dict[str, Any]:
    """
    Build a strict plan skeleton with __FILL__ placeholders.
    The LLM will fill placeholders but not add/remove keys.
    """

    # Simple even split that the LLM may adjust slightly (but must still sum to total)
    base = total_s // n_movements
    rem = total_s - base * n_movements
    durations = [base + (1 if i < rem else 0) for i in range(n_movements)]

    movements = []
    start = 0
    for i, dur in enumerate(durations, start=1):
        mid = f"m{i}"
        movements.append(
            {
                "id": mid,
                "label": FILL,
                "start_s": start,
                "duration_s": dur,
                "beat_words": [FILL, FILL],
                "energy": {"tension": FILL, "density": FILL, "brightness": FILL},
                "harmony": {
                    "tonal_center": FILL,
                    "scale": FILL,
                    "degree_pool": [FILL, FILL, FILL, FILL],
                    "change_points_pct": [0, FILL],
                },
                "layers": {
                    "drone": {
                        "state": "on",
                        "foreground": FILL,
                        "mix": {"gain_db": FILL, "fade_in_s": FILL, "fade_out_s": FILL, "pan": 0.0},
                        "gen": {
                            "type": "drone",
                            "notes": [FILL, FILL],
                            "hold_s": [FILL, FILL],
                            "voicing": {"voices": FILL, "spread_semitones": FILL},
                            "drift_cents": FILL
                        },
                        "automation": {
                            "gain_db": [[ "0%", FILL ], [ "100%", FILL ]]
                        },
                        "notes_for_humans": FILL
                    },
                    "motif": {
                        "state": FILL,
                        "foreground": FILL,
                        "mix": {"gain_db": FILL, "fade_in_s": FILL, "fade_out_s": FILL, "pan": -0.2},
                        "gen": {
                            "type": "motif_loop",
                            "grid": "1/16",
                            "motif_steps": 8,
                            "pattern_degrees": [FILL, FILL, FILL, FILL, FILL, FILL, FILL, FILL],
                            "octave": FILL,
                            "rate_hz": FILL,
                            "repeat_cycle": {"length": 4, "variant_on": [4]},
                            "variant": {
                                "mode": "substitution",
                                "substitute_steps": [7, 8],
                                "degree_pool": [FILL, FILL, FILL, FILL]
                            },
                            "gate": {"note_len": "1/8", "swing": FILL},
                            "velocity": {"min": FILL, "max": FILL}
                        },
                        "automation": {
                            "gain_db": [[ "0%", FILL ], [ "50%", FILL ], [ "100%", FILL ]]
                        },
                        "notes_for_humans": FILL
                    },
                    "lead": {
                        "state": FILL,
                        "foreground": FILL,
                        "mix": {"gain_db": FILL, "fade_in_s": FILL, "fade_out_s": FILL, "pan": 0.2},
                        "gen": {
                            "type": "wander",
                            "density_notes_per_min": [FILL, FILL],
                            "phrase_s": [FILL, FILL],
                            "range_semitones": FILL,
                            "step_bias": FILL,
                            "tendency": FILL,
                            "rests_prob": FILL,
                            "degree_pool": [FILL, FILL, FILL, FILL]
                        },
                        "automation": {
                            "gain_db": [[ "0%", FILL ], [ "100%", FILL ]]
                        },
                        "notes_for_humans": FILL
                    },
                    "texture": {
                        "state": FILL,
                        "foreground": FILL,
                        "mix": {"gain_db": FILL, "fade_in_s": FILL, "fade_out_s": FILL, "pan": 0.0},
                        "gen": {
                            "type": "texture",
                            "event_rate_per_min": [FILL, FILL],
                            "dur_s": [FILL, FILL],
                            "register": FILL,
                            "degree_pool": [FILL, FILL, FILL]
                        },
                        "automation": {
                            "gain_db": [[ "0%", FILL ], [ "100%", FILL ]]
                        },
                        "notes_for_humans": FILL
                    },
                    "field": {
                        "state": FILL,
                        "foreground": FILL,
                        "mix": {"gain_db": FILL, "fade_in_s": FILL, "fade_out_s": FILL, "pan": 0.0},
                        "gen": {
                            "type": "field",
                            "clips": [FILL, FILL],
                            "start_prob": FILL,
                            "segment_s": [FILL, FILL]
                        },
                        "automation": {
                            "gain_db": [[ "0%", FILL ], [ "100%", FILL ]]
                        },
                        "notes_for_humans": FILL
                    }
                }
            }
        )
        start += dur

    template = {
        "schema": "aimusic.ambient_storyboard.v1",
        "meta": {
            "title": FILL,
            "seed": seed,
            "source_prompt": prompt,
            "created_utc": utc_now_iso()
        },
        "global": {
            "duration_s": total_s,
            "bpm": FILL,
            "meter": "4/4",
            "tonal_center": FILL,
            "scale": FILL,
            "humanize": {"timing_ms": FILL, "velocity_jitter": FILL}
        },
        "layers": [
            {"id": "drone", "role": "drone"},
            {"id": "motif", "role": "motif_loop"},
            {"id": "lead", "role": "lead_wander"},
            {"id": "texture", "role": "texture"},
            {"id": "field", "role": "field_recording"}
        ],
        "movements": movements,
        "render_hints": {
            "foreground_policy": {"max_foreground_layers": 2, "duck_background_db": 6},
            "midi": {"default_quantize": "1/16", "sustain_cc64": True}
        }
    }
    return template


def strip_code_fences(s: str) -> str:
    s = s.strip()
    # Remove ```json ... ``` fences if model adds them
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def ollama_chat(model: str, messages: List[Dict[str, str]], host: str = "http://localhost:11434",
                temperature: float = 0.6, top_p: float = 0.9, seed: Optional[int] = None) -> str:
    url = f"{host}/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p
        }
    }
    if seed is not None:
        payload["options"]["seed"] = seed

    r = requests.post(url, json=payload, timeout=360)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


def validate_plan(plan: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    v = Draft202012Validator(schema)
    errs = []
    for e in sorted(v.iter_errors(plan), key=lambda x: x.path):
        path = "$"
        for p in e.path:
            path += f"[{p!r}]" if isinstance(p, int) else f".{p}"
        errs.append(f"{path}: {e.message}")
    # Extra semantic checks (LLMs need guardrails)
    errs.extend(extra_checks(plan))
    return errs


def extra_checks(plan: Dict[str, Any]) -> List[str]:
    errs: List[str] = []

    total = plan.get("global", {}).get("duration_s")
    movements = plan.get("movements", [])

    # duration consistency
    if isinstance(total, int) and movements:
        calc = sum(int(m.get("duration_s", 0)) for m in movements)
        if calc != total:
            errs.append(f"$.movements: sum(duration_s)={calc} must equal global.duration_s={total}")

        # start_s continuity
        s = 0
        for idx, m in enumerate(movements):
            ms = m.get("start_s")
            if ms != s:
                errs.append(f"$.movements[{idx}].start_s: expected {s}, got {ms}")
            s += int(m.get("duration_s", 0))

    # foreground policy: at most 2 foreground >= 0.6
    for i, m in enumerate(movements):
        layers = m.get("layers", {})
        loud = []
        for lid, cue in layers.items():
            try:
                fg = float(cue.get("foreground", 0))
            except Exception:
                continue
            st = cue.get("state", "off")
            if st != "off" and fg >= 0.6:
                loud.append(lid)
        if len(loud) > 2:
            errs.append(f"$.movements[{i}].layers: too many foreground layers >=0.6: {loud}")

    # motif: ensure repeat_cycle 4th variant if motif on
    for i, m in enumerate(movements):
        cue = m.get("layers", {}).get("motif", {})
        if cue.get("state") in ("on", "auto"):
            gen = cue.get("gen", {})
            rc = gen.get("repeat_cycle", {})
            if rc.get("length") != 4 or rc.get("variant_on") != [4]:
                errs.append(f"$.movements[{i}].layers.motif.gen.repeat_cycle: must be {{length:4, variant_on:[4]}}")

    return errs


def build_system_prompt() -> str:
    return (
        "You are an expert ambient composer generating a STRICT JSON music plan.\n"
        "OUTPUT RULES:\n"
        "- Output VALID JSON ONLY. No markdown. No code fences. No commentary.\n"
        "- Use ONLY the keys provided in the TEMPLATE.\n"
        "- Replace every value that equals \"__FILL__\" with a concrete value.\n"
        "- Do not change numeric types (keep ints as ints where they appear).\n"
        "- Keep degrees between 1 and 7.\n"
        "- Keep velocities 1..127.\n"
        "- Ensure movement start_s values are continuous and duration_s sums to global.duration_s.\n"
        "- Ensure motif repeat_cycle is 3x same + 4th variant: length=4 and variant_on=[4].\n"
    )


def build_user_prompt(prompt: str, n_movements: int, total_s: int, style_tags: str, intensity: float,
                      avoid: str, template_json: Dict[str, Any]) -> str:
    return (
        "COMPOSITION GOAL\n"
        f"- Prompt: {prompt}\n"
        f"- Movements: {n_movements}\n"
        f"- Total duration seconds: {total_s}\n"
        f"- Style tags: {style_tags}\n"
        f"- Intensity (0..1): {intensity}\n"
        f"- Avoid: {avoid}\n\n"
        "MUSICAL GUIDELINES\n"
        "- Arc: calmer opening, more tension/density mid, cooling resolution at end.\n"
        "- Drone anchors harmony; changes are rare.\n"
        "- Motif is a repeating loop; include a 3x same + 4th variant substitution.\n"
        "- Lead is sparse, lyrical, leaves space.\n"
        "- Texture provides atmosphere; field is intermittent environmental sound.\n"
        "- No more than 2 layers at foreground >= 0.6 per movement.\n\n"
        "TEMPLATE (fill only __FILL__ values; do not add/remove keys):\n"
        + json.dumps(template_json, indent=2)
    )


def repair_prompt(errors: List[str], bad_json_text: str, template_json: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Ask the LLM to fix ONLY the invalid fields; keep keys identical to template.
    """
    sys_msg = (
        "You are repairing a JSON music plan to satisfy a schema and constraints.\n"
        "OUTPUT RULES:\n"
        "- Output VALID JSON ONLY.\n"
        "- Do NOT add or remove any keys.\n"
        "- Keep structure identical to the TEMPLATE.\n"
        "- Fix only what is needed to resolve the listed ERRORS.\n"
    )
    usr_msg = (
        "ERRORS:\n" + "\n".join(f"- {e}" for e in errors) + "\n\n"
        "TEMPLATE (structure must match this):\n"
        + json.dumps(template_json, indent=2) + "\n\n"
        "CURRENT_JSON (fix it):\n"
        + bad_json_text
    )
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}]


def main() -> int:
    try:
        ap = argparse.ArgumentParser(description="Compose an ambient storyboard plan JSON using local Ollama.")
        ap.add_argument("--prompt", required=True, help="Short scene description.")
        ap.add_argument("--movements", type=int, default=4, help="Number of movements (3-5 recommended).")
        ap.add_argument("--duration", type=int, default=420, help="Total duration seconds.")
        ap.add_argument("--seed", type=int, default=0, help="Seed for reproducibility (0 = random).")
        ap.add_argument("--model", default="qwen2.5-coder:7b-instruct", help="Ollama model name.")
        ap.add_argument("--ollama_host", default="http://localhost:11434", help="Ollama host URL.")
        ap.add_argument("--schema", default="schemas/ambient_storyboard_v1.schema.json", help="Path to JSON schema.")
        ap.add_argument("--style", default="cinematic_ambient, slow-bloom, desolate, textural", help="Style tags.")
        ap.add_argument("--intensity", type=float, default=0.6, help="0..1 overall intensity.")
        ap.add_argument("--avoid", default="upbeat major-key pop progressions; busy drums; EDM drops", help="Negative constraints.")
        ap.add_argument("--out", default="plan.json", help="Output path.")
        ap.add_argument("--temperature", type=float, default=0.6)
        ap.add_argument("--top_p", type=float, default=0.9)
        ap.add_argument("--max_repairs", type=int, default=2)
        args = ap.parse_args()

        print("compose_plan.py starting...", flush=True)
        print("Using model:", args.model, "host:", args.ollama_host, flush=True)
        print("Schema:", args.schema, "exists:", os.path.exists(args.schema), flush=True)

        n_mov = clamp_int(args.movements, 1, 8)
        total_s = clamp_int(args.duration, 30, 36000)

        seed = args.seed if args.seed != 0 else random.randint(1, 2_000_000_000)
        schema = load_json(args.schema)

        template = make_template(args.prompt, n_mov, total_s, seed)

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(args.prompt, n_mov, total_s, args.style, args.intensity, args.avoid, template)}
        ]

        # First attempt
        content = ollama_chat(
            model=args.model,
            messages=messages,
            host=args.ollama_host,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=seed
        )
        content = strip_code_fences(content)

        # Parse + validate, with automatic repairs
        bad_text = content
        for attempt in range(args.max_repairs + 1):
            try:
                plan = json.loads(content)
            except json.JSONDecodeError as e:
                errs = [f"JSON parse error: {e}"]
                if attempt >= args.max_repairs:
                    print("Failed to parse JSON after repairs.", file=sys.stderr)
                    print(bad_text[:2000], file=sys.stderr)
                    return 2
                repair_msgs = repair_prompt(errs, bad_text, template)
                content = strip_code_fences(
                    ollama_chat(args.model, repair_msgs, host=args.ollama_host, temperature=0.2, top_p=0.9, seed=seed)
                )
                bad_text = content
                continue

            errs = validate_plan(plan, schema)
            if not errs:
                save_json(args.out, plan)
                # quick summary
                print(f"Wrote: {args.out}")
                print(f"Title: {plan['meta']['title']}")
                for m in plan["movements"]:
                    fg = []
                    for lid, cue in m["layers"].items():
                        if cue["state"] != "off" and float(cue["foreground"]) >= 0.6:
                            fg.append(lid)
                    print(f"- {m['id']} {m['label']} ({m['duration_s']}s) foreground: {fg}")
                return 0

            if attempt >= args.max_repairs:
                print("Validation failed after repairs:", file=sys.stderr)
                for e in errs[:50]:
                    print(" - " + e, file=sys.stderr)
                print("\nLast JSON (truncated):", file=sys.stderr)
                print(json.dumps(plan, indent=2)[:2000], file=sys.stderr)
                return 3

            repair_msgs = repair_prompt(errs, json.dumps(plan, indent=2), template)
            content = strip_code_fences(
                ollama_chat(args.model, repair_msgs, host=args.ollama_host, temperature=0.2, top_p=0.9, seed=seed)
            )
            bad_text = content

        return 3
    
    except Exception as e:
        print("FATAL:", repr(e), file=sys.stderr, flush=True)
        raise

if __name__ == "__main__":
    raise SystemExit(main())
