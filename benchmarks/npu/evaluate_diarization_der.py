#!/usr/bin/env python3
"""Score a diarization JSON result against an RTTM reference with pyannote."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate


def load_rttm(path: Path) -> Annotation:
    annotation = Annotation(uri=path.stem)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0] != "SPEAKER" or len(fields) < 8:
                continue
            start = float(fields[3])
            duration = float(fields[4])
            speaker = fields[7]
            annotation[Segment(start, start + duration)] = speaker
    return annotation


def load_json(path: Path) -> tuple[Annotation, float | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotation = Annotation(uri=path.stem)
    for index, segment in enumerate(payload["segments"]):
        annotation[Segment(float(segment["start"]), float(segment["end"]))] = str(
            segment["speaker"]
        )
    return annotation, payload.get("duration")


def score(
    reference: Annotation,
    hypothesis: Annotation,
    *,
    duration: float,
    collar: float,
    full_uem: bool,
) -> dict:
    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    uem = Timeline([Segment(0.0, duration)]) if full_uem else None
    kwargs = {"detailed": True}
    if uem is not None:
        kwargs["uem"] = uem
    detailed = metric(reference, hypothesis, **kwargs)
    total = detailed["total"]
    if total <= 0:
        raise RuntimeError("No reference speech was scored")
    return {
        "der_pct": round(100 * detailed["diarization error rate"], 4),
        "false_alarm_pct": round(100 * detailed["false alarm"] / total, 4),
        "missed_detection_pct": round(100 * detailed["missed detection"] / total, 4),
        "confusion_pct": round(100 * detailed["confusion"] / total, 4),
        "correct_pct": round(100 * detailed["correct"] / total, 4),
        "false_alarm_s": round(detailed["false alarm"], 3),
        "missed_detection_s": round(detailed["missed detection"], 3),
        "confusion_s": round(detailed["confusion"], 3),
        "correct_s": round(detailed["correct"], 3),
        "total_scored_s": round(total, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="RTTM or hypothesis JSON")
    parser.add_argument("--hypothesis", required=True, help="hypothesis JSON or RTTM")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--collar", type=float, default=0.0)
    parser.add_argument(
        "--speech-only-uem",
        action="store_true",
        help="Use pyannote's reference/hypothesis extents instead of the full file",
    )
    args = parser.parse_args()

    def load(path: str, *, duration: float | None) -> tuple[Annotation, float | None]:
        candidate = Path(path)
        if candidate.suffix.lower() == ".rttm":
            return load_rttm(candidate), duration
        return load_json(candidate)

    reference, ref_duration = load(args.reference, duration=args.duration)
    hypothesis, hyp_duration = load(args.hypothesis, duration=args.duration)
    duration = args.duration or ref_duration or hyp_duration
    if duration is None:
        if args.speech_only_uem:
            duration = 0.0
        else:
            raise SystemExit("Need --duration for JSON input without duration")

    summary = {
        "reference": args.reference,
        "hypothesis": args.hypothesis,
        "duration_s": duration,
        "reference_segments": len(reference),
        "hypothesis_segments": len(hypothesis),
        "reference_speakers": len(reference.labels()),
        "hypothesis_speakers": len(hypothesis.labels()),
        "reference_speech_s": round(reference.get_timeline().duration(), 3),
        "hypothesis_speech_s": round(hypothesis.get_timeline().duration(), 3),
    }
    if duration > 0:
        summary["full_uem"] = score(
            reference,
            hypothesis,
            duration=duration,
            collar=args.collar,
            full_uem=True,
        )
    summary["extent_uem"] = score(
        reference,
        hypothesis,
        duration=duration,
        collar=args.collar,
        full_uem=False,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
