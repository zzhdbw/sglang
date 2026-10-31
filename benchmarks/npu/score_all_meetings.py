#!/usr/bin/env python3
"""Score all service/CPU diarization JSON files against same-name RTTMs."""

from __future__ import annotations

import argparse
import json
import wave
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
            start, duration = float(fields[3]), float(fields[4])
            annotation[Segment(start, start + duration)] = fields[7]
    return annotation


def load_json(path: Path) -> tuple[Annotation, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotation = Annotation(uri=path.stem)
    for segment in payload["segments"]:
        annotation[Segment(float(segment["start"]), float(segment["end"]))] = str(
            segment["speaker"]
        )
    return annotation, payload


def detailed(
    reference: Annotation,
    hypothesis: Annotation,
    duration: float,
    collar: float,
) -> dict:
    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    result = metric(
        reference,
        hypothesis,
        uem=Timeline([Segment(0.0, duration)]),
        detailed=True,
    )
    total = result["total"]
    return {
        "der": result["diarization error rate"],
        "fa": result["false alarm"],
        "miss": result["missed detection"],
        "conf": result["confusion"],
        "correct": result["correct"],
        "total": total,
    }


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as source:
        return source.getnframes() / source.getframerate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--service-dir", required=True)
    parser.add_argument("--cpu-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    service_dir = Path(args.service_dir)
    cpu_dir = Path(args.cpu_dir)
    rows = []
    totals = {
        name: {"fa": 0.0, "miss": 0.0, "conf": 0.0, "total": 0.0}
        for name in ("service", "cpu")
    }
    totals25 = {
        name: {"fa": 0.0, "miss": 0.0, "conf": 0.0, "total": 0.0}
        for name in ("service", "cpu")
    }
    for wav in sorted(data_dir.glob("*.wav")):
        name = wav.stem
        ref = load_rttm(data_dir / f"{name}.rttm")
        duration = wav_duration(wav)
        row = {"name": name, "duration_s": duration}
        service_path = service_dir / f"{name}.json"
        cpu_path = cpu_dir / f"{name}.json"
        if service_path.exists():
            hyp, payload = load_json(service_path)
            s0 = detailed(ref, hyp, duration, 0.0)
            s25 = detailed(ref, hyp, duration, 0.25)
            row["service"] = {
                "der0": 100 * s0["der"],
                "der25": 100 * s25["der"],
                "fa0": 100 * s0["fa"] / s0["total"],
                "miss0": 100 * s0["miss"] / s0["total"],
                "conf0": 100 * s0["conf"] / s0["total"],
                "segments": len(hyp),
                "speakers": len(hyp.labels()),
                "time_s": payload.get("inference_time_s"),
                "rtf": payload.get("inference_time_s", 0.0) / duration,
            }
            for key in ("fa", "miss", "conf", "total"):
                totals["service"][key] += s0[key]
                totals25["service"][key] += s25[key]
        else:
            row["service"] = None
        if cpu_path.exists():
            hyp, payload = load_json(cpu_path)
            c0 = detailed(ref, hyp, duration, 0.0)
            c25 = detailed(ref, hyp, duration, 0.25)
            inference_t = payload.get("metrics", {}).get("inference_t")
            row["cpu"] = {
                "der0": 100 * c0["der"],
                "der25": 100 * c25["der"],
                "fa0": 100 * c0["fa"] / c0["total"],
                "miss0": 100 * c0["miss"] / c0["total"],
                "conf0": 100 * c0["conf"] / c0["total"],
                "segments": len(hyp),
                "speakers": len(hyp.labels()),
                "time_s": inference_t,
                "rtf": inference_t / duration if inference_t else None,
            }
            for key in ("fa", "miss", "conf", "total"):
                totals["cpu"][key] += c0[key]
                totals25["cpu"][key] += c25[key]
            if service_path.exists():
                service_hyp, _ = load_json(service_path)
                direct = detailed(service_hyp, hyp, duration, 0.0)
                row["service_vs_cpu_der0"] = 100 * direct["der"]
        else:
            row["cpu"] = None
        if row["cpu"] and row["service"]:
            row["delta_der0_pp"] = row["service"]["der0"] - row["cpu"]["der0"]
            row["delta_der25_pp"] = row["service"]["der25"] - row["cpu"]["der25"]
            if row["cpu"]["time_s"] and row["service"]["time_s"]:
                row["speedup"] = row["cpu"]["time_s"] / row["service"]["time_s"]
        rows.append(row)

    summary = {
        "rows": rows,
        "aggregate": {
            kind: {
                "der0": 100
                * (values["fa"] + values["miss"] + values["conf"])
                / values["total"]
                if values["total"]
                else None,
                "fa0": 100 * values["fa"] / values["total"] if values["total"] else None,
                "miss0": 100
                * values["miss"]
                / values["total"]
                if values["total"]
                else None,
                "conf0": 100
                * values["conf"]
                / values["total"]
                if values["total"]
                else None,
                "total_s": values["total"],
                **{
                    "der25": 100
                    * (
                        totals25[kind]["fa"]
                        + totals25[kind]["miss"]
                        + totals25[kind]["conf"]
                    )
                    / totals25[kind]["total"]
                    if totals25[kind]["total"]
                    else None,
                    "fa25": 100
                    * totals25[kind]["fa"]
                    / totals25[kind]["total"]
                    if totals25[kind]["total"]
                    else None,
                    "miss25": 100
                    * totals25[kind]["miss"]
                    / totals25[kind]["total"]
                    if totals25[kind]["total"]
                    else None,
                    "conf25": 100
                    * totals25[kind]["conf"]
                    / totals25[kind]["total"]
                    if totals25[kind]["total"]
                    else None,
                },
            }
            for kind, values in totals.items()
        },
    }
    Path(args.output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
