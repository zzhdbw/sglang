#!/usr/bin/env python3
"""Upload a recording to SGLang-Omni's /v1/audio/diarizations endpoint."""

from __future__ import annotations

import argparse
import json

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--url", default="http://127.0.0.1:18080/v1/audio/diarizations")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    with open(args.audio, "rb") as audio_file:
        response = requests.post(
            args.url,
            files={"file": (args.audio, audio_file, "application/octet-stream")},
            timeout=args.timeout,
        )
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
