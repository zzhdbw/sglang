#!/usr/bin/env python3
"""Replay a PCM16 WAV over SGLang-Omni's live diarization WebSocket."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave

import websockets


async def receive_until(websocket, expected: str):
    while True:
        event = json.loads(await websocket.recv())
        if event["type"] == "error":
            raise RuntimeError(event["message"])
        if event["type"] == "diarization.update":
            for segment in event["segments"]:
                print(
                    f'{segment["start"]:8.2f} - {segment["end"]:8.2f}s '
                    f'{segment["speaker"]}'
                )
        if event["type"] == expected:
            return event


async def replay(url: str, path: str) -> None:
    async with websockets.connect(url, max_size=1024 * 1024) as websocket:
        ready = json.loads(await websocket.recv())
        print(f"session: {ready['session_id']} ({ready['sample_rate']} Hz PCM16)")
        with wave.open(path, "rb") as source:
            if (
                source.getnchannels(),
                source.getframerate(),
                source.getsampwidth(),
            ) != (1, 16000, 2):
                raise SystemExit("Use a mono 16 kHz PCM16 WAV file")
            sent_samples = 0
            started = time.monotonic()
            while data := source.readframes(1600):
                await websocket.send(data)
                await receive_until(websocket, "audio.ack")
                sent_samples += len(data) // 2
                await asyncio.sleep(
                    max(0, started + sent_samples / 16000 - time.monotonic())
                )
        await websocket.send(json.dumps({"type": "audio.end"}))
        done = await receive_until(websocket, "diarization.done")
        print(f"finished: {done['duration']:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument(
        "--url", default="ws://127.0.0.1:18080/v1/audio/diarizations/stream"
    )
    args = parser.parse_args()
    asyncio.run(replay(args.url, args.audio))


if __name__ == "__main__":
    main()
