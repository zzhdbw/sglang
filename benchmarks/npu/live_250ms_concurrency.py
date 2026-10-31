#!/usr/bin/env python3
"""Measure live diarization capacity when every stream sends 250 ms / 250 ms."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import wave

import websockets

CHUNK_SECONDS = 0.25
SAMPLE_RATE = 16000
CHUNK_FRAMES = int(CHUNK_SECONDS * SAMPLE_RATE)
CHUNK_BYTES = CHUNK_FRAMES * 2


def load_pcm(path: str) -> bytes:
    with wave.open(path, "rb") as source:
        if (
            source.getnchannels(),
            source.getframerate(),
            source.getsampwidth(),
        ) != (1, SAMPLE_RATE, 2):
            raise SystemExit("Expected mono 16 kHz PCM16 WAV")
        return source.readframes(source.getnframes())


async def sleep_until(deadline: float) -> None:
    while (remaining := deadline - time.monotonic()) > 0:
        await asyncio.sleep(remaining)


async def run_stream(
    stream_id: int,
    url: str,
    chunks: list[bytes],
    start_event: asyncio.Event,
    all_connected: asyncio.Event,
    connected: list[int],
    t0_box: list[float],
    results: list[dict],
) -> None:
    try:
        async with websockets.connect(url, max_size=1024 * 1024) as websocket:
            ready = json.loads(await websocket.recv())
            if ready.get("type") != "session.ready":
                raise RuntimeError(ready)
            connected[0] += 1
            if connected[0] == len(results):
                all_connected.set()
            await start_event.wait()

            ack_waiting: list[float] = []
            acks: list[dict] = []
            updates = 0
            error: str | None = None
            done = False

            async def receive() -> None:
                nonlocal updates, error, done
                while True:
                    message = json.loads(await websocket.recv())
                    now = time.monotonic()
                    kind = message.get("type")
                    if kind == "audio.ack":
                        sent_at = ack_waiting.pop(0) if ack_waiting else now
                        acks.append(
                            {
                                "lag": now - sent_at,
                                "processed_until": message["processed_until"],
                                "at": now,
                            }
                        )
                    elif kind == "diarization.update":
                        updates += 1
                    elif kind == "diarization.done":
                        done = True
                        return
                    elif kind == "error":
                        error = message.get("message", "unknown error")
                        return
                    else:
                        raise RuntimeError(message)

            receive_task = asyncio.create_task(receive())
            last_send_at = 0.0
            for index, chunk in enumerate(chunks):
                await sleep_until(t0_box[0] + index * CHUNK_SECONDS)
                await websocket.send(chunk)
                last_send_at = time.monotonic()
                ack_waiting.append(last_send_at)
            await websocket.send(json.dumps({"type": "audio.end"}))
            await asyncio.wait_for(receive_task, timeout=120)
            lags = sorted(a["lag"] for a in acks)
            results[stream_id] = {
                "acks": len(acks),
                "expected_acks": len(chunks),
                "updates": updates,
                "done": done,
                "error": error,
                "ack_lag_mean": statistics.mean(lags) if lags else None,
                "ack_lag_p95": (
                    lags[max(0, int(len(lags) * 0.95) - 1)] if lags else None
                ),
                "ack_lag_max": max(lags, default=None),
                "drain_s": (
                    acks[-1]["at"] - last_send_at if acks and last_send_at else None
                ),
                "processed_until": acks[-1]["processed_until"] if acks else None,
            }
    except Exception as exc:  # noqa: BLE001 - benchmark diagnostic
        results[stream_id] = {
            "acks": 0,
            "expected_acks": len(chunks),
            "error": f"{type(exc).__name__}: {exc}",
        }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--url", default="ws://127.0.0.1:18083/v1/audio/diarizations/stream")
    parser.add_argument("--streams", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    pcm = load_pcm(args.audio)
    n_chunks = int(args.seconds / CHUNK_SECONDS)
    chunks = [pcm[i * CHUNK_BYTES : (i + 1) * CHUNK_BYTES] for i in range(n_chunks)]
    if any(len(chunk) != CHUNK_BYTES for chunk in chunks):
        raise SystemExit("Audio is shorter than requested duration")

    start_event = asyncio.Event()
    all_connected = asyncio.Event()
    connected = [0]
    t0_box = [0.0]
    results: list[dict] = [{} for _ in range(args.streams)]
    wall_start = time.monotonic()
    tasks = [
        asyncio.create_task(
            run_stream(
                index,
                args.url,
                chunks,
                start_event,
                all_connected,
                connected,
                t0_box,
                results,
            )
        )
        for index in range(args.streams)
    ]
    await asyncio.wait_for(all_connected.wait(), timeout=30)
    t0_box[0] = time.monotonic() + 0.1
    start_event.set()
    await asyncio.gather(*tasks)
    wall = time.monotonic() - wall_start
    errors = [r for r in results if r.get("error")]
    total_acks = sum(r.get("acks", 0) for r in results)
    def max_of(key: str) -> float | None:
        values = [r[key] for r in results if r.get(key) is not None]
        return round(max(values), 4) if values else None

    summary = {
        "streams": args.streams,
        "seconds_per_stream": args.seconds,
        "wall_seconds": round(wall, 3),
        "aggregate_audio_seconds": round(args.streams * args.seconds, 1),
        "aggregate_x_realtime": round(args.streams * args.seconds / wall, 2),
        "total_acks": total_acks,
        "expected_acks": args.streams * n_chunks,
        "errors": len(errors),
        "first_errors": [r["error"] for r in errors[:3]],
        "ack_lag_mean_max": max_of("ack_lag_mean"),
        "ack_lag_p95_max": max_of("ack_lag_p95"),
        "ack_lag_max": max_of("ack_lag_max"),
        "drain_s_max": max_of("drain_s"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
