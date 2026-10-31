#!/usr/bin/env python3
"""Measure live-diarization throughput across upload-chunk sizes/intervals.

Two modes are supported:

* ``--interval-ms 0``: closed-loop, send the next chunk immediately after the
  previous chunk is acknowledged.  This measures the server's sustainable
  pipeline throughput for one or more streams.
* ``--interval-ms N``: paced mode, every stream sends its next chunk every N
  milliseconds.  ``N == chunk_ms`` is a realtime stream.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import wave

import websockets

SAMPLE_RATE = 16000


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


async def closed_loop_stream(
    stream_id: int,
    url: str,
    chunks: list[bytes],
    start_event: asyncio.Event,
    connected: list[int],
    all_connected: asyncio.Event,
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

            ack_rtts: list[float] = []
            updates = 0

            async def receive_until_ack() -> None:
                nonlocal updates
                while True:
                    message = json.loads(await websocket.recv())
                    kind = message.get("type")
                    if kind == "audio.ack":
                        return
                    if kind == "diarization.update":
                        updates += 1
                    elif kind == "error":
                        raise RuntimeError(message.get("message", "unknown error"))
                    elif kind == "diarization.done":
                        return
                    else:
                        raise RuntimeError(message)

            started = time.monotonic()
            for chunk in chunks:
                sent_at = time.monotonic()
                await websocket.send(chunk)
                await receive_until_ack()
                ack_rtts.append(time.monotonic() - sent_at)
            await websocket.send(json.dumps({"type": "audio.end"}))
            # The final finish window can return logs/done after the last ack.
            while True:
                message = json.loads(await websocket.recv())
                kind = message.get("type")
                if kind == "diarization.done":
                    break
                if kind == "diarization.update":
                    updates += 1
                elif kind == "error":
                    raise RuntimeError(message.get("message", "unknown error"))
            wall = time.monotonic() - started
            results[stream_id] = {
                "acks": len(ack_rtts),
                "updates": updates,
                "wall_s": wall,
                "ack_rtt_mean": statistics.mean(ack_rtts) if ack_rtts else None,
                "ack_rtt_p95": (
                    sorted(ack_rtts)[max(0, int(len(ack_rtts) * 0.95) - 1)]
                    if ack_rtts
                    else None
                ),
            }
    except Exception as exc:  # noqa: BLE001 - benchmark diagnostic
        results[stream_id] = {"acks": 0, "error": f"{type(exc).__name__}: {exc}"}


async def paced_stream(
    stream_id: int,
    url: str,
    chunks: list[bytes],
    interval_s: float,
    start_event: asyncio.Event,
    connected: list[int],
    all_connected: asyncio.Event,
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
            ack_lags: list[float] = []
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
                        ack_lags.append(now - sent_at)
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
                await sleep_until(t0_box[0] + index * interval_s)
                await websocket.send(chunk)
                last_send_at = time.monotonic()
                ack_waiting.append(last_send_at)
            await websocket.send(json.dumps({"type": "audio.end"}))
            await asyncio.wait_for(receive_task, timeout=120)
            lags = sorted(ack_lags)
            results[stream_id] = {
                "acks": len(ack_lags),
                "updates": updates,
                "done": done,
                "error": error,
                "ack_lag_mean": statistics.mean(lags) if lags else None,
                "ack_lag_p95": (
                    lags[max(0, int(len(lags) * 0.95) - 1)] if lags else None
                ),
                "ack_lag_max": max(lags, default=None),
                "drain_s": (
                    ack_lags and (time.monotonic() - last_send_at) or None
                ),
            }
    except Exception as exc:  # noqa: BLE001 - benchmark diagnostic
        results[stream_id] = {
            "acks": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def max_of(results: list[dict], key: str) -> float | None:
    values = [r[key] for r in results if r.get(key) is not None]
    return round(max(values), 4) if values else None


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument(
        "--url", default="ws://127.0.0.1:18086/v1/audio/diarizations/stream"
    )
    parser.add_argument("--streams", type=int, default=1)
    parser.add_argument("--chunk-ms", type=int, default=1000)
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=0,
        help="0 = closed-loop send-after-ack; >0 = fixed send interval",
    )
    parser.add_argument(
        "--audio-seconds-per-stream",
        type=float,
        default=60.0,
        help="amount of audio each stream sends",
    )
    args = parser.parse_args()

    if not 1 <= args.chunk_ms <= 1000:
        raise SystemExit("chunk-ms must be in [1, 1000] (server max is 1 s)")
    chunk_frames = int(SAMPLE_RATE * args.chunk_ms / 1000)
    chunk_bytes = chunk_frames * 2
    pcm = load_pcm(args.audio)
    n_chunks = max(1, int(args.audio_seconds_per_stream * 1000 / args.chunk_ms))
    chunks = [pcm[i * chunk_bytes : (i + 1) * chunk_bytes] for i in range(n_chunks)]
    if any(len(chunk) != chunk_bytes for chunk in chunks):
        raise SystemExit("Audio is shorter than requested duration")
    total_audio = n_chunks * chunk_frames / SAMPLE_RATE

    start_event = asyncio.Event()
    all_connected = asyncio.Event()
    connected = [0]
    t0_box = [0.0]
    results: list[dict] = [{} for _ in range(args.streams)]
    interval_s = args.interval_ms / 1000
    wall_start = time.monotonic()
    if args.interval_ms == 0:
        tasks = [
            asyncio.create_task(
                closed_loop_stream(
                    i,
                    args.url,
                    chunks,
                    start_event,
                    connected,
                    all_connected,
                    results,
                )
            )
            for i in range(args.streams)
        ]
    else:
        tasks = [
            asyncio.create_task(
                paced_stream(
                    i,
                    args.url,
                    chunks,
                    interval_s,
                    start_event,
                    connected,
                    all_connected,
                    t0_box,
                    results,
                )
            )
            for i in range(args.streams)
        ]
    await asyncio.wait_for(all_connected.wait(), timeout=30)
    if args.interval_ms == 0:
        start_event.set()
    else:
        t0_box[0] = time.monotonic() + 0.1
        start_event.set()
    await asyncio.gather(*tasks)
    wall = time.monotonic() - wall_start
    errors = [r for r in results if r.get("error")]
    total_acks = sum(r.get("acks", 0) for r in results)
    summary = {
        "streams": args.streams,
        "chunk_ms": args.chunk_ms,
        "interval_ms": args.interval_ms,
        "mode": "closed_loop" if args.interval_ms == 0 else "paced",
        "audio_seconds_per_stream": round(total_audio, 3),
        "aggregate_audio_seconds": round(total_audio * args.streams, 3),
        "wall_seconds": round(wall, 4),
        "aggregate_x_realtime": round(total_audio * args.streams / wall, 3),
        "total_acks": total_acks,
        "expected_acks": args.streams * n_chunks,
        "errors": len(errors),
        "first_errors": [r["error"] for r in errors[:3]],
        "ack_lag_p95_max": max_of(results, "ack_lag_p95"),
        "ack_lag_max": max_of(results, "ack_lag_max"),
        "ack_rtt_mean_max": max_of(results, "ack_rtt_mean"),
        "ack_rtt_p95_max": max_of(results, "ack_rtt_p95"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
