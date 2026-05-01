"""Mocked-green local driver for the MLE-Bench purple agent.

Usage:
    # 1. In one shell, run the purple:
    #    OPENAI_API_KEY=... python -m src.server --host 127.0.0.1 --port 8080
    # 2. In another shell, run this driver against a local dataset dir:
    #    python scripts/local_test.py --data-dir /path/to/spaceship-titanic

The dataset dir must contain (at minimum):
    description.md
    train.csv
    test.csv
    sample_submission.csv

Mimics the green agent's wire protocol: sends instructions + competition.tar.gz
as the initial A2A message, then captures the submission.csv FilePart from
the resulting TaskArtifactUpdateEvent.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import sys
import tarfile
import time
from pathlib import Path
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import (
    FilePart,
    FileWithBytes,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TextPart,
)


INSTRUCTIONS = (
    "You are participating in MLE-bench. The competition data is attached as "
    "competition.tar.gz; once extracted, files live under home/data/. Read "
    "home/data/description.md for the task. Produce a submission.csv as a "
    "task artifact (FilePart). You may send a message containing the word "
    "'validate' with a submission.csv FilePart to validate the schema; in "
    "this local driver, validation is not available, so just submit your "
    "best CSV directly."
)


def make_tar(data_dir: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(data_dir), arcname="home/data", recursive=True)
    return buf.getvalue()


async def run(agent_url: str, data_dir: Path, out_path: Path) -> int:
    print(f"[driver] packing {data_dir} into competition.tar.gz")
    tar_bytes = make_tar(data_dir)
    print(f"[driver] tarball: {len(tar_bytes):,} bytes")

    initial_msg = Message(
        kind="message",
        role=Role.user,
        parts=[
            Part(root=TextPart(text=INSTRUCTIONS)),
            Part(root=FilePart(file=FileWithBytes(
                bytes=base64.b64encode(tar_bytes).decode("ascii"),
                name="competition.tar.gz",
                mime_type="application/gzip",
            ))),
        ],
        message_id=uuid4().hex,
    )

    submission_bytes: bytes | None = None
    t0 = time.time()

    async with httpx.AsyncClient(timeout=3600) as http:
        resolver = A2ACardResolver(httpx_client=http, base_url=agent_url)
        card = await resolver.get_agent_card()
        cfg = ClientConfig(httpx_client=http, streaming=True)
        client = ClientFactory(cfg).create(card)

        async for event in client.send_message(initial_msg):
            match event:
                case (_task, TaskStatusUpdateEvent() as upd):
                    msg = upd.status.message
                    if msg and msg.parts:
                        text_parts = [
                            p.root.text for p in msg.parts
                            if hasattr(p, "root") and isinstance(p.root, TextPart)
                        ]
                        if text_parts:
                            print(f"[purple] [{upd.status.state}] {text_parts[0][:200]}")
                case (task, TaskArtifactUpdateEvent()):
                    for art in task.artifacts or []:
                        for part in art.parts:
                            if isinstance(part.root, FilePart):
                                fd = part.root.file
                                if isinstance(fd, FileWithBytes):
                                    submission_bytes = base64.b64decode(fd.bytes)
                                    print(f"[driver] received submission: {len(submission_bytes):,} bytes")
                case _:
                    pass

    elapsed = time.time() - t0
    print(f"[driver] done in {elapsed:.1f}s")

    if submission_bytes is None:
        print("[driver] ERROR: no submission.csv received", file=sys.stderr)
        return 1
    out_path.write_bytes(submission_bytes)
    print(f"[driver] saved submission to {out_path}")

    # Quick sanity check vs sample_submission.csv
    sample = data_dir / "sample_submission.csv"
    if sample.exists():
        try:
            import pandas as pd

            sub = pd.read_csv(out_path)
            ref = pd.read_csv(sample)
            cols_match = list(sub.columns) == list(ref.columns)
            rows_match = len(sub) == len(ref)
            print(
                f"[driver] schema check: cols_match={cols_match} "
                f"rows_match={rows_match} ({len(sub)} vs {len(ref)})"
            )
            if not (cols_match and rows_match):
                return 2
        except Exception as e:
            print(f"[driver] WARN: sanity check failed: {e}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-url", default="http://127.0.0.1:8080/")
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Directory containing description.md, train.csv, test.csv, sample_submission.csv")
    parser.add_argument("--out", default=Path("submission.csv"), type=Path)
    args = parser.parse_args()

    if not args.data_dir.exists():
        sys.exit(f"data dir not found: {args.data_dir}")

    rc = asyncio.run(run(args.agent_url, args.data_dir, args.out))
    sys.exit(rc)


if __name__ == "__main__":
    main()
