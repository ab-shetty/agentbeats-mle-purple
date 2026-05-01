"""A2A executor for the MLE-Bench purple agent.

Receives the green's initial message containing:
  - TextPart: instructions
  - FilePart(FileWithBytes): competition.tar.gz

Runs the AIDE-style agent loop, optionally negotiating with the green
via the "validate" handshake, and emits the final submission.csv as a
TaskArtifactUpdateEvent FilePart.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Artifact,
    FilePart,
    FileWithBytes,
    Message,
    Part,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)

from .agent import MLEBenchAgent

logger = logging.getLogger(__name__)

TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class Executor(AgentExecutor):
    """One A2A conversation = one MLE-Bench evaluation run."""

    def __init__(self) -> None:
        self._workspace_root = Path(os.environ.get("WORKSPACE_DIR", "/tmp/purple_workspace"))
        self._workspace_root.mkdir(parents=True, exist_ok=True)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        if not message or not message.parts:
            logger.warning("Received empty message")
            return

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            logger.info("Task %s already in terminal state", task.id)
            return

        task_id = context.task_id or "unknown"
        context_id = context.context_id or "unknown"

        # Pull the instruction text + competition tarball out of the message.
        instructions = ""
        tar_bytes: bytes | None = None
        for part in message.parts:
            root = part.root if hasattr(part, "root") else part
            if isinstance(root, TextPart):
                instructions = root.text
            elif isinstance(root, FilePart):
                file_data = root.file
                if isinstance(file_data, FileWithBytes):
                    tar_bytes = base64.b64decode(file_data.bytes)

        if tar_bytes is None:
            await self._emit_status(
                event_queue, task_id, context_id,
                TaskState.failed, "No competition tarball provided", final=True,
            )
            return

        await self._emit_status(
            event_queue, task_id, context_id,
            TaskState.working, "Extracting competition data…", final=False,
        )

        try:
            workspace = Path(tempfile.mkdtemp(prefix="run_", dir=self._workspace_root))
            data_dir = workspace / "home" / "data"
            with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r:gz") as tar:
                tar.extractall(workspace)
            if not data_dir.exists():
                # Some competitions may not nest under home/data/, fall back.
                # Walk the workspace for description.md.
                for p in workspace.rglob("description.md"):
                    data_dir = p.parent
                    break
        except Exception as e:
            logger.exception("Tar extraction failed")
            await self._emit_status(
                event_queue, task_id, context_id,
                TaskState.failed, f"Tar extraction failed: {e}", final=True,
            )
            return

        await self._emit_status(
            event_queue, task_id, context_id,
            TaskState.working, "Running plan/code/execute/debug loop…", final=False,
        )

        agent = MLEBenchAgent(
            workspace=workspace,
            data_dir=data_dir,
            instructions=instructions,
        )

        try:
            submission_csv_path: Path = await asyncio.to_thread(agent.run)
        except Exception as e:
            logger.exception("Agent run failed")
            await self._emit_status(
                event_queue, task_id, context_id,
                TaskState.failed, f"Agent run failed: {e}", final=True,
            )
            return

        if not submission_csv_path.exists():
            await self._emit_status(
                event_queue, task_id, context_id,
                TaskState.failed, "Agent did not produce submission.csv", final=True,
            )
            return

        submission_bytes = submission_csv_path.read_bytes()

        # Emit the submission as a task artifact (this is what green grades).
        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                artifact=Artifact(
                    artifactId=uuid4().hex,
                    name="submission",
                    parts=[Part(root=FilePart(
                        file=FileWithBytes(
                            bytes=base64.b64encode(submission_bytes).decode("ascii"),
                            name="submission.csv",
                            mime_type="text/csv",
                        ),
                    ))],
                ),
            )
        )

        await self._emit_status(
            event_queue, task_id, context_id,
            TaskState.completed, "Submitted submission.csv", final=True,
        )

    async def _emit_status(
        self,
        event_queue: EventQueue,
        task_id: str,
        context_id: str,
        state: TaskState,
        text: str,
        *,
        final: bool,
    ) -> None:
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                status=TaskStatus(
                    state=state,
                    message=Message(
                        messageId=uuid4().hex,
                        role="agent",
                        parts=[Part(root=TextPart(kind="text", text=text))],
                    ),
                ),
                final=final,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError(message="Cancellation not supported")
