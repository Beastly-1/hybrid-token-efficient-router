"""Track 1 container entrypoint.

Reads tasks from /input/tasks.json and writes answers to /output/results.json.
Supports either a JSON array or JSON object with a top-level "tasks" field.
Each task should provide at least an id/task_id and prompt/query field.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from router import HybridRouter
from state import RoutingState


INPUT_PATH = Path(os.getenv("INPUT_FILE", "/input/tasks.json"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_FILE", "/output/results.json"))


def _load_tasks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        tasks = data.get("tasks", data.get("cases"))
        if isinstance(tasks, list):
            return tasks
    raise ValueError("tasks.json must be a JSON array or an object with a 'tasks' list")


def _task_id(task: dict, index: int) -> str:
    return str(task.get("task_id") or task.get("id") or f"task-{index}")


def _task_prompt(task: dict) -> str:
    prompt = task.get("prompt") or task.get("query") or task.get("input")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Each task must include a non-empty prompt/query/input string")
    return prompt


def main() -> int:
    router = HybridRouter()
    tasks = _load_tasks(INPUT_PATH)
    results: list[dict] = []

    for index, task in enumerate(tasks, start=1):
        task_id = _task_id(task, index)
        query = _task_prompt(task)
        state = RoutingState(
            task_id=task_id,
            query=query,
        )
        state = router.route(state)
        results.append(
            {
                "task_id": task_id,
                "answer": state.final_answer,
                "route_source": "fireworks" if state.use_remote else "local",
                "selected_model": state.selected_model or None,
                "tool_used": state.tool_used,
                "success": True,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
