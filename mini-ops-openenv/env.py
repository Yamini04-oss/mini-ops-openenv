"""
MiniOpsEnv — core environment implementation.

Implements the OpenEnv interface:
  - reset()  → Observation
  - step(action) → (Observation, float, bool, dict)
  - state()  → dict

Action format (must match inference.py / MiniOpsClient):
  {
      "action_type": "respond",
      "payload": {"text": "<agent's answer>"}
  }

The env reads action.payload["text"] and never crashes on bad input.
"""
from typing import Any, Dict, Tuple

from mini_ops_env.graders import grade
from mini_ops_env.models import Action, Observation
from mini_ops_env.tasks import get_task, num_tasks

MAX_STEPS = 3  # one step per task; episode = 3 tasks


class MiniOpsEnv:
    """
    Simulates three real-world assistant tasks:
      0. Email Classification  (easy)
      1. Task Prioritization   (medium)
      2. Data Cleaning         (hard)
    """

    def __init__(self):
        self._task_index: int = 0
        self._step_count: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._scores: list = []

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment and return the first observation."""
        self._task_index = 0
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._scores = []
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """
        Process the agent's action for the current task.

        Expects:
            action.action_type == "respond"
            action.payload     == {"text": "<answer>"}

        Returns
        -------
        observation : Observation
        reward      : float
        done        : bool
        info        : dict
        """
        # ── Guard: already finished ────────────────────────────────────
        if self._done:
            return (
                self._make_observation(message="Episode finished. Call reset()."),
                0.0,
                True,
                {"warning": "episode_done"},
            )

        # ── Coerce dict → Action (never crash) ─────────────────────────
        if isinstance(action, dict):
            try:
                action = Action(**action)
            except Exception as exc:
                return self._penalty_step(f"Could not parse action dict: {exc}")

        # ── Validate action_type ────────────────────────────────────────
        if not hasattr(action, "action_type") or action.action_type != "respond":
            return self._penalty_step(
                f"action_type must be 'respond', got '{getattr(action, 'action_type', None)}'"
            )

        # ── Extract text from payload safely ───────────────────────────
        try:
            payload = action.payload
            if isinstance(payload, dict):
                text = str(payload.get("text", "")).strip()
            else:
                text = str(payload).strip()
        except Exception as exc:
            return self._penalty_step(f"Could not read payload.text: {exc}")

        if not text:
            return self._penalty_step("payload.text is empty.")

        # ── Grade current task ─────────────────────────────────────────
        current_task = get_task(self._task_index)
        task_type = current_task["task_type"]
        expected = current_task["expected_output"]

        try:
            score = grade(task_type, text, expected)
        except Exception as exc:
            score = 0.0

        # ── Shaped reward ──────────────────────────────────────────────
        if score >= 1.0:
            reward = 1.0
        elif score >= 0.5:
            reward = 0.5
        elif score > 0.0:
            reward = 0.2
        else:
            reward = -0.1

        self._cumulative_reward += reward
        self._scores.append({"task": task_type, "score": score, "reward": reward})
        self._step_count += 1
        self._task_index += 1

        # ── Termination: done when score > 0.8 OR all steps used ──────
        if score > 0.8 or self._step_count >= MAX_STEPS or self._task_index >= num_tasks():
            # Only truly done after all 3 tasks are attempted
            if self._task_index >= num_tasks() or self._step_count >= MAX_STEPS:
                self._done = True

        info = {
            "task_graded": task_type,
            "score": score,
            "reward": reward,
            "cumulative_reward": self._cumulative_reward,
            "all_scores": list(self._scores),
        }

        obs = self._make_observation(
            message=f"Graded '{task_type}': score={score:.2f}, reward={reward:.2f}"
        )
        return obs, reward, self._done, info

    def state(self) -> Dict:
        """Return current environment state as a plain dict."""
        return {
            "task_index": self._task_index,
            "step_count": self._step_count,
            "done": self._done,
            "cumulative_reward": self._cumulative_reward,
            "scores": list(self._scores),
            "num_tasks": num_tasks(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _penalty_step(self, reason: str) -> Tuple[Observation, float, bool, Dict]:
        """Return a -0.2 penalty without advancing the task."""
        obs = self._make_observation(message=f"Invalid action: {reason}")
        return obs, -0.2, self._done, {"error": reason}

    def _make_observation(self, message: str = "") -> Observation:
        """Build an Observation for the current (or final) task."""
        if self._done or self._task_index >= num_tasks():
            return Observation(
                task_type="done",
                input_data=None,
                step_count=self._step_count,
                task_index=self._task_index,
                done=True,
                message=message or "All tasks complete.",
            )

        task = get_task(self._task_index)
        return Observation(
            task_type=task["task_type"],
            input_data={
                "description": task["description"],
                "data": task["input_data"],
            },
            step_count=self._step_count,
            task_index=self._task_index,
            done=False,
            message=message or task["description"],
        )
