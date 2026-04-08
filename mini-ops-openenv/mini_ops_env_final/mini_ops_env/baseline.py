"""
baseline.py — Baseline inference script for MiniOpsEnv.

Uses the OpenAI-compatible API (credentials from HF_TOKEN env var) to run
an LLM agent through all three tasks and report per-task scores + average.

Action format (matches MiniOpsClient / inference.py):
    action_type = "respond"
    payload     = {"text": "<answer>"}

Run locally (no server required) by instantiating MiniOpsEnv directly.

Usage
-----
  export HF_TOKEN=hf_...
  python baseline.py
"""
import json
import os
import sys

from openai import OpenAI

from mini_ops_env.env import MiniOpsEnv
from mini_ops_env.models import Action
from mini_ops_env.tasks import TASKS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable is not set.", file=sys.stderr)
    sys.exit(1)

BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    "https://api-inference.huggingface.co/v1",
)
MODEL = os.environ.get("BASELINE_MODEL", "Qwen/Qwen2.5-72B-Instruct")

client = OpenAI(api_key=HF_TOKEN, base_url=BASE_URL)

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a precise AI assistant solving structured tasks.
Follow the task instructions exactly.
Return ONLY the requested output — no explanations, no markdown fences.
"""


def call_llm(task_description: str, task_data) -> str:
    """Send a task to the LLM and return the raw text response."""
    user_message = (
        f"{task_description}\n\n"
        f"Input:\n{json.dumps(task_data, indent=2)}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main baseline loop
# ---------------------------------------------------------------------------

def run_baseline() -> None:
    env = MiniOpsEnv()
    obs = env.reset()

    scores = []
    rewards = []

    print("=" * 60)
    print("MiniOpsEnv — Baseline Evaluation")
    print(f"Model : {MODEL}")
    print(f"Tasks : {len(TASKS)}")
    print("=" * 60)

    for i, task in enumerate(TASKS):
        task_type   = task["task_type"]
        description = task["description"]
        input_data  = task["input_data"]

        print(f"\n[Task {i+1}/3] {task_type}")

        # Query the LLM — raw text answer
        raw_answer = call_llm(description, input_data)
        print(f"  LLM answer : {str(raw_answer)[:120]}")

        # Build action using the respond / text format
        action = Action(
            action_type="respond",
            payload={"text": raw_answer},
        )

        obs, reward, done, info = env.step(action)

        task_score = info.get("score", 0.0)
        scores.append(task_score)
        rewards.append(reward)

        print(f"  Score  : {task_score:.2f}")
        print(f"  Reward : {reward:.2f}")

    avg_score    = sum(scores)  / len(scores)
    total_reward = sum(rewards)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for i, (task, s) in enumerate(zip(TASKS, scores)):
        print(f"  Task {i+1} ({task['task_type']:28s}) : {s:.2f}")
    print(f"\n  Average Score  : {avg_score:.4f}")
    print(f"  Total Reward   : {total_reward:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_baseline()
