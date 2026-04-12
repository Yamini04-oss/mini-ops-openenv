"""


Reads environment variables:
    API_BASE_URL  (default: https://api-inference.huggingface.co/v1)
    MODEL_NAME    (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      (mandatory, no default)

Output format (exact):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""
import json
import os
import sys

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. Environment variables (with required defaults)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ---------------------------------------------------------------------------
# 2. OpenAI client (HF Inference API is OpenAI-compatible)
# ---------------------------------------------------------------------------
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# 3. Env HTTP helpers  (talks to the running MiniOpsEnv server)
# ---------------------------------------------------------------------------

def env_reset() -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action_type: str, payload: dict) -> dict:
    body = {"action_type": action_type, "payload": payload}
    r = requests.post(f"{ENV_BASE_URL}/step", json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def env_close() -> None:
    """Best-effort close — env server has no explicit close endpoint."""
    try:
        requests.post(f"{ENV_BASE_URL}/reset", timeout=5)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 4. LLM helper
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise AI assistant solving structured tasks. "
    "Follow the task instructions exactly. "
    "Return ONLY the requested output — no explanations, no markdown fences."
)


def call_llm(task_description: str, task_data) -> str:
    """Call the LLM and return a single clean text answer."""
    user_msg = f"{task_description}\n\nInput:\n{json.dumps(task_data, indent=2)}"
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# 5. Main episode loop
# ---------------------------------------------------------------------------

TASK_NAMES = [
    "email_classification",
    "task_prioritization",
    "data_cleaning",
]
ENV_NAME = "mini_ops_env"


def run_episode(task_name: str) -> None:
    """Run one full episode for a given task name."""
    step_num   = 0
    all_rewards = []
    last_error  = None
    success     = False

    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    try:
        obs_raw = env_reset()
        obs = obs_raw if isinstance(obs_raw, dict) else obs_raw

        done = obs.get("done", False)

        while not done and step_num < 3:
            # Extract task input for the LLM
            input_data = obs.get("input_data") or {}
            description = input_data.get("description", obs.get("message", ""))
            data        = input_data.get("data", input_data)

            # Query the LLM
            try:
                llm_answer = call_llm(description, data)
            except Exception as llm_err:
                llm_answer  = ""
                last_error  = str(llm_err)

            # Sanitise: remove newlines so [STEP] stays on one line
            action_str = llm_answer.replace("\n", " ").replace("\r", " ")[:200]

            # Send action to env
            result = env_step(
                action_type="respond",
                payload={"text": llm_answer},
            )

            reward  = float(result.get("reward", 0.0))
            done    = bool(result.get("done",   False))
            info    = result.get("info",   {})
            obs     = result.get("observation", {})

            last_error = info.get("error", None)
            step_num  += 1
            all_rewards.append(reward)

            error_str = last_error if last_error else "null"
            print(
                f"[STEP] step={step_num} "
                f"action={action_str!r} "
                f"reward={reward:.2f} "
                f"done={'true' if done else 'false'} "
                f"error={error_str}",
                flush=True,
            )

            # Treat total reward > 0.8 per step as success signal
            if done:
                total_score = sum(
                    s.get("score", 0.0)
                    for s in info.get("all_scores", [])
                )
                success = total_score > 0.0

    except Exception as exc:
        last_error = str(exc)
    finally:
        env_close()
        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards) if all_rewards else "0.00"
        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={step_num} "
            f"rewards={rewards_str}",
            flush=True,
        )


def main() -> None:
    """Run all tasks sequentially (one episode each)."""
    # The env runs all 3 tasks in a single episode.
    # We map the combined episode to the primary task label.
    run_episode("email_classification+task_prioritization+data_cleaning")


if __name__ == "__main__":
    main()
