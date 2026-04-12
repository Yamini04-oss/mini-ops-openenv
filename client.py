"""
MiniOpsEnvClient — typed client for MiniOpsEnv.

Inherits from openenv-core's EnvClient and implements the three
required abstract methods: _step_payload, _parse_result, _parse_state.
"""
from typing import Any, Dict, Tuple

try:
    from openenv.core.env_server.client import EnvClient
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from exc

from mini_ops_env.models import Action, Observation


class MiniOpsEnvClient(EnvClient):
    """
    Client for MiniOpsEnv.

    Usage
    -----
    client = MiniOpsEnvClient(base_url="http://localhost:7860")
    obs = client.reset()
    obs, reward, done, info = client.step(
        action_type="submit_answer",
        payload="important"
    )
    """

    # ------------------------------------------------------------------
    # Required abstract method implementations
    # ------------------------------------------------------------------

    def _step_payload(self, **kwargs) -> Dict:
        """
        Build the JSON payload for the POST /step endpoint.

        Parameters (via kwargs)
        -----------------------
        action_type : str   — must be "submit_answer"
        payload     : Any   — the agent's answer
        """
        return Action(
            action_type=kwargs.get("action_type", "submit_answer"),
            payload=kwargs.get("payload"),
        ).model_dump()

    def _parse_result(
        self, response: Dict
    ) -> Tuple[Observation, float, bool, Dict]:
        """
        Parse the JSON response from POST /step into a typed tuple.

        Expected response shape (from openenv-core):
        {
            "observation": {...},
            "reward": float,
            "done": bool,
            "info": {...}
        }
        """
        obs = Observation(**response["observation"])
        reward = float(response.get("reward", 0.0))
        done = bool(response.get("done", False))
        info = response.get("info", {})
        return obs, reward, done, info

    def _parse_state(self, response: Dict) -> Dict:
        """
        Parse the JSON response from GET /state into a plain dict.
        """
        return response
