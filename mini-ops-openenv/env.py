from mini_ops_env.models import OpsObservation

class MiniOpsEnv:

    def __init__(self):
        self.step_count = 0
        self.current_task = None

    def reset(self):
        self.step_count = 0

        # simple task
        self.current_task = {
            "task_type": "email",
            "input_data": "Meeting at 5pm tomorrow"
        }

        return OpsObservation(
            task_type="email",
            input_data=self.current_task,
            step_count=0
        )

    def step(self, action):
        self.step_count += 1

        # safe parsing (VERY IMPORTANT)
        try:
            text = action.payload.get("text", "")
        except:
            return self._get_obs(), -0.2, False, {"error": "invalid action"}

        # simple logic
        score = 1.0 if "important" in text.lower() else 0.0

        reward = score
        done = score == 1.0 or self.step_count >= 3

        return self._get_obs(), reward, done, {}

    def state(self):
        return {
            "step_count": self.step_count,
            "task": self.current_task
        }

    def _get_obs(self):
        return OpsObservation(
            task_type="email",
            input_data=self.current_task,
            step_count=self.step_count
        )
