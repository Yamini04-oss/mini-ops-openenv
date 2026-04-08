from openenv.core.env_server.types import Observation, Action

class OpsObservation(Observation):
    task_type: str
    input_data: dict
    step_count: int

class OpsAction(Action):
    action_type: str
    payload: dict
