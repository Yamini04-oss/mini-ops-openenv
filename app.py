from openenv.core.env_server import create_app
from ..env import MiniOpsEnv

env = MiniOpsEnv()
app = create_app(env)
