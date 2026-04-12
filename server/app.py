from fastapi import FastAPI
from env import MiniOpsEnv
import uvicorn

app = FastAPI()
env = MiniOpsEnv()

@app.get("/")
def root():
    return {"message": "OpenEnv is running"}

@app.post("/reset")
def reset():
    return {"state": env.reset()}

@app.post("/step")
def step(action: dict):
    return env.step(action)

@app.get("/state")
def get_state():
    return {"state": env.state()}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
