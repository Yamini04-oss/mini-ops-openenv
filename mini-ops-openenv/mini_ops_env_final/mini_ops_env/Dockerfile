# MiniOpsEnv — Dockerfile
# Builds on the official OpenEnv base image and runs the server on port 7860.

FROM openenv-base:latest

# OpenEnv web interface
ENV ENABLE_WEB_INTERFACE=true

# Inference script defaults (can be overridden at runtime)
ENV API_BASE_URL=https://api-inference.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV ENV_BASE_URL=http://localhost:7860

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install the package + requests (for inference.py HTTP calls)
RUN pip install --no-cache-dir -e . requests

# Expose OpenEnv standard port
EXPOSE 7860

# Start the uvicorn server
CMD ["uvicorn", "mini_ops_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
