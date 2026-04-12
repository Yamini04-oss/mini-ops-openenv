FROM openenv-base:latest

ENV ENABLE_WEB_INTERFACE=true

WORKDIR /app
COPY . .

# Install your package (important for imports to work)
RUN pip install -e .

# Expose correct port
EXPOSE 7860

# Start OpenEnv server
CMD ["uvicorn", "mini_ops_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
