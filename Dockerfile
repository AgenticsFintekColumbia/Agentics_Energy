FROM mambaorg/micromamba:1.5.8

ARG ENV_NAME=agentics
ENV ENV_NAME=${ENV_NAME}

WORKDIR /app

# 1. Create conda env from environment.yml
COPY environment.yml /tmp/environment.yml

RUN micromamba create -y -n ${ENV_NAME} -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Use env for everything below
SHELL ["micromamba", "run", "-n", "agentics", "/bin/bash", "-c"]

# 2. System tools (git + safe.directory)
USER root
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* && \
    git config --system --add safe.directory '*'

# 3. Copy repo (as root)
COPY . .

# 3b. Give ownership of /app to the micromamba user
RUN chown -R $MAMBA_USER /app

# 3c. Switch to that user for Python work
USER $MAMBA_USER

# 4. Install packages in editable mode
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ./agentics && \
    pip install --no-cache-dir -e ./agentic_energy

# 5. Expose Streamlit port
EXPOSE 8501

# 6. Start Streamlit app
CMD ["micromamba", "run", "-n", "agentics", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
