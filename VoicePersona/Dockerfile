# ---------- Stage 1: micromamba ----------
FROM mambaorg/micromamba:1.5.8 AS micromamba_stage

# ---------- Stage 2: CUDA runtime ----------
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# System dependencies (minimal)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    bzip2 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy micromamba binary
COPY --from=micromamba_stage /bin/micromamba /usr/local/bin/micromamba

# Micromamba config
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:$PATH

SHELL ["/bin/bash", "-c"]

WORKDIR /VoicePersona

# Copy only env file first (cache-friendly)
COPY docker_environment_py39.yml ./docker_environment.yml

# Create conda environment
RUN micromamba create -y \
    -n voicepersona_env \
    -f docker_environment.yml && \
    micromamba clean --all --yes

# Activate environment
ENV PATH=/opt/conda/envs/voicepersona_env/bin:$PATH

# Copy project files
COPY . .

# Expose Flask port (optional)
EXPOSE 5001

CMD ["python", "VoicePersona/app.py"]
