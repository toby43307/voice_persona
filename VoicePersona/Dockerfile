# ---------- Stage 1: micromamba ----------
FROM mambaorg/micromamba:1.5.8 AS micromamba_stage

# ---------- Stage 2: CUDA runtime ----------
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Copy micromamba binary (CORRECT PATH)
COPY --from=micromamba_stage /bin/micromamba /usr/local/bin/micromamba

# Micromamba config
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:$PATH

# Force bash (safer on WSL2)
SHELL ["/bin/bash", "-c"]

# Working directory
WORKDIR /VoicePersona

# Copy environment file
COPY docker_environment.yml .

# Create conda environment
RUN micromamba create -y -n voicepersona_env -f docker_environment.yml && \
    micromamba clean --all --yes

# Activate environment
ENV PATH=/opt/conda/envs/voicepersona_env/bin:$PATH

# Copy project files
COPY . .

# Default command
CMD ["python", "main.py"]
