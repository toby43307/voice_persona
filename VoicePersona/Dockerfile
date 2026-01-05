# ---------- Stage 1: micromamba ----------
FROM mambaorg/micromamba:1.5.8 AS micromamba_stage

# ---------- Stage 2: CUDA runtime ----------
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# System dependencies (minimal)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    bzip2 \
    git \
    build-essential \
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

# sanity check that torch is visible inside env
RUN micromamba run -n voicepersona_env python -c "import torch; import torchvision; import torchaudio; print(torch.__version__)"

# Copy project files
COPY . .

# Expose Flask port (optional)
EXPOSE 5001

CMD ["/bin/bash"]
