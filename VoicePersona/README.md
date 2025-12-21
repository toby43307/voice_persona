# VoicePersona Docker setup

This repository provides a Docker-based environment for running the VoicePersona app on Linux containers (e.g., Nvidia CUDA base images). Do not use the Windows conda export (`environment_win_py39.yml`) inside Docker; it contains OS-specific packages and prefixes.

## Prerequisites
- Docker installed
- NVIDIA Container Toolkit (if using GPU)
- Git

## Build the image
The Dockerfile uses a minimal, Linux-friendly conda environment defined in `docker_environment_py39.yml`.

Steps:
1. Clone the repo
   - `git clone https://github.com/toby43307/voice_persona`
   - `cd voice_persona`
2. Build the image
   - `docker build -t voicepersona:py39 .`

Notes:
- `docker_environment_py39.yml` includes:
  - Python 3.9
  - CUDA-enabled PyTorch (cu121)
  - Audio/vision libs: ffmpeg, opencv, libsndfile
  - Speech/CV/python libs installed via pip
- Additional pip packages added:
  - `openai`
  - `resampy==0.4.3`
  - `python-speech-features==0.6`
  - `tensorflow-cpu==2.10.0`

## Run the container
If you have an NVIDIA GPU and drivers:
- `docker run --gpus all -p 5001:5001 -v %CD%:/app voicepersona:py39`

Without GPU:
- `docker run -p 5001:5001 -v %CD%:/app voicepersona:py39`

The app will be available at `http://localhost:5001`.

## About environment_win_py39.yml
`environment_win_py39.yml` is a Windows-specific environment export. Use it on Windows hosts only (outside Docker):

- Create / update a Windows conda env:
  - `conda env create -f environment_win_py39.yml`
  - or `conda env update -f environment_win_py39.yml`

- Install additional packages on Windows
  - `pip install openai`
  - `pip install resampy==0.4.3`
  - `pip install python-speech-features==0.6`
  - `pip install tensorflow-cpu==2.10.0`

Do not copy `environment_win_py39.yml` into Docker builds; keep using `docker_environment_py39.yml` for portability.

## Serving dataset media in the app
The Flask app exposes endpoints:
- `/media/output/<file>` for `output/`
- `/media/asset/<file>` for `asset/`
- `/media/dataset/<path>` for `dataset/`

This allows generated audio and videos to be previewed in the UI.

## Troubleshooting
- If PyTorch CUDA wheels fail, verify your base image CUDA version matches cu121, or switch to CPU PyTorch by removing the extra-index URL and using CPU wheels.
- If TensorFlow errors occur inside Docker, consider keeping TensorFlow CPU only (`tensorflow-cpu==2.10.0`) as defined.
- Ensure `speaker_profiles.json` points to valid files under `asset/`.
