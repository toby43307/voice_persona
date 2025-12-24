# VoicePersona Docker 配置

本仓库提供了一个基于 Docker 的环境，用于在 Linux 容器（例如 Nvidia CUDA 基础镜像）中运行 VoicePersona 应用程序。**请勿**在 Docker 内部使用 Windows 的 conda 导出文件 (`environment_win_py39.yml`)，因为它包含操作系统特定的包和路径前缀。

## 准备工作

- 已安装 Docker
- NVIDIA 容器工具包（如果使用 GPU）
- Git

## 构建镜像

Dockerfile 使用一个定义在 `docker_environment_py39.yml` 中的最小化、Linux 友好的 conda 环境。

步骤：

1. 克隆仓库
   - `git clone https://github.com/toby43307/voice_persona`
   - `cd voice_persona`
   
2. 准备所需的数据/模型（将它们放置在您主机的当前工作副本中；它们将被挂载到容器中）
   - `data_util/face_tracking/3DMM/01_MorphableModel.mat` (≈229 MB)
   
     - Basel Face Model。从以下地址下载：https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details
   
   - `data_util/face_tracking/3DMM/3DMM_info.npy` (≈190 MB)
   
     - 使用该脚本生成: `data_util/face_tracking/convert_BFM.py`
   
   - `VoicePersona/data_util/face_parsing/79999_iter.pth` (≈50.8 MB)
   
   - `pretrained_models/CosyVoice2-0.5B/` (CosyVoice 0.5B 检查点和资源)
   
     - 遵循 CosyVoice 说明：https://github.com/FunAudioLLM/CosyVoice  
   
     - 使用 ModelScope SDK 下载：
   
       ```python
       from modelscope import snapshot_download
       snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
       ```
   
     - 或使用 HuggingFace Hub：
   
       ```python
       from huggingface_hub import snapshot_download
       snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
       ```
   
   - `VoicePersona/third_party/Matcha-TTS/` (在此处克隆 Matcha-TTS 仓库)
   
     - 遵循说明: https://github.com/shivammehta25/Matcha-TTS
   
3. 构建镜像
   - `docker build -t voice_persona_py39 .`

注意:
- `docker_environment_py39.yml` 包括:
  - Python 3.9
  - 支持 CUDA 的 PyTorch (cu121)
  - 音频/视觉库：ffmpeg, opencv, libsndfile
  - 通过 pip 安装的语音/计算机视觉/Python 库
- 额外添加的 pip 包:
  - `openai`
  - `resampy==0.4.3`
  - `python-speech-features==0.6`
  - `tensorflow-cpu==2.10.0`

## 运行容器

`docker run -it --rm --gpus all -p 5001:5001 voice_persona_py39 /bin/bash`

然后在容器内，启动应用程序：

root@c7932666d1e5:/VoicePersona# python VoicePersona/app.py

该应用程序对主机可用，请通过 `http://localhost:5001` 访问。

## 关于 environment_win_py39.yml

`environment_win_py39.yml` 是一个 Windows 特定的环境导出文件。请仅在 Windows 主机上使用（在 Docker 外部）：

- 创建 / 更新 Windows conda 环境：
  - `conda env create -f environment_win_py39.yml`
  - or `conda env update -f environment_win_py39.yml`

- 在 Windows 上安装额外的包：
  - `pip install openai`
  - `pip install resampy==0.4.3`
  - `pip install python-speech-features==0.6`
  - `pip install tensorflow-cpu==2.10.0`

**请勿**将 `environment_win_py39.yml` 复制到 Docker 构建中；为了可移植性，请始终使用 `docker_environment_py39.yml`。

## 在应用程序中提供数据集媒体文件

Flask 应用提供以下端点访问：

- `/media/output/<file>` 对应 `output/` 目录
- `/media/asset/<file>` 对应 `asset/` 目录
- `/media/dataset/<path>` 对应 `dataset/` 目录

这使得生成的音频和视频可以在 UI 中预览。

## 故障排除

- 如果 PyTorch CUDA 包安装失败，请验证您的基础镜像 CUDA 版本是否与 cu121 匹配，或者通过移除 extra-index URL 并使用 CPU 包来切换到 CPU 版本的 PyTorch。
- 如果在 Docker 内发生 TensorFlow 错误，考虑保持仅使用 TensorFlow CPU 版本（`tensorflow-cpu==2.10.0`），如定义所示。
- 确保 `speaker_profiles.json` 中的路径指向 `asset/` 目录下的有效文件。

## 评估步骤

```cmd
cd eval

mkdir output_folder

ffmpeg -i adnerf_obama.avi -vf "fps=25" output_folder/frame_%06d.png

python evaluate_talkingface.py --real path/to/real_images --fake path/to/fake_images
```

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
   
2. Prepare required data/models (place them in your working copy on the host; they will be mounted into the container)
   - `data_util/face_tracking/3DMM/01_MorphableModel.mat` (≈229 MB)
   
     - Basel Face Model. Download from: https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details
   
   - `data_util/face_tracking/3DMM/3DMM_info.npy` (≈190 MB)
   
     - Generate using: `data_util/face_tracking/convert_BFM.py`
   
   - `VoicePersona/data_util/face_parsing/79999_iter.pth` (≈50.8 MB)
   
   - `pretrained_models/CosyVoice2-0.5B/` (CosyVoice 0.5B checkpoints and assets)
   
     - Follow CosyVoice instructions: https://github.com/FunAudioLLM/CosyVoice  
   
     - To download with ModelScope SDK:
   
       ```python
       from modelscope import snapshot_download
       snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
       ```
   
     - Or with HuggingFace Hub:
   
       ```python
       from huggingface_hub import snapshot_download
       snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
       ```
   
   - `VoicePersona/third_party/Matcha-TTS/` (clone the Matcha-TTS repo here)
   
     - Follow instructions: https://github.com/shivammehta25/Matcha-TTS
   
3. Build the image
   - `docker build -t voice_persona_py39 .`

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

- `docker run -it --rm --gpus all -p 5001:5001 voice_persona_py39 /bin/bash`

then inside the container, start the app:

root@c7932666d1e5:/VoicePersona# python VoicePersona/app.py

The app will be available to the host, visit it at `http://localhost:5001`.

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

## Evaluate Step：

```python
cd eval

mkdir output_folder

ffmpeg -i adnerf_obama.avi -vf "fps=25" output_folder/frame_%06d.png

python evaluate_talkingface.py --real path/to/real_images --fake path/to/fake_images
```

