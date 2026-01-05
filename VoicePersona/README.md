# VoicePersona

本项目集成了两个前沿的开源项目：基于 NeRF 的说话人合成项目AD-NeRF（https://github.com/YudongGuo/AD-NeRF），以及用于高质量、零样本语音克隆的 CosyVoice 项目（https://github.com/FunAudioLLM/CosyVoice/）。我们构建了一个统一、可复现的运行环境，并开发了一个精简流畅的 Web UI，整合了从数据处理、模型训练、声音克隆合成，以及演示的全过程。在整合过程中，我们用了大量的精力，统一环境依赖、规范路径和数据布局，并对两套代码进行适配，使其能够在同一个支持 CUDA 的平台上协同运行，从而让用户只需极少的环境配置，即可从示例视频/音频直接完成语音合成并渲染生成视频。

# VoicePersona Docker 配置

本仓库提供了一个基于 Docker 的环境，用于在 Linux 容器（例如 Nvidia CUDA 基础镜像）中运行 VoicePersona 应用程序。**请勿**在 Docker 内部使用 Windows 的 conda 导出文件 (`environment_win_py39.yml`)，因为它包含操作系统特定的包和路径前缀。

## 准备工作

- 已安装 Docker
- NVIDIA 容器工具包
- Git

## 构建镜像

Dockerfile 使用一个定义在 `docker_environment_py39.yml` 中的最小化、Linux 友好的 conda 环境。

步骤：

1. 克隆仓库
   - `git clone https://github.com/toby43307/voice_persona`
   - `cd voice_persona/VoicePersona`
   
2. 准备所需的数据/模型（将它们放置在您主机的当前工作副本中；它们将被挂载到容器中）
   - `data_util/face_tracking/3DMM/01_MorphableModel.mat` (≈229 MB)
   
     - Basel Face Model。从以下地址下载：https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details
   
   - `data_util/face_tracking/3DMM/3DMM_info.npy` (≈190 MB)
   
     - 使用该脚本生成: `data_util/face_tracking/convert_BFM.py`
   
   - `VoicePersona/data_util/face_parsing/79999_iter.pth` (≈50.8 MB)
   
     - [Download from HuggingFace](https://huggingface.co/afrizalha/musetalk-models/blob/main/face-parse-bisent/79999_iter.pth) 
       **or**  
   
     - [https://github.com/neuralchen/SimSwap Download from Google Drive](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view)
   
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

## 运行容器

在支持GPU上的机器上运行:

- `docker run -it --gpus all -p 5001:5001 voice_persona_py39`

- 然后在容器内，完成以下pytorch3d安装：
    root@2984306f9a83:/VoicePersona# source /opt/conda/etc/profile.d/conda.sh
    root@2984306f9a83:/VoicePersona# conda activate voicepersona_env
    (voicepersona_env) root@2984306f9a83:/VoicePersona# python --version
    Python 3.9.23
    (voicepersona_env) root@2984306f9a83:/VoicePersona# python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
    2.3.1+cu121 True 12.1
    (voicepersona_env) root@2984306f9a83:/VoicePersona# pip install "git+https://github.com/facebookresearch/pytorch3d.git"

**注意：** 根据您的环境或上游基础镜像的变动，Docker 镜像构建过程中可能不会自动安装某些列出的 Python 包。如果您在运行应用时遇到缺失包的错误，请在容器内手动安装依赖：
```
pip install -r requirements.txt
```
或者根据需要单独安装缺失的特定包：
  - `pip install openai`
  - `pip install resampy==0.4.3`
  - `pip install python-speech-features==0.6`
  - `pip install tensorflow-cpu==2.10.0`
  - `pip install hyperpyyaml==1.2.3`
  - `pip install modelscope==1.10.0`
  - `pip install onnxruntime==1.16.3`
  - `pip install omegaconf==2.3.0`
  - `pip install conformer==0.3.2`
  - `pip install hydra-core==1.3.2`
  - `pip install wget==3.2`
  - `pip install natsort==8.4.0`

或者直接运行：
    pip install tensorflow-cpu==2.10.0 hyperpyyaml==1.2.3 modelscope==1.10.0 onnxruntime==1.16.3 omegaconf==2.3.0 conformer==0.3.2 hydra-core==1.3.2 wget==3.2 natsort==8.4.0 configargparse==1.7.1
或：
    pip install -i https://mirrors.aliyun.com/pypi/simple/ tensorflow-cpu==2.10.0 hyperpyyaml==1.2.3 modelscope==1.10.0 onnxruntime==1.16.3 omegaconf==2.3.0 conformer==0.3.2 hydra-core==1.3.2 wget==3.2 natsort==8.4.0 configargparse==1.7.1

- 然后在容器内，启动应用程序：

      (voicepersona_env) root@fec46a8303ef:/VoicePersona# python VoicePersona/app.py

该应用程序对主机可用，请通过 `http://localhost:5001` 访问。

**注意：** 本项目仅使用 CPU 运行在实际应用中是不现实的。如果没有 GPU，该项目仅可用于流程演示目的。大多数模型推理和视频生成任务在纯 CPU 系统上会极其缓慢，甚至可能完全无法运行。

## 关于 environment_win_py39.yml

`environment_win_py39.yml` 是一个 Windows 特定的环境导出文件。请仅在 Windows 主机上使用（在 Docker 外部）：

- 创建 / 更新 Windows conda 环境：
  - `conda env create -f environment_win_py39.yml`
  - or `conda env update -f environment_win_py39.yml`

不要将 `environment_win_py39.yml` 复制到 Docker 构建中；为了可移植性，请继续使用 `docker_environment_py39.yml`。

## 使用 requirements.txt（适用于 pip 用户）

如果你不使用 Conda，可以通过 pip 安装 Python 依赖：

```
pip install -r requirements.txt
```

- 这会安装 `environment_win_py39.yml` 文件中 `pip:` 部分列出的所有 Python 包。
- **系统级或 Conda 管理的包**（例如 CUDA、ffmpeg、OpenCV 等）**不会包含在 `requirements.txt` 中**，如果需要，必须单独安装。

### 如何检查你的环境

1. **检查 Python 版本**

   ```
   python --version
   # 应为 Python 3.9.x
   ```

2. **检查已安装的包**

   ```
   pip list
   # 或使用 pip check 检查依赖冲突
   pip check
   ```

3. **（可选）检查 PyTorch 是否能使用 CUDA**

   ```
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **检查 ffmpeg 是否可用**

   ```
   ffmpeg -version
   ```

如果缺少系统库（如 ffmpeg、CUDA、OpenCV 等），请通过操作系统的包管理器或 Conda 单独安装。

- **Windows 用户**：建议使用提供的 `environment_win_py39.yml` 文件配合 Conda，以确保完整兼容性。
- **Linux 或 Docker 用户**：请使用 Dockerfile 和 `docker_environment_py39.yml`。

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

# VoicePersona

This project integrates two cutting-edge open-source projects: AD-NeRF (https://github.com/YudongGuo/AD-NeRF
), which performs NeRF-based talking-head synthesis, and CosyVoice (https://github.com/FunAudioLLM/CosyVoice/
), which enables high-quality, zero-shot voice cloning. We built a unified, reproducible execution environment and developed a streamlined, user-friendly Web UI that brings together the entire pipeline—from data preprocessing and model training to voice cloning, speech synthesis, and interactive demonstrations.Throughout the integration process, substantial effort was devoted to consolidating environment dependencies, standardizing paths and data layouts, and adapting both codebases so that they can run cohesively on a single CUDA-enabled platform. As a result, users can go from sample video/audio to synthesized speech and rendered video with minimal environment setup.

# VoicePersona Docker setup

This repository provides a Docker-based environment for running the VoicePersona app on Linux containers (e.g., Nvidia CUDA base images). Do not use the Windows conda export (`environment_win_py39.yml`) inside Docker; it contains OS-specific packages and prefixes.

## Prerequisites

- Docker installed
- NVIDIA Container Toolkit
- Git

## Build the image

The Dockerfile uses a minimal, Linux-friendly conda environment defined in `docker_environment_py39.yml`.

Steps:

1. Clone the repo
   - `git clone https://github.com/toby43307/voice_persona`
   - `cd voice_persona/VoicePersona`
   
2. Prepare required data/models (place them in your working copy on the host; they will be mounted into the container)
   - `data_util/face_tracking/3DMM/01_MorphableModel.mat` (≈229 MB)
   
     - Basel Face Model. Download from: https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details
   
   - `data_util/face_tracking/3DMM/3DMM_info.npy` (≈190 MB)
   
     - Generate using: `data_util/face_tracking/convert_BFM.py`
   
   - `VoicePersona/data_util/face_parsing/79999_iter.pth` (≈50.8 MB)
   
     - [Download from HuggingFace](https://huggingface.co/afrizalha/musetalk-models/blob/main/face-parse-bisent/79999_iter.pth)  
       **or**  
     - [https://github.com/neuralchen/SimSwap Download from Google Drive](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view)
   
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

## Run the container

On machine with GPU support, run:

- `docker run -it --gpus all -p 5001:5001 voice_persona_py39`

- Then inside the container, complete the pytorch3d installation:
    root@2984306f9a83:/VoicePersona# source /opt/conda/etc/profile.d/conda.sh
    root@2984306f9a83:/VoicePersona# conda activate voicepersona_env
    (voicepersona_env) root@2984306f9a83:/VoicePersona# python --version
    Python 3.9.23
    (voicepersona_env) root@2984306f9a83:/VoicePersona# python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
    2.3.1+cu121 True 12.1
    (voicepersona_env) root@2984306f9a83:/VoicePersona# pip install "git+https://github.com/NVIDIA/pytorch3d.git"

**Note:** Some Python packages listed may not be installed automatically during Docker image creation, depending on your environment or changes in upstream images. If you encounter missing package errors when running the app, manually install them inside the container using:

```bash
pip install -r requirements.txt
```

or install the specific missing package as needed.
  - `pip install openai`
  - `pip install resampy==0.4.3`
  - `pip install python-speech-features==0.6`
  - `pip install tensorflow-cpu==2.10.0`
  - `pip install hyperpyyaml==1.2.3`
  - `pip install modelscope==1.10.0`
  - `pip install onnxruntime==1.16.3`
  - `pip install omegaconf==2.3.0`
  - `pip install conformer==0.3.2`
  - `pip install hydra-core==1.3.2`
  - `pip install wget==3.2`
  - `pip install natsort==8.4.0`

Or directly run:

    pip install tensorflow-cpu==2.10.0 hyperpyyaml==1.2.3 modelscope==1.10.0 onnxruntime==1.16.3 omegaconf==2.3.0 conformer==0.3.2 hydra-core==1.3.2 wget==3.2 natsort==8.4.0 configargparse==1.7.1

or:

    pip install -i https://mirrors.aliyun.com/pypi/simple/ tensorflow-cpu==2.10.0 hyperpyyaml==1.2.3 modelscope==1.10.0 onnxruntime==1.16.3 omegaconf==2.3.0 conformer==0.3.2 hydra-core==1.3.2 wget==3.2 natsort==8.4.0 configargparse==1.7.1


then inside the container, start the app:

    (voicepersona_env) root@fec46a8303ef:/VoicePersona# python VoicePersona/app.py

The app will be available to the host, visit it at `http://localhost:5001`.

## About environment_win_py39.yml

`environment_win_py39.yml` is a Windows-specific environment export. Use it on Windows hosts only (outside Docker):

- Create / update a Windows conda env:
  - `conda env create -f environment_win_py39.yml`
  - or `conda env update -f environment_win_py39.yml`

Do not copy `environment_win_py39.yml` into Docker builds; keep using `docker_environment_py39.yml` for portability.

## Using requirements.txt (pip users)

If you are not using conda, you can install the Python dependencies with pip:

```bash
pip install -r requirements.txt
```

- This will install all Python packages listed in `environment_win_py39.yml` under the `pip:` section.
- System/conda packages (e.g., CUDA, ffmpeg, opencv, etc.) are **not** included in `requirements.txt` and must be installed separately if needed.

### How to check your environment

1. **Check Python version**

   ```bash
   python --version
   # Should be Python 3.9.x
   ```

2. **Check required packages**

   ```bash
   pip list
   # Or use pip check for dependency issues
   pip check
   ```

3. **(Optional) Check CUDA availability for PyTorch**

   ```python
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Check ffmpeg**

   ```bash
   ffmpeg -version
   ```

If you encounter missing system libraries (e.g., ffmpeg, CUDA, OpenCV), install them using your OS package manager or conda.

- For Windows users, use the provided `environment_win_py39.yml` with conda for full compatibility.
- For Linux/Docker, use the Dockerfile and `docker_environment_py39.yml`.

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

