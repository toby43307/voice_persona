@@@ downgrade from py310 to py39

pip install  librosa==0.10.2  soundfile==0.12.1  soxr==1.0.0  audioread==3.1.0  resampy==0.4.3  python-speech-features==0.6  openai-whisper==20231117
  
  pip install  dlib==19.24.1  face-alignment==1.4.1  face-recognition==1.3.0  face-recognition-models==0.3.0  scikit-image==0.24.3  pillow==9.5.0
  
  conda install -c conda-forge   dlib=19.24.1   scikit-image=0.24.3   pillow=9.5.0   face-alignment=1.4.1   face-recognition=1.3.0   face-recognition-models=0.3.0
  
  pip install  transformers==4.51.3  diffusers==0.29.0  safetensors==0.7.0  tokenizers==0.21.4  huggingface-hub==0.36.0
  
  pip install  lightning==2.2.4  pytorch-lightning==2.6.0  torchmetrics==1.8.2  tensorboard==2.14.0  tensorboardx==2.6.4
  
  pip install  fastapi==0.115.6  uvicorn==0.30.0  gradio==4.44.1  flask==2.3.3

  pip install openai==1.30.5
  pip install "httpx<0.27"
  pip install hyperpyyaml==1.2.2
  pip install modelscope==1.10.0
  pip install onnxruntime==1.16.3
  pip install inflect==7.0.0
	
	pip install wetext==0.1.2

    You can patch cosyvoice/cli/frontend.py to bypass imports that fail:
	try:
        import ttsfrd
        # ttsfrd-based normalizer (if needed)
    except ModuleNotFoundError:
        # Fallback: simple identity normalizer
        class DummyNormalizer:
            def normalize(self, text):
                return text
        ZhNormalizer = EnNormalizer = DummyNormalizer()

  pip install omegaconf==2.3.0
  pip install speechbrain

  pip install conformer==0.3.2
  pip install hydra-core==1.3.2
  pip install gdown
  pip install wget
  pip install pyworld
  pip install face-alignment==1.4.1

  pip install tensorflow-cpu==2.10.0



@@@

upgrade torch from 2.3.1 to 2.5.1
	pip uninstall torch torchaudio torchvision -y

	pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

@@@

conda install -c conda-forge opencv ffmpeg

python -m pip install dlib

pip install  python-speech-features==0.6  resampy==0.4.3  fvcore==0.1.5.post20221221  iopath==0.1.10  yacs==0.1.8  tensorboardx==2.6.4  tabulate==0.9.0

pip install  face-recognition==1.3.0  face-recognition-models==0.3.0  face-alignment==1.4.1

(py310_asr_tts) D:\git-repos\CosyVoice>pip uninstall -y onnx onnxruntime numpy

(py310_asr_tts) D:\git-repos\CosyVoice>pip uninstall numpy
Found existing installation: numpy 2.2.6
Uninstalling numpy-2.2.6:

(py310_asr_tts) D:\git-repos\CosyVoice>python -m pip install numpy==1.26.4

(py310_asr_tts) D:\git-repos\CosyVoice>python -m pip install onnx==1.16.0 onnxruntime==1.18.0

python -m pip install flask

@@@ 20251217

(py310_asr_tts) D:\git-repos\VoicePersona>python -m pip install openai

@@@ 20251217 - tensorflow is required for AD-NeRF data processing : step 0; not yet resolved

	(py310_asr_tts) D:\git-repos\VoicePersona>python -m pip install tensorflow

@@@ Docker install


D:\git-repos\VoicePersona>
D:\git-repos\VoicePersona>
D:\git-repos\VoicePersona>docker builder prune -f
ID                                              RECLAIMABLE     SIZE            LAST ACCESSED
wo6kmlsv14f7u54esirive8in*                      true            8.192kB         2 minutes ago
9q4fsf33wgltg6mkgf4lwbpm2*                      true    165.2MB         2 minutes ago
1l5px1amf6o41pvvjygannooy                       true    16.38kB         2 minutes ago
s5m0b4gi6ybvl6y4dvy6v0lvj*                      true    8.192kB         2 minutes ago
fgbkojfgq2hou7pj8zjh6swie                       true    8.192kB         4 minutes ago
p46r21ups3lsi098at464scaa                       true    17.47MB         4 minutes ago
tv0g79x32w02l14kb4n51cqo7                       true    6.263MB         4 minutes ago
8heqgm61yjj12ckt0377cejby                       true    1.861GB         26 minutes ago
t9cnxrstd61vkxgaxe29wm5ud                       true    17.91kB         About an hour ago
kicqekitctjmn3wjvth4wvio6                       true    46.74kB         About an hour ago
qn8fbx20cdwbcim7h6soioh62                       true    346.5kB         About an hour ago
nuil8l9fz78vpfr07zoditppf                       true    3.28GB          About an hour ago
Total:  5.33GB

D:\git-repos\VoicePersona>docker build -t voicepersona .
[+] Building 79.5s (10/14)                                                                                                      docker:desktop-linux
 => [internal] load build definition from Dockerfile                                                                                            0.1s
 => => transferring dockerfile: 1.08kB                                                                                                          0.0s
 => [internal] load metadata for docker.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04                                                        4.2s
 => [internal] load metadata for docker.io/mambaorg/micromamba:1.5.8                                                                            9.1s
 => [auth] mambaorg/micromamba:pull token for registry-1.docker.io                                                                              0.0s
 => [internal] load .dockerignore                                                                                                               0.1s
 => => transferring context: 342B                                                                                                               0.0s
 => [stage-1 1/7] FROM docker.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04@sha256:f3a7fb39fa3ffbe54da713dd2e93063885e5be2f4586a705c39031b  70.0s
 => => resolve docker.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04@sha256:f3a7fb39fa3ffbe54da713dd2e93063885e5be2f4586a705c39031b8284d379a  0.1s
 => => sha256:8fea63e3f8460c272e05808353658b865a48b7729ff3fc8817261b3fd7935b9a 22.02MB / 724.96MB                                              69.0s
 => => sha256:e7016935dd60c632d835fe53b96c59b79194151f22ed555675a41525e066a99f 1.52kB / 1.52kB                                                  1.5s
 => => sha256:fd355de1d1f25492195368f3c3859f24af856e5d7a2ffb34951776daa50bd3e7 63.89kB / 63.89kB                                                1.8s
 => => sha256:3480bb79c6384806f3ae4d8854b5e7ea3e51c3e0ed913965790cdb1ac06cb0c4 1.69kB / 1.69kB                                                  1.1s
 => => sha256:5e5846364eee50e93288b9e4085bc9e558ed543163636c9ca2e61a528cb4952d 15.73MB / 1.29GB                                                67.9s
 => [internal] load build context                                                                                                               3.2s
 => => transferring context: 164.83MB                                                                                                           3.0s
 => [micromamba_stage 1/1] FROM docker.io/mambaorg/micromamba:1.5.8@sha256:475730daef12ff9c0733e70092aeeefdf4c373a584c952dac3f7bdb739601990    69.9s
 => => resolve docker.io/mambaorg/micromamba:1.5.8@sha256:475730daef12ff9c0733e70092aeeefdf4c373a584c952dac3f7bdb739601990                      0.1s
 => => sha256:1002294c9bed4933726f7a34c01d526e5c57a16d1ef0d5e12d144429ef7a4a2c 346B / 346B                                                      0.4s
 => => sha256:38b0c124d9b4103abdce4bfb08bc6e310b4dc3ffb27978d77350e7955ddec8b2 645B / 645B                                                      0.4s
 => => sha256:f14bee2f36c553c162791b50a3033087beeb8efd64c469c55b4b213655f9ce8e 683B / 683B                                                      0.7s
 => => sha256:514e6f58a3981e58d107f52f42ddac4b4630ad6856303762cc8687e81c6411f9 214B / 214B                                                      0.7s
 => => sha256:5ff289cba971d27fd2ed5c6e13cd944739ec9489998da93400bccf66d7c03d91 3.78kB / 3.78kB                                                  1.4s
 => => sha256:27dded3b689263d0fa2c4007a9a20c455f8c57d3a07dea42b747bf8fcae1fbae 283B / 283B                                                      0.7s
 => => sha256:8d1e1fc2982befe8a4222f190deb1dd1a465bdcd39d08e540b227dc2c78671b7 521B / 521B                                                      0.4s
 => => sha256:c53a092f01542f27743b8e5eb6380a3f6c4126e1454bb229adc803c6159b236e 6.06MB / 6.06MB                                                 14.6s
 => => sha256:c9a3dce55fa977c4811896aae460a280c5e10b6babb0166e7df22869d05cca09 122.68kB / 122.68kB                                              3.5s
 => => sha256:e4fff0779e6ddd22366469f08626c3ab1884b5cbe1719b26da238c95f247b305 29.13MB / 29.13MB                                               59.8s
 => => extracting sha256:e4fff0779e6ddd22366469f08626c3ab1884b5cbe1719b26da238c95f247b305                                                       0.8s
 => => extracting sha256:c9a3dce55fa977c4811896aae460a280c5e10b6babb0166e7df22869d05cca09                                                       0.1s
 => => extracting sha256:c53a092f01542f27743b8e5eb6380a3f6c4126e1454bb229adc803c6159b236e                                                       0.1s
 => => extracting sha256:4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1                                                       0.0s
 => => extracting sha256:8d1e1fc2982befe8a4222f190deb1dd1a465bdcd39d08e540b227dc2c78671b7                                                       0.0s
 => => extracting sha256:27dded3b689263d0fa2c4007a9a20c455f8c57d3a07dea42b747bf8fcae1fbae                                                       0.0s
 => => extracting sha256:5ff289cba971d27fd2ed5c6e13cd944739ec9489998da93400bccf66d7c03d91                                                       0.1s
 => => extracting sha256:514e6f58a3981e58d107f52f42ddac4b4630ad6856303762cc8687e81c6411f9                                                       0.0s
 => => extracting sha256:f14bee2f36c553c162791b50a3033087beeb8efd64c469c55b4b213655f9ce8e                                                       0.1s
 => => extracting sha256:38b0c124d9b4103abdce4bfb08bc6e310b4dc3ffb27978d77350e7955ddec8b2                                                       0.0s
 => => extracting sha256:1002294c9bed4933726f7a34c01d526e5c57a16d1ef0d5e12d144429ef7a4a2c                                                       0.0s
 => CANCELED [stage-1 2/7] RUN apt-get update && apt-get install -y     ca-certificates     curl     bzip2     && rm -rf /var/lib/apt/lists/*   0.0s
 => ERROR [stage-1 3/7] COPY --from=micromamba_stage /usr/local/bin/micromamba /usr/local/bin/micromamba                                        0.0s
------
 > [stage-1 3/7] COPY --from=micromamba_stage /usr/local/bin/micromamba /usr/local/bin/micromamba:
------
Dockerfile:15
--------------------
  13 |
  14 |     # Copy micromamba binary safely
  15 | >>> COPY --from=micromamba_stage /usr/local/bin/micromamba /usr/local/bin/micromamba
  16 |
  17 |     # Micromamba config
--------------------
ERROR: failed to build: failed to solve: failed to compute cache key: failed to calculate checksum of ref 5z5j6ms3q9pfjc04glhlc0giu::nrt49yw5j5u9v4rthjzjw6vh8: "/usr/local/bin/micromamba": not found

D:\git-repos\VoicePersona>docker build -t voicepersona .
[+] Building 7025.6s (8/15)                                                                                                                                         docker:desktop-linux
 => [internal] load build definition from Dockerfile                                                                                                                                0.0s
 => => transferring dockerfile: 1.06kB                                                                                                                                              0.0s
 => [internal] load metadata for docker.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04                                                                                            3.1s
 => [internal] load metadata for docker.io/mambaorg/micromamba:1.5.8                                                                                                                3.0s
 => [auth] nvidia/cuda:pull token for registry-1.docker.io                                                                                                                          0.0s
 => [auth] mambaorg/micromamba:pull token for registry-1.docker.io                                                                                                                  0.0s
 => [internal] load .dockerignore                                                                                                                                                   0.0s
 => => transferring context: 342B                                                                                                                                                   0[+] Building 7025.7s (8/15)                                                                                                                                     docker:desktop-linux
 => [internal] load build definition from Dockerfile                                                                                                                            0.0s2
 => => transferring dockerfile: 1.06kB                                                                                                                                          0.0s
 => [internal] load metadata for docker.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04                                                                                        3.1s0
 => [internal] load metadata for docker.io/mambaorg/micromamba:1.5.8                                                                                                            3.0s
 => [auth] nvidia/cuda:pull token for registry-1.docker.io                                                                                                                      0.0s0
 => [auth] mambaorg/micromamba:pull token for registry-1.docker.io                                                                                                              0.0s
 => [internal] load .dockerignore                                                                                                                                               0.0s2
 => => transferring context: 342B                                                                                                                                        [+] Building 7025.9s (8/15)                                                                                                                         docker:desktop-linux
 => [internal] load build definition from Dockerfile                                                                                                                0.0s
 => => transferring dockerfile: 1.06kB                                                                                                                              0.0s
 => [internal] load metadata for docker.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04                                                                            3.1s
 => [internal] load metadata for docker.io/mambaorg/micromamba:1.5.8                                                                                                3.0s
 => [auth] nvidia/cuda:pull token for registry-1.docker.io                                                                                                          0.0s
 => [auth] mambaorg/micromamba:pull token for registry-1.docker.io                                                                                                  0.0s
 => [internal] load .dockerignore                                                                                                                                   0.0s
 => => transferring context: 342B                                                                                                                                [+] Building 7026.0s (8/15)                                                                                                                 docker:desktop-linu[+] Buildi[+] Buil[+] Building 9964.9s (14/15)                                                                                                                        docker:desktop-linux
 => [internal] load build definition from Dockerfile                                                                                                                0.0s
 => => transferring dockerfile: 1.06kB                                                                                                                              0.0s
 => [internal] load metadata for docker.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04                                                                            3.1s
 => [internal] load metadata for docker.io/mambaorg/micromamba:1.5.8                                                                                                3.0s
 => [auth] nvidia/cuda:pull token for registry-1.docker.io                                                                                                          0.0s
 => [auth] mambaorg/micromamba:pull token for registry-1.docker.io                                                                                                  0.0s
 => [internal] load .dockerignore                                                                                                                                   0.0s
 => => transferring context: 342B                                                                                                                                   0.0s
 => [stage-1 1/7] FROM docker.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04@sha256:f3a7fb39fa3ffbe54da713dd2e93063885e5be2f4586a705c39031b8284d379a           8221.8s
 => => resolve docker.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04@sha256:f3a7fb39fa3ffbe54da713dd2e93063885e5be2f4586a705c39031b8284d379a                      0.1s
 => => sha256:8fea63e3f8460c272e05808353658b865a48b7729ff3fc8817261b3fd7935b9a 724.96MB / 724.96MB                                                               4010.8s
 => => sha256:5e5846364eee50e93288b9e4085bc9e558ed543163636c9ca2e61a528cb4952d 1.29GB / 1.29GB                                                                   8209.9s
 => => extracting sha256:5e5846364eee50e93288b9e4085bc9e558ed543163636c9ca2e61a528cb4952d                                                                           7.5s
 => => extracting sha256:fd355de1d1f25492195368f3c3859f24af856e5d7a2ffb34951776daa50bd3e7                                                                           0.1s
 => => extracting sha256:3480bb79c6384806f3ae4d8854b5e7ea3e51c3e0ed913965790cdb1ac06cb0c4                                                                           0.0s
 => => extracting sha256:e7016935dd60c632d835fe53b96c59b79194151f22ed555675a41525e066a99f                                                                           0.0s
 => => extracting sha256:8fea63e3f8460c272e05808353658b865a48b7729ff3fc8817261b3fd7935b9a                                                                           4.1s
 => [internal] load build context                                                                                                                                   0.1s
 => => transferring context: 1.49MB                                                                                                                                 0.1s
 => CACHED [micromamba_stage 1/1] FROM docker.io/mambaorg/micromamba:1.5.8@sha256:475730daef12ff9c0733e70092aeeefdf4c373a584c952dac3f7bdb739601990                  0.1s
 => => resolve docker.io/mambaorg/micromamba:1.5.8@sha256:475730daef12ff9c0733e70092aeeefdf4c373a584c952dac3f7bdb739601990                                          0.1s
 => [stage-1 2/7] RUN apt-get update && apt-get install -y     ca-certificates     curl     bzip2     && rm -rf /var/lib/apt/lists/*                             1473.5s
 => [stage-1 3/7] COPY --from=micromamba_stage /bin/micromamba /usr/local/bin/micromamba                                                                            0.2s
 => [stage-1 4/7] WORKDIR /VoicePersona                                                                                                                             0.1s
 => [stage-1 5/7] COPY docker_environment.yml .                                                                                                                     0.1s
 => ERROR [stage-1 6/7] RUN micromamba create -y -n voicepersona_env -f docker_environment.yml &&     micromamba clean --all --yes                                265.6s
------
 > [stage-1 6/7] RUN micromamba create -y -n voicepersona_env -f docker_environment.yml &&     micromamba clean --all --yes:
265.4 error    libmamba Could not solve for environment specs
265.4     The following packages are incompatible
265.4     ├─ _libavif_api 1.1.1**  does not exist (perhaps a typo or a missing channel);
265.4     ├─ ffmpeg 7.1.0**  is installable with the potential options
265.4     │  ├─ ffmpeg 7.1.0 would require
265.4     │  │  └─ libopenvino-auto-batch-plugin >=2024.6.0,<2024.6.1.0a0 , which can be installed;
265.4     │  ├─ ffmpeg 7.1.0 would require
265.4     │  │  └─ libopenvino-auto-batch-plugin >=2024.4.0,<2024.4.1.0a0 , which can be installed;
265.4     │  ├─ ffmpeg 7.1.0 would require
265.4     │  │  └─ libopenvino-auto-batch-plugin >=2024.5.0,<2024.5.1.0a0 , which can be installed;
265.4     │  └─ ffmpeg 7.1.0 would require
265.4     │     └─ liblzma >=5.6.4,<6.0a0  with the potential options
265.4     │        ├─ liblzma 5.8.1 would require
265.4     │        │  └─ xz ==5.8.1 *_0, which can be installed;
265.4     │        ├─ liblzma 5.8.1 would require
265.4     │        │  └─ xz 5.8.1.* , which can be installed;
265.4     │        └─ liblzma 5.6.4 would require
265.4     │           └─ xz ==5.6.4 *_0, which can be installed;
265.4     ├─ harfbuzz 11.0.0**  is requested and can be installed;
265.4     ├─ icu 75.1**  is requested and can be installed;
265.4     ├─ khronos-opencl-icd-loader 2024.10.24**  does not exist (perhaps a typo or a missing channel);
265.4     ├─ libabseil 20240722.0**  is requested and can be installed;
265.4     ├─ libasprintf 0.22.5**  is requested and can be installed;
265.4     ├─ libclang13 21.1.7**  is installable and it requires
265.4     │  └─ libllvm21 >=21.1.7,<21.2.0a0 , which requires
265.4     │     └─ libxml2-16 >=2.14.6  with the potential options
265.4     │        ├─ libxml2-16 2.14.6 would require
265.4     │        │  ├─ liblzma >=5.8.1,<6.0a0 , which can be installed (as previously explained);
265.4     │        │  └─ libxml2 2.14.6 , which can be installed;
265.4     │        ├─ libxml2-16 [2.14.6|2.15.0|2.15.1] would require
265.4     │        │  ├─ icu <0.0a0 , which conflicts with any installable versions previously reported;
265.4     │        │  └─ liblzma >=5.8.1,<6.0a0 , which can be installed (as previously explained);
265.4     │        ├─ libxml2-16 2.15.0 would require
265.4     │        │  ├─ liblzma >=5.8.1,<6.0a0 , which can be installed (as previously explained);
265.4     │        │  └─ libxml2 2.15.0 , which can be installed;
265.4     │        └─ libxml2-16 2.15.1 would require
265.4     │           ├─ liblzma >=5.8.1,<6.0a0 , which can be installed (as previously explained);
265.4     │           └─ libxml2 2.15.1 , which can be installed;
265.4     ├─ libgettextpo 0.22.5**  is requested and can be installed;
265.4     ├─ libglib 2.84.0**  is requested and can be installed;
265.4     ├─ libhwloc 2.12.1**  is installable with the potential options
265.4     │  ├─ libhwloc 2.12.1 would require
265.4     │  │  └─ __cuda, which is missing on the system;
265.4     │  ├─ libhwloc 2.12.1 would require
265.4     │  │  └─ libxml2 >=2.13.8,<2.14.0a0  but there are no viable options
265.4     │  │     ├─ libxml2 [2.13.8|2.13.9] would require
265.4     │  │     │  └─ liblzma >=5.8.1,<6.0a0 , which can be installed (as previously explained);
265.4     │  │     ├─ libxml2 2.13.9 would require
265.4     │  │     │  ├─ icu <0.0a0 , which conflicts with any installable versions previously reported;
265.4     │  │     │  └─ liblzma >=5.8.1,<6.0a0 , which can be installed (as previously explained);
265.4     │  │     └─ libxml2 [2.13.8|2.13.9] would require
265.4     │  │        └─ icu >=73.1,<74.0a0 , which conflicts with any installable versions previously reported;
265.4     │  └─ libhwloc 2.12.1 would require
265.4     │     └─ libxml2-16 [>=2.14.5 |>=2.14.6 ] with the potential options
265.4     │        ├─ libxml2-16 2.14.6, which can be installed (as previously explained);
265.4     │        ├─ libxml2-16 [2.14.6|2.15.0|2.15.1], which cannot be installed (as previously explained);
265.4     │        ├─ libxml2-16 2.15.0, which can be installed (as previously explained);
265.4     │        ├─ libxml2-16 2.15.1, which can be installed (as previously explained);
265.4     │        └─ libxml2-16 2.14.5 would require
265.4     │           └─ liblzma >=5.8.1,<6.0a0 , which can be installed (as previously explained);
265.4     ├─ libintl 0.22.5**  does not exist (perhaps a typo or a missing channel);
265.4     ├─ libopencv 4.11.0**  is installable with the potential options
265.4     │  ├─ libopencv 4.11.0 would require
265.4     │  │  ├─ libopenvino >=2024.6.0,<2024.6.1.0a0 , which can be installed;
265.4     │  │  └─ libopenvino-ir-frontend >=2024.6.0,<2024.6.1.0a0 , which requires
265.4     │  │     └─ pugixml >=1.14,<1.15.0a0 , which can be installed;
265.4     │  ├─ libopencv 4.11.0 would require
265.4     │  │  ├─ harfbuzz >=11.0.1 , which conflicts with any installable versions previously reported;
265.4     │  │  └─ libglib >=2.84.2,<3.0a0 , which conflicts with any installable versions previously reported;
265.4     │  ├─ libopencv 4.11.0 would require
265.4     │  │  └─ ffmpeg >=7.1.1,<8.0a0  with the potential options
265.4     │  │     ├─ ffmpeg 7.1.1 would require
265.4     │  │     │  ├─ openh264 >=2.6.0,<2.6.1.0a0 , which can be installed;
265.4     │  │     │  └─ svt-av1 >=3.0.2,<3.0.3.0a0 , which can be installed;
265.4     │  │     ├─ ffmpeg 7.1.1 would require
265.4     │  │     │  ├─ harfbuzz >=11.0.1 , which conflicts with any installable versions previously reported;
265.4     │  │     │  └─ svt-av1 >=3.0.2,<3.0.3.0a0 , which can be installed;
265.4     │  │     ├─ ffmpeg 7.1.1 would require
265.4     │  │     │  ├─ harfbuzz >=11.4.3 , which conflicts with any installable versions previously reported;
265.4     │  │     │  └─ svt-av1 >=3.1.2,<3.1.3.0a0 , which can be installed;
265.4     │  │     ├─ ffmpeg 7.1.1 would require
265.4     │  │     │  ├─ harfbuzz >=11.4.5 , which conflicts with any installable versions previously reported;
265.4     │  │     │  └─ svt-av1 >=3.1.2,<3.1.3.0a0 , which can be installed;
265.4     │  │     ├─ ffmpeg 7.1.1 would require
265.4     │  │     │  ├─ openh264 >=2.6.0,<2.6.1.0a0 , which can be installed;
265.4     │  │     │  └─ svt-av1 >=3.0.1,<3.0.2.0a0 , which can be installed;
265.4     │  │     ├─ ffmpeg 7.1.1 would require
265.4     │  │     │  ├─ openh264 >=2.6.0,<2.6.1.0a0 , which can be installed;
265.4     │  │     │  └─ svt-av1 >=3.0.0,<3.0.1.0a0 , which can be installed;
265.4     │  │     ├─ ffmpeg 7.1.1 would require
265.4     │  │     │  ├─ harfbuzz >=11.5.1 , which conflicts with any installable versions previously reported;
265.4     │  │     │  └─ svt-av1 >=3.1.2,<3.1.3.0a0 , which can be installed;
265.4     │  │     └─ ffmpeg 7.1.1 would require
265.4     │  │        ├─ harfbuzz >=11.4.3 , which conflicts with any installable versions previously reported;
265.4     │  │        └─ svt-av1 >=3.1.1,<3.1.2.0a0 , which can be installed;
265.4     │  ├─ libopencv 4.11.0 would require
265.4     │  │  ├─ harfbuzz >=11.1.0 , which conflicts with any installable versions previously reported;
265.4     │  │  └─ libglib >=2.84.1,<3.0a0 , which conflicts with any installable versions previously reported;
265.4     │  ├─ libopencv 4.11.0 would require
265.4     │  │  ├─ harfbuzz >=11.0.1 , which conflicts with any installable versions previously reported;
265.4     │  │  └─ libglib >=2.84.1,<3.0a0 , which conflicts with any installable versions previously reported;
265.4     │  ├─ libopencv 4.11.0 would require
265.4     │  │  ├─ libasprintf >=0.23.1,<1.0a0 , which conflicts with any installable versions previously reported;
265.4     │  │  └─ libgettextpo >=0.23.1,<1.0a0 , which conflicts with any installable versions previously reported;
265.4     │  └─ libopencv 4.11.0 would require
265.4     │     └─ harfbuzz >=11.0.1 , which conflicts with any installable versions previously reported;
265.4     ├─ libopenvino-auto-batch-plugin 2025.0.0**  is not installable because it conflicts with any installable versions previously reported;
265.4     ├─ libopenvino-onnx-frontend 2025.0.0**  is not installable because there are no viable options
265.4     │  ├─ libopenvino-onnx-frontend 2025.0.0 would require
265.4     │  │  └─ libabseil >=20250127.0,<20250128.0a0 , which conflicts with any installable versions previously reported;
265.4     │  ├─ libopenvino-onnx-frontend 2025.0.0 would require
265.4     │  │  └─ libopenvino 2025.0.0 hac27bb2_0, which conflicts with any installable versions previously reported;
265.4     │  ├─ libopenvino-onnx-frontend 2025.0.0 would require
265.4     │  │  └─ libopenvino 2025.0.0 hdc3f47d_1, which conflicts with any installable versions previously reported;
265.4     │  └─ libopenvino-onnx-frontend 2025.0.0 would require
265.4     │     └─ libopenvino 2025.0.0 hdc3f47d_2, which conflicts with any installable versions previously reported;
265.4     ├─ libopenvino 2025.0.0**  is installable with the potential options
265.4     │  ├─ libopenvino 2025.0.0 conflicts with any installable versions previously reported;
265.4     │  ├─ libopenvino 2025.0.0 conflicts with any installable versions previously reported;
265.4     │  ├─ libopenvino 2025.0.0 conflicts with any installable versions previously reported;
265.4     │  └─ libopenvino 2025.0.0, which can be installed;
265.4     ├─ libwinpthread 12.0.0.r4.gg4f2fc60ca**  does not exist (perhaps a typo or a missing channel);
265.4     ├─ libxml2 2.13.9** , which cannot be installed (as previously explained);
265.4     ├─ openh264 2.5.0**  is not installable because it conflicts with any installable versions previously reported;
265.4     ├─ pugixml 1.15**  is not installable because it conflicts with any installable versions previously reported;
265.4     ├─ py-opencv 4.11.0**  is installable with the potential options
265.4     │  ├─ py-opencv 4.11.0 would require
265.4     │  │  └─ libopencv [4.11.0 headless_py310h02f0763_0|4.11.0 headless_py311hae1f5d3_0|...|4.11.0 qt6_py39h47cf047_600], which can be installed (as previously explained);
265.4     │  ├─ py-opencv 4.11.0 would require
265.4     │  │  └─ libopencv [4.11.0 headless_py310hc866e42_1|4.11.0 headless_py310hc866e42_2|...|4.11.0 qt6_py39hd96f159_602], which cannot be installed (as previously explained);
265.4     │  ├─ py-opencv 4.11.0 would require
265.4     │  │  └─ libopencv [4.11.0 headless_py310h8ace835_4|4.11.0 headless_py310hf33b295_3|...|4.11.0 qt6_py39hbfaaa73_603], which can be installed (as previously explained);
265.4     │  ├─ py-opencv 4.11.0 would require
265.4     │  │  └─ libopencv [4.11.0 headless_py310h8ace835_5|4.11.0 headless_py311h518f0c0_5|...|4.11.0 qt6_py39h1a5cd75_605], which cannot be installed (as previously explained);
265.4     │  ├─ py-opencv 4.11.0 would require
265.4     │  │  └─ libopencv [4.11.0 headless_py310hada27fe_6|4.11.0 headless_py310hada27fe_7|...|4.11.0 qt6_py39h2c622e1_607], which cannot be installed (as previously explained);
265.4     │  └─ py-opencv 4.11.0 would require
265.4     │     └─ libopencv [4.11.0 headless_py310h52fee8e_8|4.11.0 headless_py310h52fee8e_9|...|4.11.0 qt6_py39h8df4609_609], which cannot be installed (as previously explained);
265.4     ├─ qt6-main 6.8.3**  is installable and it requires
265.4     │  └─ icu >=75.1,<76.0a0 , which can be installed;
265.4     ├─ svt-av1 2.3.0**  is not installable because it conflicts with any installable versions previously reported;
265.4     ├─ ucrt 10.0.22621.0**  does not exist (perhaps a typo or a missing channel);
265.4     ├─ vc 14.3**  does not exist (perhaps a typo or a missing channel);
265.4     ├─ vc14_runtime 14.44.35208**  does not exist (perhaps a typo or a missing channel);
265.4     ├─ vs2015_runtime 14.44.35208**  does not exist (perhaps a typo or a missing channel);
265.4     └─ xz 5.6.4**  is not installable because it conflicts with any installable versions previously reported.
265.5 critical libmamba Could not solve for environment specs
------
Dockerfile:31
--------------------
  30 |     # Create conda environment
  31 | >>> RUN micromamba create -y -n voicepersona_env -f docker_environment.yml && \
  32 | >>>     micromamba clean --all --yes
  33 |
--------------------
ERROR: failed to build: failed to solve: process "/bin/bash -c micromamba create -y -n voicepersona_env -f docker_environment.yml &&     micromamba clean --all --yes" did not complete successfully: exit code: 1

D:\git-repos\VoicePersona>
D:\git-repos\VoicePersona>conda env export --no-builds > docker_environment_portable.yml
'conda' is not recognized as an internal or external command,
operable program or batch file.