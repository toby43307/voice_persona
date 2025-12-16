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