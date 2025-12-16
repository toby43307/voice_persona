import sys
sys.path.append('third_party/Matcha-TTS')
print(sys.path)
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav('./asset/Trump_1st_15s.wav', 16000)
#sentence = "We have virtually stopped drugs comining into our country by sea. We call them the water drugs."
sentence = "I don't mind making this speech without a teleprompter because the teleprompter is not working."
for i, j in enumerate(cosyvoice.inference_zero_shot(sentence, "were they opened up a lot of different plants, energy plants, energy producing plants and they're doing well. I give Germany a lot of credit for that. They've said this is a disaster, what's happening, They were going all green. all", prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_Trump_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)