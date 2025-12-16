import sys
import os
from datetime import datetime
from typing import List

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def concat_chunks(chunks: List[torch.Tensor]) -> torch.Tensor:
    if not chunks:
        raise ValueError("No audio chunks returned from inference.")
    # Expect shape: [channels, samples]; concatenate along samples dim
    return torch.cat(chunks, dim=1)


def main():
    # Read input text from command line args
    if len(sys.argv) < 2:
        print("Usage: python tts_clone_speaker_based_on_speech_sample.py \"<text to speak>\"")
        sys.exit(1)
    sentence_to_say = sys.argv[1]

    # Prepare output folder and filenames
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    ensure_output_dir(output_dir)
    timestamp_str = datetime.now().strftime('%y%m%d_%H%M%S')
    out_wav_path = os.path.join(output_dir, f"{timestamp_str}_output.wav")
    out_txt_path = os.path.join(output_dir, f"{timestamp_str}_script.txt")

    # Initialize model
    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

    # Prompt and reference sentence
    prompt_speech_16k = load_wav('./asset/Trump_1st_15s.wav', 16000)
    sentence_to_learn = (
        "Where they opened up a lot of different plants, energy plants, energy producing plants and they're doing well. "
        "I give Germany a lot of credit for that. They've said this is a disaster, what's happening. They were going all green."
    )

    # Run zero-shot inference (can return multiple streamed chunks)
    audio_chunks: List[torch.Tensor] = []
    for _, result in enumerate(cosyvoice.inference_zero_shot(
        sentence_to_say,
        sentence_to_learn,
        prompt_speech_16k,
        stream=False
    )):
        # Each result has key 'tts_speech' with Tensor [channels, samples]
        audio_chunks.append(result['tts_speech'])

    # Concatenate chunks to a single waveform
    waveform = concat_chunks(audio_chunks)

    # Save to output files
    torchaudio.save(out_wav_path, waveform, cosyvoice.sample_rate)
    with open(out_txt_path, 'w', encoding='utf-8') as f:
        f.write(sentence_to_say.strip() + "\n")

    print(f"Saved audio: {out_wav_path}")
    print(f"Saved script: {out_txt_path}")


if __name__ == '__main__':
    main()

# test samples
# "We have virtually stopped drugs comining into our country by sea. We call them the water drugs."
# "I don't mind making this speech without a teleprompter because the teleprompter is not working."
