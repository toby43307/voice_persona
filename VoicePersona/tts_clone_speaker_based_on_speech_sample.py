import sys
import os
from datetime import datetime
from typing import List

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch


# Map target speakers to their reference wav paths and sample text
SPEAKER_PROFILE_MAP = {
    'obama': {
        'wav': './asset/obama_1st_15s.wav',
        'sample_text': (
            "Speak with you about the battle we're waging against an oil spill that is assaulting our shores and our citizens."
            "On April twentieth, an explosion ripped through BP deep water horizon drilling rig, about forty miles off"
        ),
    },
    'trump': {
        'wav': './asset/trump_1st_15s.wav',
        'sample_text': (
            "Where they opened up a lot of different plants, energy plants, energy producing plants and they're doing well. "
            "I give Germany a lot of credit for that. They've said this is a disaster, what's happening. They were going all green."
        ),
    },
    'achu': {
        'wav': './asset/test.wav',
        'sample_text': (
            "This is a test sample used for conditioning the voice cloning model for the target speaker."
        ),
    },
}


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def concat_chunks(chunks: List[torch.Tensor]) -> torch.Tensor:
    if not chunks:
        raise ValueError("No audio chunks returned from inference.")
    # Expect shape: [channels, samples]; concatenate along samples dim
    return torch.cat(chunks, dim=1)


def main():
    # Read input text and target speaker from command line args
    if len(sys.argv) < 3:
        available = ', '.join(sorted({k.capitalize() for k in SPEAKER_PROFILE_MAP.keys()}))
        print(
            "Usage: python tts_clone_speaker_based_on_speech_sample.py \"<target_speaker>\" <text to speak>\n"
            f"       target_speaker one of: {available}"
        )
        sys.exit(1)

    target_speaker = sys.argv[1].strip().lower()
    sentence_to_say = sys.argv[2]

    profile = SPEAKER_PROFILE_MAP.get(target_speaker)
    if not profile:
        available = ', '.join(sorted({k.capitalize() for k in SPEAKER_PROFILE_MAP.keys()}))
        print(f"Unknown target_speaker '{sys.argv[2]}'. Available: {available}")
        sys.exit(1)

    wav_path = profile.get('wav')
    if not wav_path or not os.path.isfile(wav_path):
        print(f"Reference wav not found for speaker '{sys.argv[2]}': {wav_path}")
        sys.exit(1)

    sentence_to_learn = profile.get('sample_text', '').strip()
    if not sentence_to_learn:
        print(f"No sample_text configured for speaker '{sys.argv[2]}'.")
        sys.exit(1)

    # Prepare output folder and filenames
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    ensure_output_dir(output_dir)
    timestamp_str = datetime.now().strftime('%y%m%d_%H%M%S')
    out_wav_path = os.path.join(output_dir, f"{timestamp_str}_output.wav")
    out_txt_path = os.path.join(output_dir, f"{timestamp_str}_script.txt")

    # Initialize model
    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

    # Load wave file accordingly based on target speaker
    prompt_speech_16k = load_wav(wav_path, 16000)

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
# "We have virtually stopped drugs coming into our country by sea. We call them the water drugs."
# "I don't mind making this speech without a teleprompter because the teleprompter is not working."
