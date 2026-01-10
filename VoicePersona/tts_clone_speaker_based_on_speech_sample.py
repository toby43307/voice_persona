import sys
import os
from datetime import datetime
from typing import List
import json

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch


# Load speaker profiles from JSON instead of hardcoded map
def load_speaker_profiles() -> dict:
    script_dir = os.path.dirname(__file__)
    profiles_path = os.path.join(script_dir, 'speaker_profiles.json')
    try:
        with open(profiles_path, 'r', encoding='utf-8') as f:
            data = json.load(f) or {}
        # Normalize keys to lowercase for lookup
        return { (k or '').lower(): v for k, v in data.items() }
    except Exception as e:
        print(f"Failed to read speaker_profiles.json: {e}")
        return {}


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def concat_chunks(chunks: List[torch.Tensor]) -> torch.Tensor:
    if not chunks:
        raise ValueError("No audio chunks returned from inference.")
    # Expect shape: [channels, samples]; concatenate along samples dim
    return torch.cat(chunks, dim=1)


def resolve_project_relative(path: str) -> str:
    """Resolve project-relative paths like './asset/xxx.wav' to absolute filesystem path."""
    if not path:
        return ''
    norm = path.replace('\\', '/').strip()
    # If path is already absolute, return as-is
    if os.path.isabs(norm):
        return norm
    # Resolve relative to project root (script directory's parent)
    script_dir = os.path.dirname(__file__)
    project_root = script_dir  # speaker_profiles.json uses paths relative to project root
    return os.path.normpath(os.path.join(project_root, norm))


def main():
    # Read input text and target speaker from command line args
    profiles = load_speaker_profiles()

    # Help message with available speakers (from JSON)
    available = ', '.join(sorted({k.capitalize() for k in profiles.keys()})) if profiles else 'N/A'

    if len(sys.argv) < 3:
        print(
            "Usage: python tts_clone_speaker_based_on_speech_sample.py \"<target_speaker>\" <text to speak>\n"
            f"       target_speaker one of: {available}"
        )
        sys.exit(1)

    target_speaker = sys.argv[1].strip().lower()
    sentence_to_say = sys.argv[2]

    profile = profiles.get(target_speaker)
    if not profile:
        print(f"Unknown target_speaker '{sys.argv[1]}'. Available: {available}")
        sys.exit(1)

    wav_rel = (profile.get('wav') or '').strip()
    sample_text = (profile.get('sample_text') or '').strip()

    wav_path = resolve_project_relative(wav_rel)
    if not wav_rel:
        print(f"No 'wav' path configured for speaker '{sys.argv[1]}'.")
        sys.exit(1)
    if not os.path.isfile(wav_path):
        print(f"Reference wav not found for speaker '{sys.argv[1]}': {wav_rel} -> {wav_path}")
        sys.exit(1)

    if not sample_text:
        print(f"No sample_text configured for speaker '{sys.argv[1]}'.")
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
        sample_text,
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
