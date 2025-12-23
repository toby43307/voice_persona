from flask import Flask, render_template, request, jsonify, send_from_directory
import subprocess
import os
import json
import glob
import sys
from openai import OpenAI
import requests
import time
import shutil

app = Flask(__name__)

# Resolve project root (repo root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
ASSET_DIR = os.path.join(PROJECT_ROOT, 'asset')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
DATASET_VIDS_DIR = os.path.join(DATASET_DIR, 'vids')
SPEAKER_PROFILES_PATH = os.path.join(PROJECT_ROOT, 'speaker_profiles.json')
LOCAL_SETTINGS_PATH = os.path.join(PROJECT_ROOT, 'local_settings.json')

# Load secrets/settings from external untracked file or environment
def _load_local_settings():
    settings = {}
    try:
        if os.path.isfile(LOCAL_SETTINGS_PATH):
            with open(LOCAL_SETTINGS_PATH, 'r', encoding='utf-8') as f:
                settings = json.load(f) or {}
    except Exception:
        settings = {}
    return settings

_local = _load_local_settings()
API_KEY = os.getenv('DEEPSEEK_API_KEY') or _local.get('deepseek_api_key') or ''
PUBLIC_CLONE_IP = os.getenv('VOICEPERSONA_CLONE_IP') or _local.get('clone_public_ip') or ''

# OpenAI client init using external config
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com/v1"
)


# 1. 主页
@app.route('/')
def index():
    return render_template('index.html')

# 2. 训练模型页面
@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        print("开始训练...", request.form)
        return jsonify({"status": "success", "message": "Training Started!"})
    return render_template('train.html')

# Static serving for generated output files
@app.route('/media/output/<path:filename>')
def media_output(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)

# Static serving for asset reference audio/video files
@app.route('/media/asset/<path:filename>')
def media_asset(filename: str):
    return send_from_directory(ASSET_DIR, filename)

# Static serving for dataset files (read-only)
@app.route('/media/dataset/<path:relpath>')
def media_dataset(relpath: str):
    # Only serve within dataset directory
    safe_path = os.path.normpath(os.path.join(DATASET_DIR, relpath))
    if not safe_path.startswith(DATASET_DIR):
        return jsonify({"error": "invalid dataset path"}), 400
    # Split directory and filename for send_from_directory
    directory, filename = os.path.split(safe_path)
    return send_from_directory(directory, filename)

# Serve speaker profiles JSON for frontend
@app.route('/config/speakers')
def config_speakers():
    try:
        with open(SPEAKER_PROFILES_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: add a new speaker by uploading a video and sample text
@app.route('/api/add_speaker', methods=['POST'])
def api_add_speaker():
    try:
        name = (request.form.get('name') or '').strip()
        sample_text = (request.form.get('sample_text') or '').strip()
        file = request.files.get('video')
        if not name:
            return jsonify({"status": "error", "message": "name is required"}), 400
        if not file:
            return jsonify({"status": "error", "message": "video file is required"}), 400

        # Build safe base name for files and dict key
        def _safe_base(s: str) -> str:
            s = s.strip().lower()
            # keep unicode letters/digits, replace others with underscore
            return ''.join(ch if ch.isalnum() else '_' for ch in s) or 'speaker'

        base = _safe_base(name)
        os.makedirs(ASSET_DIR, exist_ok=True)
        os.makedirs(DATASET_VIDS_DIR, exist_ok=True)
        video_filename = f"{base}.mp4"
        wav_filename = f"{base}.wav"
        video_path = os.path.join(ASSET_DIR, video_filename)
        wav_path = os.path.join(ASSET_DIR, wav_filename)
        dataset_video_path = os.path.join(DATASET_VIDS_DIR, video_filename)

        # Save uploaded video (overwrite if exists) under asset/
        file.save(video_path)
        # Also save a copy under dataset/vids/
        try:
            shutil.copyfile(video_path, dataset_video_path)
        except Exception as e:
            # Non-fatal: continue, but report in response
            copy_error = str(e)
        else:
            copy_error = None

        # Extract 16kHz mono PCM WAV using ffmpeg
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',
            '-ac', '1',
            '-ar', '16000',
            '-c:a', 'pcm_s16le',
            wav_path
        ]
        try:
            print('FFmpeg extract audio cmd:', ' '.join(ffmpeg_cmd))
            subprocess.run(ffmpeg_cmd, cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as e:
            return jsonify({"status": "error", "message": f"ffmpeg failed: {e}"}), 500

        # Update speaker_profiles.json
        profile_entry = {
            "wav": f"./asset/{wav_filename}",
            "sample_text": sample_text,
            "video": f"./asset/{video_filename}"
        }
        try:
            profiles = {}
            if os.path.isfile(SPEAKER_PROFILES_PATH):
                with open(SPEAKER_PROFILES_PATH, 'r', encoding='utf-8') as f:
                    profiles = json.load(f) or {}
            profiles[base] = profile_entry
            with open(SPEAKER_PROFILES_PATH, 'w', encoding='utf-8') as f:
                json.dump(profiles, f, ensure_ascii=False, indent=2)
        except Exception as e:
            return jsonify({"status": "error", "message": f"failed to update profiles: {e}"}), 500

        resp = {
            "status": "success",
            "speaker": base,
            "profile": profile_entry,
            "video_url": f"/media/asset/{video_filename}",
            "wav_url": f"/media/asset/{wav_filename}"
        }
        if copy_error:
            resp["note"] = f"saved to asset/, but failed to copy to dataset/vids: {copy_error}"
        else:
            resp["dataset_video_url"] = f"/media/dataset/vids/{video_filename}"
        return jsonify(resp)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# API: list all head and torso checkpoints for a dataset
@app.route('/api/checkpoints')
def api_checkpoints():
    dataset = (request.args.get('dataset') or '').strip()
    if not dataset:
        return jsonify({"error": "dataset is required"}), 400
    # Normalize and build absolute path
    rel_path = dataset.replace('\\', '/').lstrip('/')
    base_path = os.path.normpath(os.path.join(PROJECT_ROOT, rel_path))
    # Safety: ensure within project root
    if not base_path.startswith(PROJECT_ROOT):
        return jsonify({"error": "invalid dataset path"}), 400
    logs_dir = os.path.join(base_path, 'logs')

    # Specific subfolders based on naming convention
    head_dir = os.path.join(logs_dir, f"{os.path.basename(rel_path)}_head")
    torso_dir = os.path.join(logs_dir, f"{os.path.basename(rel_path)}_com")

    def list_files(search_dir, pattern):
        if not os.path.isdir(search_dir):
            return []
        files = glob.glob(os.path.join(search_dir, pattern))
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return [os.path.relpath(p, PROJECT_ROOT).replace('\\', '/') for p in files]

    head_list = list_files(head_dir, "*_head.tar")   # e.g., dataset/Obama/logs/Obama_head/*_head.tar
    torso_list = list_files(torso_dir, "*_body.tar") # e.g., dataset/Obama/logs/Obama_com/*_body.tar

    return jsonify({
        "head": head_list,
        "torso": torso_list
    })

# API: extract DeepSpeech features for a given audio path
@app.route('/api/extract_ds_features', methods=['POST'])
def api_extract_ds_features():
    audio_path = (request.form.get('audio_path')
                  or (request.json.get('audio_path') if request.is_json and request.json else None) or '').strip()
    if not audio_path:
        return jsonify({"status": "error", "message": "audio_path is required"}), 400

    def resolve_fs_path(path: str) -> str:
        norm = path.replace('\\', '/')
        if norm.startswith('/media/asset/'):
            rel = norm[len('/media/asset/'):]
            return os.path.join(ASSET_DIR, rel)
        if norm.startswith('/media/output/'):
            rel = norm[len('/media/output/'):]
            return os.path.join(OUTPUT_DIR, rel)
        if norm.startswith('/'):
            return os.path.join(PROJECT_ROOT, norm.lstrip('/'))
        return os.path.join(PROJECT_ROOT, norm)

    fs_path = resolve_fs_path(audio_path)
    if not os.path.isfile(fs_path):
        return jsonify({"status": "error", "message": f"audio file not found: {fs_path}"}), 400

    # Build output npy path (same folder, same stem)
    file_stem, _ = os.path.splitext(fs_path)
    out_npy = file_stem + '.npy'

    script_path = os.path.join(PROJECT_ROOT, 'data_util', 'deepspeech_features', 'extract_ds_features.py')
    try:
        cmd = ['python', script_path, '--input', fs_path, '--output', out_npy]
        print('Extract DS cmd:', ' '.join(cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": f"extract_ds_features failed: {e}"}), 500

    # Build a URL for the npy if under known served dirs
    if out_npy.startswith(ASSET_DIR):
        rel = os.path.relpath(out_npy, ASSET_DIR).replace('\\', '/')
        npy_url = f"/media/asset/{rel}"
    elif out_npy.startswith(OUTPUT_DIR):
        rel = os.path.relpath(out_npy, OUTPUT_DIR).replace('\\', '/')
        npy_url = f"/media/output/{rel}"
    else:
        npy_url = None

    return jsonify({"status": "success", "npy_path": npy_url or out_npy})

# API: generate video using TorsoNeRF test with aud_file npy
@app.route('/api/generate_video', methods=['POST'])
def api_generate_video():
    speaker = (request.form.get('speaker')
               or (request.json.get('speaker') if request.is_json and request.json else None) or '').strip()
    npy_path = (request.form.get('npy_path')
                or (request.json.get('npy_path') if request.is_json and request.json else None) or '').strip()
    if not speaker:
        return jsonify({"status": "error", "message": "speaker is required"}), 400
    if not npy_path:
        return jsonify({"status": "error", "message": "npy_path is required"}), 400

    sp_cap = speaker.lower().capitalize()
    dataset_rel = f"dataset/{sp_cap}"

    def resolve_fs_path(path: str) -> str:
        norm = path.replace('\\', '/')
        if norm.startswith('/media/asset/'):
            rel = norm[len('/media/asset/'):]
            return os.path.join(ASSET_DIR, rel)
        if norm.startswith('/media/output/'):
            rel = norm[len('/media/output/'):]
            return os.path.join(OUTPUT_DIR, rel)
        if norm.startswith('/media/dataset/'):
            rel = norm[len('/media/dataset/'):]
            return os.path.join(DATASET_DIR, rel)
        if norm.startswith('/'):
            return os.path.join(PROJECT_ROOT, norm.lstrip('/'))
        return os.path.join(PROJECT_ROOT, norm)

    npy_fs = resolve_fs_path(npy_path)
    if not os.path.isfile(npy_fs):
        return jsonify({"status": "error", "message": f"npy file not found: {npy_fs}"}), 400

    # Build command to run TorsoNeRF
    config_rel = os.path.join(dataset_rel, 'TorsoNeRFTest_config.txt')
    cmd = ['python', 'NeRFs/TorsoNeRF/run_nerf.py', '--config', config_rel, '--aud_file', npy_fs, '--test_size', '-1']
    print('Generate video cmd:', ' '.join(cmd))

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": f"Video generation failed: {e}"}), 500

    # Check for result.avi
    logs_dir = os.path.join(PROJECT_ROOT, dataset_rel, 'logs')
    out_dir = os.path.join(logs_dir, f"{sp_cap}_com", 'test_aud_rst')
    result_avi = os.path.join(out_dir, 'result.avi')
    if not os.path.isfile(result_avi):
        return jsonify({"status": "error", "message": f"result video not found: {result_avi}"}), 500

    # Convert AVI to MP4 with ffmpeg and save to OUTPUT_DIR as result.mp4 (overwrite if exists)
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        mp4_name = "result.mp4"
        result_mp4 = os.path.join(OUTPUT_DIR, mp4_name)
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', result_avi,
            '-vcodec', 'libx264',
            '-acodec', 'aac',
            '-movflags', 'faststart',
            result_mp4
        ]
        print('FFmpeg convert cmd:', ' '.join(ffmpeg_cmd))
        subprocess.run(ffmpeg_cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": f"ffmpeg conversion failed: {e}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"conversion error: {e}"}), 500

    # Return URL to MP4 under /media/output
    video_url = f"/media/output/{mp4_name}"
    return jsonify({"status": "success", "video_path": video_url})

# API: run training based on selection
@app.route('/api/train', methods=['POST'])
def api_train():
    train_type = (request.form.get('train_type')
                  or (request.json.get('train_type') if request.is_json and request.json else None) or '').strip()
    dataset_path = (request.form.get('dataset_path')
                    or (request.json.get('dataset_path') if request.is_json and request.json else None) or '').strip()
    if not train_type or train_type not in {'head', 'torso'}:
        return jsonify({"status": "error", "message": "train_type must be 'head' or 'torso'"}), 400
    if not dataset_path:
        return jsonify({"status": "error", "message": "dataset_path is required"}), 400

    rel_path = dataset_path.replace('\\', '/').lstrip('/')
    base_path = os.path.normpath(os.path.join(PROJECT_ROOT, rel_path))
    if not base_path.startswith(PROJECT_ROOT):
        return jsonify({"status": "error", "message": "invalid dataset path"}), 400

    # Build command according to preview templates
    if train_type == 'head':
        cmd = ['python', 'NeRFs/HeadNeRF/run_nerf.py', '--config', os.path.join(rel_path, 'HeadNeRF_config.txt')]
        title = 'Head-NeRF'
    else:
        cmd = ['python', 'NeRFs/TorsoNeRF/run_nerf.py', '--config', os.path.join(rel_path, 'TorsoNeRF_config.txt')]
        title = 'Torso-NeRF'

    print('Training command:', ' '.join(cmd))
    try:
        # Non-blocking suggestion: in production use a task queue; here run synchronously
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        return jsonify({"status": "success", "message": f"{title} training completed."})
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": f"Training failed: {e}"}), 500


def _find_conda_exe():
    """Try to resolve a usable conda executable path on Windows and other OS."""
    # Respect CONDA_EXE if present
    conda_exe = os.environ.get('CONDA_EXE')
    if conda_exe and os.path.isfile(conda_exe):
        return conda_exe

    # Common Windows install locations
    home = os.path.expanduser('~')
    candidates = [
        os.path.join(home, 'miniconda3', 'Scripts', 'conda.exe'),
        os.path.join(home, 'Anaconda3', 'Scripts', 'conda.exe'),
        os.path.join(home, 'Miniconda3', 'Scripts', 'conda.exe'),
        'conda',
        'conda.exe'
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Fallback to plain name (may work if on PATH)
    return 'conda'

# API: run data processing step
@app.route('/api/process_data', methods=['POST'])
def api_process_data():
    # Prefer form fields, then JSON, then query args for robustness
    speaker = (request.form.get('speaker')
               or (request.json.get('speaker') if request.is_json and request.json else None)
               or request.args.get('speaker')
               or '')
    step = (request.form.get('step')
            or (request.json.get('step') if request.is_json and request.json else None)
            or request.args.get('step')
            or '')

    speaker = (speaker or '').strip()
    step = (str(step) if step is not None else '').strip()

    if not speaker or step == '':
        return jsonify({"status": "error", "message": "speaker and step are required"}), 400

    # Validate step is an integer
    if not step.isdigit():
        return jsonify({"status": "error", "message": "step must be an integer"}), 400

    # Steps that require ASR env
    use_asr_env = step in {'10', '13', '16'}

    # Build command; prefer conda run to activate env reliably in subprocess
    if use_asr_env:
        conda_exe = _find_conda_exe()
        cmd = [conda_exe, 'run', '-n', 'py39asr_clean', 'python', 'data_util/process_data.py', f'--id={speaker}', f'--step={step}']
    else:
        cmd = ['python', 'data_util/process_data.py', f'--id={speaker}', f'--step={step}']

    # Echo the command to the Flask console for visibility
    print('Running command:', ' '.join(cmd))

    try:
        # Run in project root, inherit stdout/stderr so child output appears in console
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        return jsonify({"status": "success", "message": "Process data completed."})
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": f"Process data failed: {e}"}), 500
    except FileNotFoundError as e:
        # conda not found or python not found
        return jsonify({"status": "error", "message": f"Dependency not found: {e}. Tried command: {' '.join(cmd)}"}), 500

# 3. Generate audio
@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        # Expect JSON or form with input_text and target_speaker
        input_text = request.form.get('input_text') or (request.json.get('input_text') if request.is_json else None)
        target_speaker = request.form.get('target_speaker') or (request.json.get('target_speaker') if request.is_json else None)
        print("开始生成...", input_text, target_speaker)
        output_url = None
        if input_text and input_text.strip():
            # Run TTS script in project root to generate audio
            script_path = os.path.join(PROJECT_ROOT, 'tts_clone_speaker_based_on_speech_sample.py')
            try:
                # Pass target_speaker as the first argument (script expects target_speaker then text)
                subprocess.run(['python', script_path, target_speaker, input_text], cwd=PROJECT_ROOT, check=True)
                # Determine latest generated file in output folder
                latest_file = None
                latest_mtime = -1
                if os.path.isdir(OUTPUT_DIR):
                    for f in os.listdir(OUTPUT_DIR):
                        if f.endswith('_output.wav'):
                            p = os.path.join(OUTPUT_DIR, f)
                            m = os.path.getmtime(p)
                            if m > latest_mtime:
                                latest_mtime = m
                                latest_file = f
                if latest_file:
                    # Return a URL that the browser can fetch
                    output_url = f"/media/output/{latest_file}"
                else:
                    return jsonify({"status": "error", "message": "No output WAV generated."}), 500
            except subprocess.CalledProcessError as e:
                return jsonify({"status": "error", "message": f"Generation failed: {e}"}), 500
        return jsonify({"status": "success", "message": "Generation Completed!", "output_path": output_url})
    return render_template('generate.html')

# Huawei Cloud TTS proxy endpoint
@app.route('/generate_huawei', methods=['POST'])
def generate_huawei():
    # Read input
    input_text = request.form.get('input_text') or (request.json.get('input_text') if request.is_json else None)
    target_speaker = request.form.get('target_speaker') or (request.json.get('target_speaker') if request.is_json else None)
    reference_audio_path = request.form.get('reference_audio_path') or (request.json.get('reference_audio_path') if request.is_json else None)
    reference_text = request.form.get('reference_text') or (request.json.get('reference_text') if request.is_json else None)

    input_text = (input_text or '').strip()
    target_speaker = (target_speaker or '').strip()
    reference_audio_path = (reference_audio_path or '').strip()
    reference_text = (reference_text or '').strip()

    if not input_text:
        return jsonify({"status": "error", "message": "input_text is required"}), 400
    if not target_speaker:
        return jsonify({"status": "error", "message": "target_speaker is required"}), 400
    if not reference_audio_path:
        return jsonify({"status": "error", "message": "reference_audio_path is required"}), 400

    # Map a media URL like /media/asset/xxx.wav to the filesystem path under ASSET_DIR
    def resolve_reference_path(path: str) -> str:
        # Accept absolute Windows paths, project-relative paths, or our media route.
        norm = path.replace('\\', '/')
        if norm.startswith('/media/asset/'):
            rel = norm[len('/media/asset/'):]
            return os.path.join(ASSET_DIR, rel)
        if norm.startswith('/'):
            return os.path.join(PROJECT_ROOT, norm.lstrip('/'))
        # Otherwise treat as project-relative
        return os.path.join(PROJECT_ROOT, norm)

    ref_fs_path = resolve_reference_path(reference_audio_path)
    if not os.path.isfile(ref_fs_path):
        return jsonify({"status": "error", "message": f"reference audio not found: {ref_fs_path}"}), 400

    try:
        # Read public IP from external settings/env
        if not PUBLIC_CLONE_IP:
            return jsonify({"status": "error", "message": "Clone server IP not configured"}), 500
        url = f"http://{PUBLIC_CLONE_IP}:8000/clone"

        with open(ref_fs_path, 'rb') as f:
            files = { 'reference_audio': f }
            data = {
                'reference_text': reference_text or '',
                'target_text': input_text
            }
            resp = requests.post(url, files=files, data=data, timeout=120)

        if resp.status_code != 200:
            return jsonify({"status": "error", "message": f"Remote API error: {resp.status_code} {resp.text}"}), 502

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = time.strftime('%y%m%d_%H%M%S')
        filename = f"{timestamp}_output.wav"
        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, 'wb') as out:
            out.write(resp.content)

        output_url = f"/media/output/{filename}"
        return jsonify({"status": "success", "output_path": output_url})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# 4. 人机对话页面
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    return render_template('chat.html')

@app.route('/ai_response', methods=['POST'])
def ai_response():
    user_message = request.form.get('message', '').strip()
    if not user_message:
        return jsonify({"error": "消息不能为空"}), 400

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Limit your reply to 50 characters maximum."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=512
        )
        ai_reply = response.choices[0].message.content.strip()
        # Hard-limit to 50 characters server-side
        #ai_reply = ai_reply[:50]
        return jsonify({"reply": ai_reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    #app.run(debug=True, port=5001)
    app.run(host="0.0.0.0", port=5001, debug=True)