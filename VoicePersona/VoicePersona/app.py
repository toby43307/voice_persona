from flask import Flask, render_template, request, jsonify, send_from_directory
import subprocess
import os
import json
import glob
import sys
from openai import OpenAI
import requests
import time

app = Flask(__name__)

# Resolve project root (repo root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
ASSET_DIR = os.path.join(PROJECT_ROOT, 'asset')
SPEAKER_PROFILES_PATH = os.path.join(PROJECT_ROOT, 'speaker_profiles.json')


app = Flask(__name__)
client = OpenAI(
    api_key="sk-84a40a46260844e3901324bffd05e906", #os.getenv("DEEPSEEK_API_KEY"),
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
        # 这里写实际连接 AI 训练的代码
        print("开始训练...", request.form)
        return jsonify({"status": "success", "message": "Training Started!"})
    return render_template('train.html')

# Static serving for generated output files
@app.route('/media/output/<path:filename>')
def media_output(filename: str):
    # Serve files from the repo-level output directory
    return send_from_directory(OUTPUT_DIR, filename)

# Static serving for asset reference audio files
@app.route('/media/asset/<path:filename>')
def media_asset(filename: str):
    return send_from_directory(ASSET_DIR, filename)

# Serve speaker profiles JSON for frontend
@app.route('/config/speakers')
def config_speakers():
    try:
        with open(SPEAKER_PROFILES_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

# 3. 视频生成页面
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
        if norm.startswith('/'):  # repo-root relative like /asset/xxx.wav
            return os.path.join(PROJECT_ROOT, norm.lstrip('/'))
        # Otherwise treat as project-relative
        return os.path.join(PROJECT_ROOT, norm)

    ref_fs_path = resolve_reference_path(reference_audio_path)
    if not os.path.isfile(ref_fs_path):
        return jsonify({"status": "error", "message": f"reference audio not found: {ref_fs_path}"}), 400

    try:
        # Hard-coded remote clone API per your request
        public_ip = '1.94.211.55'
        url = f"http://{public_ip}:8000/clone"

        # Send file and form data per example
        with open(ref_fs_path, 'rb') as f:
            files = { 'reference_audio': f }
            data = {
                'reference_text': reference_text or '',
                'target_text': input_text
            }
            resp = requests.post(url, files=files, data=data, timeout=120)

        if resp.status_code != 200:
            return jsonify({"status": "error", "message": f"Remote API error: {resp.status_code} {resp.text}"}), 502

        # Save binary audio content to OUTPUT_DIR
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
            messages=[{"role": "user", "content": user_message}],
            temperature=0.7,
            max_tokens=512
        )
        ai_reply = response.choices[0].message.content.strip()
        return jsonify({"reply": ai_reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, port=5001)