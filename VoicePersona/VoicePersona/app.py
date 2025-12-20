from flask import Flask, render_template, request, jsonify, send_from_directory
import subprocess
import os
import json
import glob
from openai import OpenAI

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

# API: run data processing step
@app.route('/api/process_data', methods=['POST'])
def api_process_data():
    speaker = (request.form.get('speaker') or request.json.get('speaker') if request.is_json else '').strip()
    step = (request.form.get('step') or request.json.get('step') if request.is_json else '').strip()
    if not speaker or step == '':
        return jsonify({"error": "speaker and step are required"}), 400
    cmd = ['python', 'data_util/process_data.py', f'--id={speaker}', f'--step={step}']
    try:
        # Run in project root
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        return jsonify({"status": "success", "message": "Process data completed."})
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": f"Process data failed: {e}"}), 500

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