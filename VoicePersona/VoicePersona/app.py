from flask import Flask, render_template, request, jsonify, send_from_directory
import subprocess
import os

app = Flask(__name__)

# Resolve project root (repo root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

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

# 3. 视频生成页面
@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        # Expect JSON or form with input_text
        input_text = request.form.get('input_text') or (request.json.get('input_text') if request.is_json else None)
        print("开始生成...", input_text)
        output_url = None
        if input_text and input_text.strip():
            # Run TTS script in project root to generate audio
            script_path = os.path.join(PROJECT_ROOT, 'tts_clone_speaker_based_on_speech_sample.py')
            try:
                subprocess.run(['python', script_path, input_text], cwd=PROJECT_ROOT, check=True)
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

if __name__ == '__main__':
    app.run(debug=True, port=5001)