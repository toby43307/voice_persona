from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model_training')
def model_training():
    return render_template('training.html')


@app.route('/generate')
def generate():
    return render_template('generate.html')


@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/train', methods=['POST'])
def train():
    model_name = request.form['model_name']
    video_path = request.form['video_path']
    gpu = request.form['gpu']
    epoch = request.form['epoch']
    custom_params = request.form['custom_params']
    
    print(f"--- 收到训练任务：{model_name} ---")
    print(f"参考视频路径: {video_path}")
    print(f"GPU选择: {gpu}")
    print(f"训练轮数: {epoch}")
    print(f"自定义参数: {custom_params}")
    
    return "训练任务已提交！请在VS Code终端查看详情。"

if __name__ == '__main__':
    app.run(debug=True, port=5001)