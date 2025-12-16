from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

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

# 3. 视频生成页面
@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        print("开始生成...", request.form)
        return jsonify({"status": "success", "message": "Generation Started!"})
    return render_template('generate.html')

# 4. 人机对话页面
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
    