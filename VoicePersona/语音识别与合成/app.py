# app.py
import os
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

# 加载 .env 文件（可选）
load_dotenv()

# 初始化 Flask
app = Flask(__name__, static_folder='dist/assets')
CORS(app)  # 允许跨域（开发时有用，生产可限制）

# 配置 Google GenAI
GOOGLE_API_KEY = "api-key"#os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("请设置环境变量 GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# 默认模型
model = genai.GenerativeModel('gemini-1.5-flash')


# === 静态文件托管 ===

@app.route('/')
def serve_index():
    return send_from_directory('dist', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    # 处理根目录下的静态资源（如 favicon.ico）
    if os.path.exists(os.path.join('dist', filename)):
        return send_from_directory('dist', filename)
    # 否则尝试从 assets 目录加载（Vite 默认把 JS/CSS 放在 assets/）
    return send_from_directory('dist/assets', filename)


# === API 接口 ===

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        # 调用 Google GenAI
        response = model.generate_content(user_message)
        reply = response.text.strip() if response.text else "No response."

        return jsonify({"reply": reply})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Failed to process request", "details": str(e)}), 500


# === 启动 ===

if __name__ == '__main__':
    # 确保 dist/ 存在
    if not os.path.exists('dist/index.html'):
        print("❌ 错误：未找到 dist/index.html，请先运行 `npm run build`")
        exit(1)

    print("✅ Flask 正在启动...")
    print("🌐 访问 http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)