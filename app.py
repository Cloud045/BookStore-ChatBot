from flask import Flask, render_template, request, jsonify
from chatbotQA import BookStoreLangGraphChatBot

app = Flask(__name__)
bot = BookStoreLangGraphChatBot()
user_id = "web_user"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"response": "Bạn chưa nhập tin nhắn."})

    response = bot.chat(user_id, user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
