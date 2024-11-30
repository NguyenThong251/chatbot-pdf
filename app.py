from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import PDFChatbot

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)
# Tạo chatbot từ file PDF
chatbot = PDFChatbot("data/example.pdf")
chatbot.load_pdf()
chatbot.build_index()

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")  # Câu hỏi từ người dùng
    if not user_input:
        return jsonify({"error": "Message is required!"}), 400

    try:
        response = chatbot.answer_question(user_input)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
