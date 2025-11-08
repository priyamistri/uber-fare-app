from flask import Blueprint, request, jsonify
from .llm_tools import run_chat

chat_bp = Blueprint("chat_bp", __name__)

@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    messages = data.get("messages", [])
    result = run_chat(messages)
    status = 200 if "error" not in result else 500
    return jsonify(result), status
