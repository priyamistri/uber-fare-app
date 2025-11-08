# chat/llm_tools.py
import os, json
from datetime import datetime, timedelta
from dateutil import parser as dtparser
import pytz, requests
from flask import request
from openai import OpenAI

# Load environment variables from .env (if present)
from dotenv import load_dotenv
load_dotenv()

# ----- Config
TZ = pytz.timezone("America/New_York")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# (Optional) debug: show whether key is visible without printing it fully
_key_dbg = os.getenv("OPENAI_API_KEY")
print(
    "DEBUG: OPENAI_API_KEY loaded:",
    ("YES (masked: " + (_key_dbg[:7] + "..." + _key_dbg[-6:]) + ")") if _key_dbg else "NO"
)

# ----- Natural-language time -> ISO8601 (NY tz)
def parse_when(s: str | None) -> str:
    now = datetime.now(TZ)
    if not s:
        return now.isoformat()
    low = s.strip().lower()
    if low == "now":
        return now.isoformat()
    if low.startswith("in "):
        try:
            parts = low.split()
            num = int(parts[1]); unit = parts[2]
            if "min" in unit:
                dt = now + timedelta(minutes=num)
            elif "hour" in unit or "hr" in unit:
                dt = now + timedelta(hours=num)
            elif "day" in unit:
                dt = now + timedelta(days=num)
            else:
                dt = now
            return dt.isoformat()
        except:
            return now.isoformat()
    try:
        base = now.replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None)
        dt = dtparser.parse(s, fuzzy=True, default=base)
        if dt.tzinfo:
            dt = dt.astimezone(TZ)
        else:
            dt = TZ.localize(dt)
        return dt.isoformat()
    except:
        return now.isoformat()

# ----- Tool adapters (reuse your own endpoints)
def tool_get_fare(pickup_address: str, dropoff_address: str, pickup_datetime: str | None, passenger_count: int | None):
    payload = {
        "pickup_address": pickup_address,
        "dropoff_address": dropoff_address,
        "pickup_datetime": parse_when(pickup_datetime),
        "passenger_count": passenger_count or 1,
    }
    base = request.url_root.rstrip("/")
    r = requests.post(f"{base}/predict", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

def tool_autocomplete(query: str):
    base = request.url_root.rstrip("/")
    r = requests.get(f"{base}/autocomplete", params={"q": query}, timeout=12)
    r.raise_for_status()
    return r.json()

# ----- Function-calling schemas for the LLM
LLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_fare",
            "description": "Get fare estimate and cheapest time window",
            "parameters": {
                "type": "object",
                "properties": {
                    "pickup_address": {"type": "string"},
                    "dropoff_address": {"type": "string"},
                    "pickup_datetime": {"type": "string", "description": "ISO8601 or natural language ('now', 'today 6:30pm')"},
                    "passenger_count": {"type": "integer", "minimum": 1}
                },
                "required": ["pickup_address", "dropoff_address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "autocomplete",
            "description": "US-only address suggestions for disambiguation",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    }
]

# ----- Run the LLM with optional tool calls
def run_chat(messages: list[dict]) -> dict:
    """
    messages: [{role, content}, ...]
    returns: {"reply": str, "ui": {...}} or {"error": "..."}
    """
    # 🔑 Fetch key dynamically at request time
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return {"error": "OPENAI_API_KEY not set"}

    oai = OpenAI(api_key=key)

    sys_prompt = (
        "You are a ride-fare assistant. Extract pickup, dropoff, pickup time, passengers. "
        "Call tools to get real prices; never guess. If addresses are ambiguous, call autocomplete "
        "and ask user to pick one of 3 options. Be concise (<120 words). Use $, and suggest quick actions "
        "(e.g., +15m, +30m, Swap). Timezone is America/New_York."
    )
    chat_messages = [{"role": "system", "content": sys_prompt}] + messages

    first = oai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=chat_messages,
        tools=LLM_TOOLS,
        tool_choice="auto",
        temperature=0.2,
    )
    msg = first.choices[0].message

    # If the model asked to call tools, execute them and do a second pass
    if getattr(msg, "tool_calls", None):
        tool_msgs = []
        for tc in msg.tool_calls:
            fn = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            try:
                if fn == "get_fare":
                    res = tool_get_fare(
                        pickup_address=args.get("pickup_address", ""),
                        dropoff_address=args.get("dropoff_address", ""),
                        pickup_datetime=args.get("pickup_datetime"),
                        passenger_count=args.get("passenger_count"),
                    )
                elif fn == "autocomplete":
                    res = tool_autocomplete(args.get("query", ""))
                else:
                    res = {"error": f"unknown tool {fn}"}
            except Exception as e:
                res = {"error": str(e)}
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fn,
                "content": json.dumps(res),
            })

        second = oai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=chat_messages + [msg] + tool_msgs,
            temperature=0.2,
        )
        final_text = second.choices[0].message.content
        return {"reply": final_text, "ui": {"chips": ["+15m", "+30m", "Swap locations", "Change passengers"]}}

    # No tool call (clarification/small talk)
    return {"reply": msg.content, "ui": {"chips": ["+15m", "+30m", "Swap locations", "Change passengers"]}}
