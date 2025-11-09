from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import traceback
import time
import pandas as pd
import numpy as np
import joblib
import requests

# Optional OpenAI (used in /chat for a nicer reply; we fall back gracefully)
OPENAI_AVAILABLE = True
try:
    from openai import OpenAI
except Exception:
    OPENAI_AVAILABLE = False

# ---------------------------
# 1) Initialize Flask App
# ---------------------------
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "uber-fare-api"}), 200

# ---------------------------
# 2) Load Trained Model
# ---------------------------
# The model file must be in the same folder with this exact name.
model = joblib.load("uber_fare_rf_model.pkl")

# ---------------------------
# 3) Geocoding helpers (US-only, robust)
# ---------------------------
NOMINATIM_HEADERS = {"User-Agent": "uber-fare-app/1.0 (contact: you@example.com)"}

def geocode_us(query, limit=1):
    """US-only geocoding via Nominatim search. Returns (lat, lon, display_name) or (None, None, None)."""
    if not query:
        return None, None, None
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "limit": str(limit),
        "countrycodes": "us",
        "addressdetails": 0
    }
    try:
        r = requests.get(url, params=params, headers=NOMINATIM_HEADERS, timeout=12)
        r.raise_for_status()
        data = r.json() or []
        if not data:
            return None, None, None
        top = data[0]
        return float(top["lat"]), float(top["lon"]), top.get("display_name", query)
    except Exception:
        return None, None, None

def infer_city_state_hint(*labels):
    """Infer a coarse 'City, ST' hint from labels for a second-pass geocode."""
    for label in labels:
        if label and ", " in label:
            parts = [p.strip() for p in label.split(",")]
            if len(parts) >= 2:
                return ", ".join(parts[-2:])
    return None

def with_city_state(q, hint):
    if not hint:
        return q
    if hint.lower() in q.lower():
        return q
    return f"{q}, {hint}"

# ---------------------------
# 4) Distance (Haversine)
# ---------------------------
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ---------------------------
# 5) Autocomplete (US-only)
# ---------------------------
@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    query = request.args.get("q")
    if not query:
        return jsonify([])

    url = (
        f"https://nominatim.openstreetmap.org/search?"
        f"q={query}&format=json&limit=5&countrycodes=us"
    )
    r = requests.get(url, headers=NOMINATIM_HEADERS, timeout=12)
    r.raise_for_status()
    suggestions = [
        {"display_name": place["display_name"], "lat": place["lat"], "lon": place["lon"]}
        for place in r.json()
    ]
    return jsonify(suggestions)

# ---------------------------
# 6) Core fare computation
# ---------------------------
def compute_fares(pickup_address, dropoff_address, pickup_datetime, passenger_count):
    """
    Returns (fares_dict, best_window, best_fare, current_fare, pickup_label, drop_label, distance_km).
    Raises ValueError with a friendly message for user-facing errors.
    """
    # Geocode
    p_lat, p_lon, p_label = geocode_us(pickup_address)
    d_lat, d_lon, d_label = geocode_us(dropoff_address)

    # Fallbacks with inferred hint
    if p_lat is None or d_lat is None:
        hint = infer_city_state_hint(p_label, d_label)
        if p_lat is None and pickup_address:
            alt = with_city_state(pickup_address, hint)
            p_lat, p_lon, p_label = geocode_us(alt)
        if d_lat is None and dropoff_address:
            alt = with_city_state(dropoff_address, hint)
            d_lat, d_lon, d_label = geocode_us(alt)

    # Airport alias fallback
    if p_lat is None and "airport" in pickup_address.lower():
        p_lat, p_lon, p_label = geocode_us(pickup_address + ", USA")
    if d_lat is None and "airport" in dropoff_address.lower():
        d_lat, d_lon, d_label = geocode_us(dropoff_address + ", USA")

    if p_lat is None:
        raise ValueError(f"Could not geocode pickup: '{pickup_address}'. Try adding city/ZIP (e.g., 'Phoenix, AZ').")
    if d_lat is None:
        raise ValueError(f"Could not geocode dropoff: '{dropoff_address}'. Try adding city/ZIP.")

    # Distance
    distance_km = haversine(p_lon, p_lat, d_lon, d_lat)

    # Time windows and features
    windows = [0, 15, 30, 60]
    fares = {}
    for w in windows:
        t = pickup_datetime + pd.Timedelta(minutes=w)
        features = pd.DataFrame([{
            "passenger_count": int(passenger_count),
            "hour": t.hour,
            "day": t.day,
            "month": t.month,
            "weekday": t.weekday(),
            "distance_km": distance_km
        }])
        pred = model.predict(features)[0]
        fares[w] = round(float(pred), 2)

    best_window = min(fares, key=fares.get)
    best_fare = fares[best_window]
    current_fare = fares[0]
    return fares, best_window, best_fare, current_fare, p_label, d_label, distance_km

# ---------------------------
# 7) Predict (structured JSON)
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}

    pickup_address = data.get("pickup_address", "")
    dropoff_address = data.get("dropoff_address", "")
    dt = pd.to_datetime(str(data.get("pickup_datetime", "")), errors="coerce")
    passenger_count = int(data.get("passenger_count", 1))

    if dt is pd.NaT:
        return jsonify({"error": "Invalid pickup_datetime"}), 400

    try:
        fares, best_window, best_fare, current_fare, _, _, _ = compute_fares(
            pickup_address, dropoff_address, dt, passenger_count
        )
    except ValueError as ue:
        return jsonify({"error": str(ue)}), 400

    if best_window == 0:
        suggestion = f"Best to book now at ${current_fare:.2f}. Waiting doesn’t reduce the fare."
    else:
        future_time = (dt + pd.Timedelta(minutes=best_window)).strftime("%-I:%M %p")
        suggestion = (
            f"Cheapest fare is ${best_fare:.2f} if you wait {best_window} minutes "
            f"(around {future_time}). Current fare: ${current_fare:.2f}"
        )

    return jsonify({
        "current_fare": current_fare,
        "best_fare": best_fare,
        "best_window": best_window,
        "suggestion": suggestion,
        "all_fares": fares
    })

# ---------------------------
# 8) Chat (free-text OR structured) — uses OpenAI for natural reply
# ---------------------------

FREE_TEXT_HINT = (
    "You can type things like:\n"
    "• jfk airport to 100 n 3rd st phoenix az for 2\n"
    "• from phoenix sky harbor to 100 n 3rd st for 3\n"
    "• 1600 pennsylvania ave to reagan airport for 1 at 3:15 pm\n"
)

def parse_free_text(msg: str):
    """
    Parse free-text like 'jfk airport to 100 n 3rd st for 2' into
    pickup, dropoff, passenger_count, pickup_datetime(optional).
    Returns (pickup, dropoff, passenger_count, pickup_dt or None).
    """
    original = msg.strip()
    text = " ".join(original.split())
    # Extract passenger count: "for 2", "for2", "2 passengers", "passenger 2"
    pax = None
    pax_patterns = [
        r"\bfor\s+(\d+)\b",
        r"\b(\d+)\s*passenger[s]?\b",
        r"\bpassenger[s]?\s*(\d+)\b"
    ]
    for pat in pax_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            pax = int(m.group(1))
            # remove this segment so split 'to' is cleaner
            text = text[:m.start()] + text[m.end():]
            break
    if pax is None:
        pax = 1

    # Extract time like "at 3pm" or "at 15:30" (optional)
    pickup_dt = None
    m_time = re.search(r"\bat\s+(\d{1,2}(:\d{2})?\s*(am|pm)?)\b", text, flags=re.IGNORECASE)
    if m_time:
        time_str = m_time.group(1).strip()
        text = text[:m_time.start()] + text[m_time.end():]
        # We'll interpret time today in UTC for simplicity (or server time).
        # If you want a specific timezone, adjust here.
        today = pd.Timestamp.utcnow().floor("D")
        try:
            pickup_dt = pd.to_datetime(f"{today.date()} {time_str}", errors="coerce")
        except Exception:
            pickup_dt = None

    # Split pickup/dropoff: prefer "from X to Y", else "X to Y"
    pickup = dropoff = None
    m_from = re.search(r"\bfrom\s+(.*?)\s+to\s+(.+)$", text, flags=re.IGNORECASE)
    if m_from:
        pickup = m_from.group(1).strip(", ")
        dropoff = m_from.group(2).strip(", ")
    else:
        parts = re.split(r"\bto\b", text, flags=re.IGNORECASE)
        if len(parts) >= 2:
            pickup = parts[0].strip(", ")
            dropoff = " to ".join(parts[1:]).strip(", ")

    # Clean common trailing words
    if dropoff:
        dropoff = re.sub(r"\bfor\s*$", "", dropoff, flags=re.IGNORECASE).strip()

    return pickup or "", dropoff or "", pax, pickup_dt

def openai_client_or_none():
    if not OPENAI_AVAILABLE:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None

@app.route("/chat", methods=["POST"])
def chat():
    """
    Accepts either:
      - JSON with fields: pickup_address, dropoff_address, passenger_count, pickup_datetime (ISO)
      - or JSON with { "message": "free text like 'jfk airport to 100 n 3rd st for 2'" }
    Returns a chat-style JSON with reply + quick chips + raw fares.
    """
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid or missing JSON body."}), 400

    # Structured payload wins if present
    pickup_address = data.get("pickup_address")
    dropoff_address = data.get("dropoff_address")
    passenger_count = data.get("passenger_count")
    pickup_datetime = data.get("pickup_datetime")

    # Or parse free text
    if not (pickup_address and dropoff_address):
        msg = data.get("message", "")
        if not msg:
            return jsonify({"error": "Please provide a message like 'jfk airport to 100 n 3rd st for 2'.",
                            "hint": FREE_TEXT_HINT}), 400
        pickup_address, dropoff_address, pax, maybe_dt = parse_free_text(msg)
        if not pickup_address or not dropoff_address:
            return jsonify({"error": "Could not parse pickup/dropoff from your message.",
                            "hint": FREE_TEXT_HINT}), 400
        if passenger_count is None:
            passenger_count = pax
        if pickup_datetime is None and maybe_dt is not None:
            pickup_datetime = maybe_dt.isoformat()

    # Defaults
    if passenger_count is None:
        passenger_count = 1
    try:
        passenger_count = int(passenger_count)
    except Exception:
        passenger_count = 1

    # Time default = now (UTC)
    dt = pd.to_datetime(str(pickup_datetime), errors="coerce")
    if dt is pd.NaT:
        dt = pd.Timestamp.utcnow()

    # Compute fares
    try:
        fares, best_window, best_fare, current_fare, p_label, d_label, distance_km = compute_fares(
            pickup_address, dropoff_address, dt, passenger_count
        )
    except ValueError as ue:
        return jsonify({"error": str(ue)}), 400

    if best_window == 0:
        suggestion = f"Best to book now at ${current_fare:.2f}. Waiting doesn’t reduce the fare."
    else:
        future_time = (dt + pd.Timedelta(minutes=best_window)).strftime("%-I:%M %p")
        suggestion = (
            f"Cheapest fare is ${best_fare:.2f} if you wait {best_window} minutes "
            f"(around {future_time}). Current fare: ${current_fare:.2f}"
        )

    # Compose reply: prefer OpenAI for tone; fall back to a templated message
    reply_text = (
        f"Pickup: {p_label}\n"
        f"Dropoff: {d_label}\n"
        f"Pickup Time: {dt.strftime('%-I:%M %p UTC')}\n"
        f"Passengers: {passenger_count}\n\n"
        f"{suggestion}"
    )

    client = openai_client_or_none()
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if client:
        try:
            system = (
                "You are a ride fare assistant. Be concise, friendly, and helpful. "
                "Use the provided suggestion and fares to guide the user."
            )
            user_msg = (
                f"Pickup: {p_label}\n"
                f"Dropoff: {d_label}\n"
                f"Pickup Time: {dt.strftime('%-I:%M %p UTC')}\n"
                f"Passengers: {passenger_count}\n\n"
                f"Suggestion: {suggestion}\n"
                f"All fares: {fares}"
            )
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.2,
            )
            reply_text = completion.choices[0].message.content.strip() or reply_text
        except Exception as e:
            # If OpenAI fails for any reason, we still return a valid response.
            print("OPENAI_ERROR:", repr(e))
            traceback.print_exc()

    return jsonify({
        "reply": reply_text,
        "ui": {"chips": ["+15m", "+30m", "Swap locations", "Change passengers"]},
        "raw": {
            "pickup_label": p_label,
            "dropoff_label": d_label,
            "distance_km": round(distance_km, 2),
            "current_fare": current_fare,
            "best_fare": best_fare,
            "best_window": best_window,
            "all_fares": fares
        }
    }), 200

# ---------------------------
# 9) Run (local dev)
# ---------------------------
if __name__ == "__main__":
    print("✅ Flask server running at http://127.0.0.1:5000")
    app.run(debug=True)
