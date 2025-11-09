from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import requests
import time

# NEW: import the chat blueprint
from chat import chat_bp

# ---------------------------
# 1. Initialize Flask App
# ---------------------------
app = Flask(__name__)
CORS(app)

# Register the chatbot blueprint (exposes POST /chat)
app.register_blueprint(chat_bp)

# ---------------------------
# 2. Load Trained Model
# ---------------------------
model = joblib.load("uber_fare_rf_model.pkl")

# ---------------------------
# 3. Geocoding (US-wide & robust)
# ---------------------------
NOMINATIM_HEADERS = {"User-Agent": "uber-fare-app/1.0 (contact: you@example.com)"}

def geocode_us(query, limit=1):
    """
    US-only geocoding via Nominatim HTTP search.
    Returns (lat, lon, display_name) or (None, None, None).
    """
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
    """
    Try to infer a 'City, ST' or 'State, ZIP' hint from any display_name label.
    Takes the last two comma-separated tokens as a coarse hint.
    """
    for label in labels:
        if label and ", " in label:
            parts = [p.strip() for p in label.split(",")]
            if len(parts) >= 2:
                return ", ".join(parts[-2:])
    return None

def with_city_state(q, hint):
    """Append a city/state hint if not already present."""
    if not hint:
        return q
    q_lower = q.lower()
    if hint.lower() in q_lower:
        return q
    return f"{q}, {hint}"

# ---------------------------
# 4. Haversine Distance Function
# ---------------------------
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ---------------------------
# 5. Autocomplete Endpoint (US-only)
# ---------------------------
@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    query = request.args.get("q")
    if not query:
        return jsonify([])

    # Always restrict to the US
    url = (
        f"https://nominatim.openstreetmap.org/search?"
        f"q={query}&format=json&limit=5&countrycodes=us"
    )

    response = requests.get(url, headers=NOMINATIM_HEADERS, timeout=12)
    suggestions = [
        {"display_name": place["display_name"], "lat": place["lat"], "lon": place["lon"]}
        for place in response.json()
    ]
    return jsonify(suggestions)

# ---------------------------
# 6. Prediction Endpoint
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}

    # Parse input
    pickup_address = data.get("pickup_address", "")
    dropoff_address = data.get("dropoff_address", "")
    pickup_datetime = pd.to_datetime(str(data.get("pickup_datetime", "")), errors="coerce")
    passenger_count = int(data.get("passenger_count", 1))

    if pickup_datetime is pd.NaT:
        return jsonify({"error": "Invalid pickup_datetime"}), 400

    # --- Convert addresses → lat/lon (US-wide, robust) ---
    pickup_lat, pickup_long, pickup_label = geocode_us(pickup_address)
    drop_lat, drop_long, drop_label   = geocode_us(dropoff_address)

    # If one side failed, try gentle fallbacks using a coarse city/state hint
    if pickup_lat is None or drop_lat is None:
        hint = infer_city_state_hint(pickup_label, drop_label)
        if pickup_lat is None and pickup_address:
            alt = with_city_state(pickup_address, hint)
            pickup_lat, pickup_long, pickup_label = geocode_us(alt)
        if drop_lat is None and dropoff_address:
            alt = with_city_state(dropoff_address, hint)
            drop_lat, drop_long, drop_label = geocode_us(alt)

    # Explicit airport alias fallback (works anywhere in the US)
    if pickup_lat is None and "airport" in pickup_address.lower():
        pickup_lat, pickup_long, pickup_label = geocode_us(pickup_address + ", USA")
    if drop_lat is None and "airport" in dropoff_address.lower():
        drop_lat, drop_long, drop_label = geocode_us(dropoff_address + ", USA")

    # Final check
    if pickup_lat is None:
        return jsonify({"error": f"Could not geocode pickup: '{pickup_address}'. Try adding city/ZIP (e.g., 'Phoenix, AZ')."}), 400
    if drop_lat is None:
        return jsonify({"error": f"Could not geocode dropoff: '{dropoff_address}'. Try adding city/ZIP."}), 400

    # Compute distance
    distance_km = haversine(pickup_long, pickup_lat, drop_long, drop_lat)

    # Time windows (minutes from now)
    windows = [0, 15, 30, 60]
    fares = {}

    for w in windows:
        time_dt = pickup_datetime + pd.Timedelta(minutes=w)
        features = pd.DataFrame([{
            "passenger_count": passenger_count,
            "hour": time_dt.hour,
            "day": time_dt.day,
            "month": time_dt.month,
            "weekday": time_dt.weekday(),
            "distance_km": distance_km
        }])
        fare = model.predict(features)[0]
        fares[w] = round(float(fare), 2)

    # Best option
    best_window = min(fares, key=fares.get)
    best_fare = fares[best_window]
    current_fare = fares[0]

    if best_window == 0:
        suggestion = f"Best to book now at ${current_fare:.2f}. Waiting doesn’t reduce the fare."
    else:
        future_time = (pickup_datetime + pd.Timedelta(minutes=best_window)).strftime("%H:%M")
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
# 7. Run App
# ---------------------------
if __name__ == "__main__":
    print("✅ Flask server running at http://127.0.0.1:5000")
    app.run(debug=True)