# Uber Fare API — Render Deploy

This pack contains the two needed files for a free deploy on Render:
- `requirements.txt`
- `Procfile`

## Steps

1. Put these files next to your existing `uber_fare_app.py` and `uber_fare_rf_model.pkl`.
   Folder example:
   ```
   /your-project/
   ├── uber_fare_app.py
   ├── uber_fare_rf_model.pkl
   ├── requirements.txt
   ├── Procfile
   ├── .gitignore
   └── (optional) .env
   ```

2. Push the folder to a GitHub repo.

3. Go to https://render.com → New → Web Service → connect repo.
   - Build Command: (leave default) `pip install -r requirements.txt`
   - Start Command: `gunicorn uber_fare_app:app`
   - Region: any
   - Plan: Free

4. After deploy, your public API will be at:
   - `https://<your-service>.onrender.com/predict`
   - `https://<your-service>.onrender.com/autocomplete`

### Notes
- If you keep `.env` locally, do not upload it. Add any required keys in Render → Environment.
- The app loads `uber_fare_rf_model.pkl` from the same folder. Keep the same filename.
