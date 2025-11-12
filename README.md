# 🚕 Uber Fare Predictor API — Run Locally

This project provides a **Machine Learning-powered Uber Fare Prediction API** built with **Python** and **Flask**, using a **Random Forest Regression Model** trained on Uber’s open fare dataset.


## 📦 Files Included

* `uber_fare_app.py` → main Flask app
* `uber_fare_rf_model.pkl` → trained Random Forest model
* `requirements.txt` → required dependencies
* `Procfile` → for optional deployment (not needed locally)

---

## ⚙️ How to Run the Project Locally

Follow these steps to set up and run the app on your system 👇

### 1️⃣ Clone or Download the Repository

```bash
git clone https://github.com/your-username/uber-fare-api.git
cd uber-fare-api
```

### 2️⃣ Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
```

Activate it:

* **Windows:** `venv\Scripts\activate`
* **Mac/Linux:** `source venv/bin/activate`

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App

```bash
python uber_fare_app.py
```

You should see output like:

```
Running on http://127.0.0.1:5000/
```

### 5️⃣ Test the API

Open your browser or use any API tool (like Postman) to access:

* **Predict endpoint:**

  ```
  http://127.0.0.1:5000/predict
  ```
* **Autocomplete endpoint:**

  ```
  http://127.0.0.1:5000/autocomplete
  ```

---

## 🧠 Notes

* Make sure `uber_fare_rf_model.pkl` is in the **same folder** as `uber_fare_app.py`.
* If using environment variables, create a `.env` file (optional).
* Works with **Python 3.8+**.

---

## ✅ After Successfully Running

Once your backend is running locally, you can check out the **live web app interface** here:
👉 [**Uber Fare Predictor Frontend**](https://uberfarepredictor.netlify.app/)

---

### 💻 Tech Stack

Python · Flask · Scikit-learn · Random Forest · Streamlit · NLP · LLM · Machine Learning
