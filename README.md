
<h1 align="center">🔥 Algerian Forest Fire Prediction App</h1>

<p align="center">
  A clean, interactive, and production-ready Machine Learning web app that predicts Fire Weather Index (FWI) using real Algerian forest climate data.
</p>

<p align="center">
  <a href="https://forest-fire-prediction-wvfl.onrender.com/" target="_blank">
    🟢 View Live App
  </a>
</p>

---

## 🎥 Demo Video

https://github.com/user-attachments/assets/2f51972c-08dd-4015-87e2-22ae1136e708

> 📌 *Click the video link above to watch the full demo!*

---

## 🚀 About the Project

This project leverages a trained **Ridge Regression ML model** to predict the **Fire Weather Index (FWI)** based on various weather and climate features. The app is built with Flask for the backend and a clean, responsive Tailwind CSS-based frontend.

It’s designed for:
- 🔥 Early wildfire detection systems
- 🛰️ Environmental research and risk management

---

## 🌐 Live Deployed App

**🖥️ [Click here to use the app →](https://forest-fire-prediction-wvfl.onrender.com/)**  
> ⚠️ *Note: Free Render plan may take 30–60 seconds to wake up from sleep.*

---

## 💡 How It Works

1. 🌡️ User enters weather values (Temperature, RH, Rain, etc.)
2. 📦 Inputs are scaled and fed to a trained ML model
3. 🧠 Ridge Regression predicts the Fire Weather Index
4. 💬 Prediction appears instantly in a styled popup modal

---

## 🧪 Features

- ✅ Ridge Regression Model (trained on real Algerian data)
- 🧠 Scikit-learn preprocessor + model pickle files
- 🌐 Live Flask app deployed on Render
- 💻 Beautiful responsive UI using Tailwind CSS
- 📱 Mobile-friendly form with modern input UX
- 🔮 Prediction popup instead of redirecting to new page
- 📂 Clean folder structure for easy collaboration

---

## 📂 Project Structure

```bash
├── app.py                   # Flask backend
├── models/
│   ├── ridge.pkl            # Trained ML model
│   └── scaler.pkl           # StandardScaler
├── templates/
│   ├── index.html           # Landing page
│   ├── predict.html         # Form page
│   ├── footer.html          # Reusable footer
│   └── prediction.html      # For modal result injection
├── static/                  # Assets (optional)
├── requirements.txt         # Dependencies
└── README.md                # You’re reading it!
````

---

## 🛠️ Built With

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&style=flat-square" />
  <img src="https://img.shields.io/badge/Flask-2.3-black?logo=flask&style=flat-square" />
  <img src="https://img.shields.io/badge/scikit--learn-1.7-orange?logo=scikit-learn&style=flat-square" />
  <img src="https://img.shields.io/badge/TailwindCSS-3.0-38bdf8?logo=tailwindcss&style=flat-square" />
  <img src="https://img.shields.io/badge/Render-Cloud-blueviolet?logo=render&style=flat-square" />
</p>

---

## 🧠 ML Input Features

| Feature     | Description                          |
| ----------- | ------------------------------------ |
| Temperature | in Celsius                           |
| RH          | Relative Humidity (%)                |
| WS          | Wind Speed (km/h)                    |
| Rain        | Rainfall (mm)                        |
| FFMC        | Fine Fuel Moisture Code              |
| DMC         | Duff Moisture Code                   |
| ISI         | Initial Spread Index                 |
| Classes     | Binary class (Fire = 1, No Fire = 0) |
| Region      | Region Code (Bejaia/Sidi-Bel-Abbes)  |

---

## ⚙️ Setup & Run Locally

### 1️⃣ Clone the repo

```bash
git clone https://github.com/TechWithAkash/algerian_forest_fire_prediction_app.git
cd algerian_forest_fire_prediction_app
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app

```bash
python app.py
```

Go to `http://localhost:5000` in your browser.

---

## 🙋 About the Author

**Akash Vishwakarma**
🚀 Passionate about AI, ML, and building powerful user-centric tools.
📬 [Connect on LinkedIn](https://www.linkedin.com/in/akash-vishwakarma-creator/)
🐱 [GitHub Profile](https://github.com/TechWithAkash)

---

## 📃 License

This project is licensed under the MIT License.

---

## ⭐ Show Some ❤️

If you found this project helpful:

⭐ **Star this repository**
🍴 **Fork and contribute**
📢 **Share it on LinkedIn/Twitter**

> *“Technology is best when it brings people together — even to prevent disasters.”*
