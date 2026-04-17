# ❤️ CARDIA — Cardiac Digital Twin

Real-time cardiac monitoring system with AI coaching.

## 🔧 Hardware
- ESP32 NodeMCU (WiFi)
- MAX30102 Pulse Oximeter Sensor
- LiPo Battery (wireless)

## 🧠 ML Stack
- XGBoost + Random Forest + LightGBM Ensemble (CES Score)
- LSTM (7-day forecast)
- Isolation Forest (anomaly detection)
- RL Agent (strategy picker)
- SHAP (explainability)
- Kalman Filter (signal denoising)

## 🤖 AI
- Llama 3 via Ollama (local AI coach)

## 💻 Tech Stack
- Backend: FastAPI + SQLite
- Frontend: React + Recharts
- Hardware: ESP32 + MAX30102
- Data: PhysioNet MIT-BIH + Synthetic

## 🚀 Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
python synthetic_data.py
python xgboost_ces.py
python isolation_forest.py
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm start
```

### ESP32
- Open `cardiac_sensor.ino` in Arduino IDE
- Update WiFi credentials + laptop IP
- Upload to ESP32

## 📊 Features
- Live HR, HRV, SpO2 monitoring
- Cardiac Enhancement Score (CES)
- Safety alerts (SAFE/CAUTION/DANGER)
- AI coaching via Llama 3
- 7-day cardiac forecast
- Session history

## 🏆 Results
- CES Model R²: 0.994
- Anomaly Detection: 95%+ accuracy
- Real-time latency: <2 seconds

## 👥 Team Members

- **Sai Pavan S Varada** – ML Engineer , Backend 
- **Sumukh Simha** – Developer, Frontend 
- **Shyam** –  Hardware, Developer

## 🙌 Acknowledgements

Special thanks to all contributors and mentors who supported this project.
