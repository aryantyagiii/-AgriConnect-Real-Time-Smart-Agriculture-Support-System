
# 🌾 AgriConnect – Real-Time Smart Agriculture Support System

## 📝 Description

**AgriConnect** is an AI-powered, real-time agriculture support platform designed to empower farmers with data-driven insights, smart communication, and market connectivity. The system bridges the gap between farmers, agri-experts, government services, and buyers by providing multilingual support, live weather alerts, soil data, crop advisory, and emergency help — especially for rural and under-connected regions.

AgriConnect enables small and marginal farmers to make informed decisions using AI-powered crop suggestions, IoT-based environment sensing, and digital marketplace access.

---

## 💡 Key Features

- 🌦️ **AI-Based Weather Forecasting** – Alerts farmers to upcoming weather risks  
- 🌱 **Crop Recommendation System** – Based on soil, weather, and location data  
- 🌾 **Soil Health Monitoring** – Through sensors or manual data input  
- 📢 **Multilingual Notifications** – SMS, voice, and app push in local languages  
- 🧑‍🌾 **Farmer Help Portal** – Chat with agri-experts or submit problems via voice/image  
- 🛒 **Digital Agri Marketplace** – Connects farmers directly with buyers and suppliers  
- 🛰️ **Live Dashboard for Authorities** – To monitor trends, needs, and send advisories  
- 📴 **Offline SMS Support** – For feature-phone users in low-internet zones  

---

## 🛠️ Technologies Used

**Frontend:**
- React.js – Admin and Agri Dashboard (web)
- Kivy (Python) or Flutter – Mobile App for farmers
- Leaflet.js – Interactive map for market and crop zones

**Backend:**
- Flask / FastAPI – API for crop recommendations, marketplace, and messages
- Firebase – Realtime data storage for alerts, crop entries, messages
- Twilio / Exotel – For multilingual SMS, IVR, and voice alerts

**AI/ML:**
- Scikit-learn – For crop and fertilizer recommendation models
- TensorFlow Lite – For on-device image-based plant disease detection
- gTTS – For audio advisory generation

**IoT (optional):**
- Soil sensors, DHT11 (temperature/humidity), NodeMCU
- MQTT / Firebase integration for real-time sensor feeds

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/agriconnect.git
cd agriconnect
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Firebase and APIs
- Add Firebase credentials
- Configure OpenWeather, Twilio/Exotel API keys

### 4. Run Flask backend
```bash
python app.py
```

### 5. Run React frontend
```bash
cd frontend
npm install
npm start
```

---

## ▶️ Usage

- **Farmer App:**
  - View weather forecasts, crop tips, and submit photos of diseased plants
  - Receive multilingual voice/SMS alerts
  - Access marketplace to list or find products

- **Admin/Dashboard:**
  - View real-time farmer data on map
  - Send announcements or alerts
  - Analyze crop trends and regional needs

---

## 🎯 Future Scope

- Integrate satellite-based NDVI data for crop health
- Partner with government bodies for subsidy integration
- Use blockchain for transparent crop pricing
- Enable voice assistant integration for illiterate users

---

## 👥 Team

- Aryan Tyagi (Team Leader)  
- Akshit Tyagi  

---

## 📫 Contact

📧 Email: aryantyagi761@gmail.com  
🔗 GitHub: [github.com/aryantyagiii](https://github.com/aryantyagiii)  
🔗 LinkedIn: [linkedin.com/in/aryantyagiii](https://linkedin.com/in/aryantyagiii)  

---

## 📚 References

- 💬 Project write-up, formatting, and documentation guidance powered by [ChatGPT](https://openai.com/chatgpt) by OpenAI  
- 🌐 Weather Data API: [OpenWeatherMap](https://openweathermap.org/)  
- 🌍 Earthquake Feed: [USGS API](https://earthquake.usgs.gov/fdsnws/event/1/)  
- 🔊 Voice services: [gTTS](https://pypi.org/project/gTTS/) and [Twilio](https://www.twilio.com/)  
- 🔧 Realtime Database: [Firebase](https://firebase.google.com/)  

