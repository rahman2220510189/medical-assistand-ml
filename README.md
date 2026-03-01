# 🏥 Medical Assistant ML — Disease Prediction API

An AI-powered medical symptom checker that predicts diseases from user-described symptoms using a trained Machine Learning model with **Groq AI fallback** for symptoms outside the dataset.

---

## 🔗 Related Repositories

| Repository | Description |
|------------|-------------|
| [🖥️ Frontend](https://github.com/rahman2220510189/medical-assistant-chat-fontend) | Chat interface for the medical assistant |
| [⚙️ Backend](https://github.com/rahman2220510189/medical-assistant-backend) | Node.js backend API wrapper |
| [🤖 ML API](https://github.com/rahman2220510189/medical-assistand-ml) | This repository — Python ML model & FastAPI |

---

## 🧠 How It Works

```
User Input (symptoms)
        ↓
Extract & Match Symptoms (RapidFuzz + NLTK)
        ↓
   Found in Dataset?
    ✅ Yes → Random Forest ML Model → Predict Disease
    ❌ No  → Groq AI (LLaMA 3) → Predict Disease
        ↓
Return: Disease, Medicines, Doctor Specialist, Precautions
```

---

## ✨ Features

- 🔍 **Symptom Extraction** — Extracts symptoms from free-text using NLP + fuzzy matching
- 🌲 **Random Forest Model** — Trained on 4920 samples, 131 symptoms, 41 diseases (100% accuracy)
- 🤖 **Groq AI Fallback** — If symptoms are not in the dataset, Groq AI (LLaMA 3) answers
- 💊 **Medicine Suggestions** — Disease-specific medicine recommendations for all 41 diseases
- 👨‍⚕️ **Doctor Specialist** — Recommends the right type of doctor based on disease
- 📖 **Disease Description** — Detailed description for each predicted disease
- ⚡ **FastAPI** — Fast, modern REST API with auto-generated Swagger docs

---

## 📁 Dataset Files

| File | Description |
|------|-------------|
| `Original_Dataset.csv` | 4920 rows, 41 diseases, 17 symptom columns |
| `Symptom_Weights.csv` | Weight/severity of 130 symptoms |
| `Disease_Description.csv` | Description for 41 diseases |
| `medicine.csv` | 21,714 medicines with brand, generic, dosage info |
| `Doctor_Specialist.csv` | 19 doctor specialist types |
| `Doctor_Versus_Disease.csv` | Disease → Doctor mapping (40 diseases) |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/rahman2220510189/medical-assistand-ml.git
cd medical-assistand-ml
```

### 2. Install dependencies
```bash
pip install fastapi uvicorn scikit-learn joblib numpy pandas rapidfuzz nltk groq pyngrok python-dotenv
```

### 3. Setup environment variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
NGROK_TOKEN=your_ngrok_token_here
```

> 🔑 Get your free Groq API key at [console.groq.com](https://console.groq.com)  
> 🔑 Get your free ngrok token at [ngrok.com](https://ngrok.com)

### 4. Train the model
```bash
python train_model.py
```

This will generate the following `.pkl` files:
- `disease_prediction_model.pkl`
- `label_encoder.pkl`
- `symptom_to_index.pkl`
- `all_symptoms.pkl`
- `disease_descriptions.pkl`
- `doctor_map.pkl`
- `symptom_weights.pkl`

### 5. Run the API
```bash
python api.py
```

The terminal will show your public ngrok URL:
```
🚀 Public URL: https://xxxx-xxx.ngrok-free.app
📖 Docs: https://xxxx-xxx.ngrok-free.app/docs
```

---

## 📡 API Endpoints

### `GET /`
Health check — returns API status and dataset info.

### `GET /health`
Returns system operational status.

### `GET /symptoms`
Returns all 131 symptoms available in the dataset.

### `POST /predict`
Predict disease from free-text symptom description.

**Request:**
```json
{
  "symptoms": "I have itching, skin rash and fever"
}
```

**Response:**
```json
{
  "success": true,
  "disease": "Fungal infection",
  "confidence": 98.5,
  "description": "A fungal infection is caused by fungi...",
  "matched_symptoms": ["itching", "skin_rash"],
  "input_symptoms": ["I have itching, skin rash and fever"],
  "suggested_medicines": ["Fluconazole", "Clotrimazole cream", "Terbinafine"],
  "precautions": ["Consult a doctor immediately", "..."],
  "doctor_specialty": "Dermatologist",
  "source": "dataset",
  "disclaimer": "⚠️ AI prediction. NOT a medical diagnosis."
}
```

### `POST /predict-from-list`
Predict disease from a list of symptoms.

**Request:**
```json
{
  "symptoms": ["itching", "skin rash", "fever"]
}
```

---

## 🤖 Groq AI Fallback

When the user describes symptoms not found in the dataset, the API automatically falls back to **Groq AI (LLaMA 3-8B)** which returns a structured medical response.

The `source` field in the response tells you where the answer came from:

| Source | Meaning |
|--------|---------|
| `dataset` | Predicted by the Random Forest ML model |
| `groq_ai` | Predicted by Groq AI (LLaMA 3) |
| `fallback` | Generic response when both fail |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | Random Forest (scikit-learn) |
| API Framework | FastAPI + Uvicorn |
| AI Fallback | Groq AI — LLaMA 3-8B |
| NLP | NLTK + RapidFuzz |
| Tunneling | ngrok |
| Language | Python 3.10 |

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. The predictions made by this system are **NOT a substitute for professional medical advice, diagnosis, or treatment**. Always consult a qualified doctor for medical concerns.

---

## 👨‍💻 Author

**Riyad Rahman**
- GitHub: [@rahman2220510189](https://github.com/rahman2220510189)
