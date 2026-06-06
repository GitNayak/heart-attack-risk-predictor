# 🫀 Heart Attack Risk Predictor (with RAG AI)

**[🚀 Click here to view the live app!](https://heart-attack-risk-predictor-fscgumdirznradhepuk6tg.streamlit.app)**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-6%20Models%20Benchmarked-green)
![RAG](https://img.shields.io/badge/AI-RAG%20Integrated-purple)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-orange)

A modern, interactive web application that predicts a patient's risk of having a heart attack based on clinical data, and provides personalized, medically-grounded explanations using **Retrieval-Augmented Generation (RAG)**.

---

## ✨ Features

- **ML Model Benchmark:** Evaluated 6 classifiers (Logistic Regression, Decision Tree, Random Forest, KNN, SVM, AdaBoost) on the UCI Heart Disease dataset (303 patients, 10 clinical features) using GridSearchCV 5-fold cross-validation
- **AI-Powered Explanations:** RAG pipeline using ChromaDB + HuggingFace all-MiniLM-L6-v2 embeddings — LLM explanations grounded strictly to model output to prevent hallucination
- **Medical Assistant Chat:** Dedicated tab for clinical questions answered from a verified knowledge base
- **Multi-Provider LLM:** Switch between Groq, Google Gemini, and OpenAI from the sidebar
- **Voice I/O:** Whisper STT for speech input + gTTS for audio output
- **Premium UI:** Custom dark mode Streamlit interface with dynamic risk cards

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| ML Models | Scikit-Learn (6 classifiers) |
| Data Processing | Pandas, NumPy, StandardScaler |
| RAG Pipeline | LangChain LCEL + ChromaDB |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| LLM Providers | Groq / Google Gemini / OpenAI |
| Voice I/O | Whisper STT + gTTS |
| Web UI | Streamlit |

---

## 🤖 Agent Pipeline

Clinical Input (Age, Cholesterol, BP, ECG...)
│
▼
ML Model (best of 6 classifiers via GridSearchCV)
│
▼
Risk Probability Score
│
▼
RAG Pipeline (ChromaDB + HuggingFace Embeddings)
│
▼
LLM Explanation (Groq / Gemini / OpenAI)
│
▼
Grounded Medical Explanation + Chat Interface

---

## 🚀 Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/GitNayak/Heart-Attack-Risk-Predictor.git
cd Heart-Attack-Risk-Predictor
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
copy .env.example .env
```
Add your API key to `.env` — only one provider needed (Groq is free).

**5. Run the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

Heart-Attack-Risk-Predictor/
├── app.py                             # Streamlit UI
├── rag_pipeline.py                    # RAG pipeline (ChromaDB + LangChain)
├── heart.csv                          # UCI Heart Disease dataset
├── heart.pkl                          # Pre-trained ML model
├── rag_knowledge_base.md              # Verified medical knowledge base
├── Heart_Attack_Risk_Predictor.ipynb  # ML training notebook
├── Heart_Attack_RAG_Demo.ipynb        # RAG demo notebook
├── .env.example                       # API key template
└── requirements.txt                   # Dependencies


---

## ⚠️ Medical Disclaimer

This tool is for **educational and demonstrative purposes only**.
It is not a substitute for professional medical advice, diagnosis, or treatment.
Always consult a licensed physician for medical decisions.

---

*Built by Satyajit Nayak — MCA Graduate | AI/ML Enthusiast*