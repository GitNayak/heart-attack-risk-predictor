# 🫀 Heart Attack Risk Predictor (with RAG AI)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-green)
![RAG](https://img.shields.io/badge/AI-RAG%20Integrated-purple)

A modern, interactive web application that predicts a patient's risk of having a heart attack based on clinical data, and provides personalized, medically-grounded explanations using **Retrieval-Augmented Generation (RAG)**.

## 🚀 Features

* **Machine Learning Prediction:** Uses a trained Logistic Regression model (`heart.pkl`) on standard clinical metrics (Age, Cholesterol, Blood Pressure, ECG, etc.) to instantly calculate risk probability.
* **AI-Powered Explanations:** Integrates a RAG pipeline (using ChromaDB) to explain *why* the model made its prediction, anchoring the LLM strictly to the mathematical output to prevent hallucinations.
* **Medical Assistant Chat:** A dedicated tab allowing users to ask general clinical questions about heart health, answered by the AI using a verified knowledge base (`rag_knowledge_base.md`).
* **Multi-Provider LLM Support:** Easily switch between **Groq**, **Google Gemini**, and **OpenAI** via the sidebar settings.
* **Premium UI:** Custom-styled dark mode Streamlit interface with dynamic risk cards and clean chat elements.

## 🛠️ Technologies Used

* **Frontend:** Streamlit (with custom CSS injections)
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **AI/RAG:** LangChain, ChromaDB
* **Embeddings:** HuggingFace / SentenceTransformers
* **LLM Providers:** Groq, Google Gemini, OpenAI

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd "YOUR_REPO"
```

**2. Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up Environment Variables**
Copy the example environment file and add your API keys:
```bash
cp .env.example .env
```
*Note: You only need one API key (e.g., Groq or Gemini) to run the AI features.*

**5. Run the Application**
```bash
streamlit run app.py
```

## 📂 Project Structure
* `app.py`: The main Streamlit user interface and frontend logic.
* `rag_pipeline.py`: Core logic for the Retrieval-Augmented Generation (document loading, vector database, LLM chain).
* `heart.csv`: The raw dataset used for model standard scaling.
* `heart.pkl`: The pre-trained Logistic Regression machine learning model.
* `rag_knowledge_base.md`: The textual knowledge base the AI uses to ground its answers.

## ⚠️ Medical Disclaimer
This tool is for educational and demonstrative purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a licensed physician for medical decisions.
