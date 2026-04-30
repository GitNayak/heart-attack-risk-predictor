"""
Heart Attack Risk Predictor — RAG Web App
==========================================

This file runs the Streamlit web interface. It combines a trained Machine Learning 
model (Logistic Regression) with an AI Assistant powered by Generative AI (RAG). 

Run with:   streamlit run app.py
"""

import streamlit as st
import pickle       # Used to load the pre-trained ML model (heart.pkl)
import os           # Used to read environment variables for the API Key
import pandas as pd # Used to manipulate the raw dataset for scaling
import numpy as np  # Used to build the mathematical array for model prediction
from dotenv import load_dotenv

# Load secret API keys from the visible .env file into the environment
load_dotenv()

# ─────────────────────────────────────────────────────────────
# VOICE & AUDIO HELPERS
# ─────────────────────────────────────────────────────────────
import tempfile
from gtts import gTTS
import io

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

def transcribe_audio(audio_bytes, provider, api_key):
    if not api_key:
        return "Please provide an API key in the sidebar."
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        fp.write(audio_bytes.getvalue())
        tmp_path = fp.name
    try:
        if provider == "groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            with open(tmp_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                  file=(tmp_path, file.read()),
                  model="whisper-large-v3",
                )
            return transcription.text
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            with open(tmp_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                  file=file,
                  model="whisper-1",
                )
            return transcription.text
        else:
            return "Audio transcription is supported only for Groq and OpenAI. Please type your question."
    except Exception as e:
        return f"Transcription failed: {e}"
    finally:
        import os
        os.remove(tmp_path)

# ─────────────────────────────────────────────────────────────
# 1. PAGE CONFIGURATION & STYLING
# ─────────────────────────────────────────────────────────────
# Sets up the basic browser tab details and minimizes the sidebar by default.
st.set_page_config(
    page_title="Heart Attack Risk Predictor + AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject modern, rich aesthetics using raw CSS. This overrides default Streamlit styling.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0f1117; } /* Deep modern dark mode background */

/* The main title banner at the top of the interface */
.hero-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e3a5f 100%);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(99,102,241,0.3);
}
.hero-card h1 { color: #e0e7ff; font-size: 2rem; margin: 0; }
.hero-card p  { color: #a5b4fc; margin: 0.5rem 0 0; }

/* Dynamic risk output cards */
.risk-high {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.risk-low {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #22c55e;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.risk-label { font-size: 2rem; font-weight: 700; }
.risk-high .risk-label { color: #fca5a5; }
.risk-low  .risk-label { color: #86efac; }

/* AI Assistant Chat UI elements */
.chat-user {
    background: #1e3a5f;
    border-radius: 12px 12px 4px 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #e0f2fe;
    border-left: 3px solid #3b82f6;
}
.chat-ai {
    background: #1a1a2e;
    border-radius: 12px 12px 12px 4px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #e2e8f0;
    border-left: 3px solid #8b5cf6;
}

/* Base Streamlit Buttons styling for a premium feel */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    width: 100%;
}
.stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

/* Sidebar styling */
div[data-testid="stSidebar"] { background: #0f172a; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 2. MACHINE LEARNING & AI SYSTEM INITIALIZATION
# ─────────────────────────────────────────────────────────────

# @st.cache_resource ensures the system only initializes once during app load.
# It holds the heavy ML models and Vector Database securely in memory without restarting.
@st.cache_resource(show_spinner=False)
def load_rag(api_key, provider):
    """
    Creates the AI Chat connection and loads the ML Model.
    If the API Key is completely missing, the RAG (AI) pipeline will temporarily securely disable itself, 
    but the ML Prediction pipeline will still remain fully operational.
    """
    from rag_pipeline import create_rag_pipeline
    
    # Initialize the specific RAG pipeline instance
    rag = create_rag_pipeline(
        api_key=api_key if api_key else None,
        provider=provider,
        rebuild=False,
        base_dir="."
    )
    
    # Safely load the previously trained Logistic Regression machine learning model
    if rag.model is None and os.path.exists("heart.pkl"):
        with open("heart.pkl", "rb") as f:
            rag.model = pickle.load(f)
            
    # Load the raw dataset so we can calculate mathematically proper mean/standard deviation.
    # We drop unrequired features because the ML model was only trained on 10 specific features.
    if rag.df is None and os.path.exists("heart.csv"):
        rag.df = pd.read_csv("heart.csv").drop(["oldpeak","slp","thall"], axis=1, errors="ignore")
        
    return rag


# ─────────────────────────────────────────────────────────────
# 3. SIDEBAR (API SETTINGS)
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.caption("Setup AI Chat features.")

    # Hidden in an expander so it doesn't clutter the main UI
    with st.expander("API Configuration", expanded=False):
        provider = st.selectbox("🤖 Provider", ["groq", "gemini", "openai"])
        
        # Check standard environment variable endpoints based on dropdown selection
        env_key = ""
        if provider == "groq":
            env_key = os.getenv("GROQ_API_KEY", "")
        elif provider == "gemini":
            env_key = os.getenv("GOOGLE_API_KEY", "")
        else:
            env_key = os.getenv("OPENAI_API_KEY", "")
            
        api_key = st.text_input("API Key", type="password", value=env_key)
        
        if st.button("Apply / Refresh", use_container_width=True):
            st.session_state.clear() # Clears Streamlit cache to securely reload models
            st.rerun()

# --- INSTANT AUTO INITIALIZATION ---
# By checking if 'rag' is in session_state, we guarantee it loads immediately the moment 
# the user opens the web page. No clicking required.
if "rag" not in st.session_state:
    st.session_state.rag = load_rag(api_key, provider)


# ─────────────────────────────────────────────────────────────
# 4. MAIN USER INTERFACE (HERO HEADER & TABS)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <h1>🫀 Heart Attack Risk Predictor</h1>
    <p>Predictive Assessment & Medical AI Assistant</p>
</div>
""", unsafe_allow_html=True)

# Define exactly two simple Tabs for the user to navigate
tab1, tab2 = st.tabs(["🏥 Patient Assessment", "💬 AI Assistant"])

# ═══════════════════════════════════════════════════
# TAB 1 — PATIENT ASSESSMENT FORM (PREDICTIONS)
# ═══════════════════════════════════════════════════
with tab1:
    st.markdown("### 🏥 Risk Dashboard")
    st.caption("Enter clinical values to get an instant ML prediction.")

    # A logical form wrapper ensures the algorithm only triggers when "Assess Risk" is actually clicked
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        # Basic Demographics and Chest Data
        with col1:
            age = st.number_input("Age (years)", 20, 100, 55)
            # Radio buttons map user-friendly labels back into strict binary math (0 or 1)
            sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
            sex_val = 0 if "Female" in sex else 1
            
            cp = st.selectbox("Chest Pain Type", ["0 - Typical angina","1 - Atypical angina","2 - Non-anginal pain","3 - Asymptomatic"])
            cp_val = int(cp[0]) # Extract standard integer
            
        # Vitals Data
        with col2:
            trtbps = st.number_input("Resting Blood Pressure (mmHg)", 80, 200, 120)
            chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
            
            fbs_opt = st.radio("Fasting Blood Sugar > 120 mg/dL?", ["No", "Yes"], horizontal=True)
            fbs_val = 0 if "No" in fbs_opt else 1
            
            restecg = st.selectbox("Resting ECG", ["0 - Normal","1 - ST Abnormality","2 - LV Hypertrophy"])
            restecg_val = int(restecg[0])

        # Stress Test Outcomes
        with col3:
            thalachh = st.number_input("Max Heart Rate", 70, 210, 150)
            
            exng_opt = st.radio("Exercise Induced Angina?", ["No", "Yes"], horizontal=True)
            exng_val = 0 if "No" in exng_opt else 1
            
            caa = st.selectbox("Major Vessels Blocked", [0, 1, 2, 3])

        explain = st.checkbox("Generate AI Explanation (Requires API Key)", value=True)
        submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)

    # ────── EXECUTE MACHINE LEARNING MODEL ──────
    if submitted:
        # 1. Structure the raw inputs strictly into the order the Model was originally trained on.
        patient_data = {
            "age": age, "sex": sex_val, "cp": cp_val,
            "trtbps": trtbps, "chol": chol, "fbs": fbs_val,
            "restecg": restecg_val, "thalachh": thalachh,
            "exng": exng_val, "caa": caa
        }
        
        rag = st.session_state.rag
        
        if rag.model is not None:
            # 2. Extract out the raw user variables
            feature_order = ["age","sex","cp","trtbps","chol","fbs","restecg","thalachh","exng","caa"]
            raw_X = np.array([[patient_data[f] for f in feature_order]])
            
            # 3. MATHEMATICAL FIX: Apply Standard Scaling
            # The original ML LogisticRegression was trained using `StandardScaler` inside the Jupyter Notebook.
            # If we don't scale the inputs here (e.g. shrink a 180 Cholesterol down to roughly 0.1), 
            # the regression mathematics will completely collapse. We determine the exact mean and std dev 
            # straight from the heart.csv dataset so our scaling perfectly mimics the original notebook training.
            df_features = rag.df[feature_order]
            means = df_features.mean().values
            stds = df_features.std(ddof=0).values # (ddof=0 perfectly matches sklearn's algorithm)
            stds[stds == 0] = 1.0 # Fail-safe specifically to prevent mathematical division by zero
            
            X_scaled = (raw_X - means) / stds
            
            # 4. Generate the ML Probability
            pred = rag.model.predict(X_scaled)[0]
            
            # NOTE ON LOGIC FLIP: In the default Kaggle/UCI Heart Disease dataset, 
            # Output Target 1 actually signifies "No Disease/Healthy" and 0 signifies "Heart Disease".
            # We strictly map `0` to HIGH RISK so it displays logical real-world conclusions.
            risk = "HIGH RISK" if pred == 0 else "LOW RISK"
            css_class = "risk-high" if pred == 0 else "risk-low"
            emoji = "⚠️" if pred == 0 else "✅"

            # Check predict_proba attribute to grab the exact % likelihood. 
            if hasattr(rag.model, "predict_proba"):
                prob = rag.model.predict_proba(X_scaled)[0]
                high_pct = round(float(prob[0]) * 100, 1) # index 0 specifically refers to the High Risk class.
            else:
                high_pct = 100.0 if pred == 0 else 0.0

            # 5. Output Card
            st.markdown(f"""
            <div class="{css_class}">
                <div class="risk-label">{emoji} {risk}</div>
                <p style="color:#cbd5e1;margin-top:0.5rem;font-size:1.1rem;">Risk Probability: <b>{high_pct}%</b></p>
            </div>
            <br>
            """, unsafe_allow_html=True)
            
            # ────── AI EXPLANATION INTEGRATION ──────
            if explain:
                # Disables if the API check failed due to missing secrets.
                if getattr(rag, "qa_chain", None) is None:
                    st.warning("⚠️ The AI Explanation feature requires an API key. Please configure it in the sidebar.")
                else:
                    with st.spinner("🤖 AI is interpreting the results..."):
                        try:
                            cp_map = {0:"typical angina",1:"atypical angina",2:"non-anginal pain",3:"asymptomatic"}
                            ecg_map = {0:"normal",1:"ST-T wave abnormality",2:"left ventricular hypertrophy"}
                            
                            # CRITICAL ARCHITECTURE STRATEGY:
                            # We deliberately inject the explicit ML variable (`{risk}`) directly into the AI's prompt!
                            # This strictly anchors the AI to the Mathematical Truth, completely preventing the LLM 
                            # from 'hallucinating' and claiming the patient is low risk when the ML says otherwise.
                            question = f"""
                            The Machine Learning model predicts this patient is: {risk} ({high_pct}% probability).
                            Based on this patient's clinical data, explain WHY their risk is {risk}:
                            Age: {age}, Sex: {'Male' if sex_val else 'Female'}, Chest Pain: {cp_map[cp_val]}, 
                            BP: {trtbps}, Chol: {chol}, Fasting Blood Sugar >120: {'Yes' if fbs_val else 'No'},
                            ECG: {ecg_map[restecg_val]}, HR: {thalachh}, Angina: {'Yes' if exng_val else 'No'}, Vessels Blocked: {caa}
                            Keep the explanation concise, clear, and focus on the most important features driving this {risk} prediction.
                            """
                            # Send standard text into Langchain Expression Language (LCEL) endpoint
                            answer = rag.qa_chain.invoke(question)
                            st.markdown(f'<div class="chat-ai">🤖 {answer}</div>', unsafe_allow_html=True)
                            
                            # Convert Explanation to Speech
                            audio_fp = text_to_speech(answer)
                            st.audio(audio_fp, format='audio/mp3', autoplay=True)
                        except Exception as e:
                            st.error(f"❌ AI connection error: {e}")

# ═══════════════════════════════════════════════════
# TAB 2 — MEDICAL Q&A CHAT
# ═══════════════════════════════════════════════════
with tab2:
    st.markdown("### 💬 Medical Assistant")
    
    # Ensure they have authorized Langchain access
    if getattr(st.session_state.rag, "qa_chain", None) is None:
        st.info("💡 To chat with the AI, open the sidebar on the left and enter your free API Key.")
    else:
        # Load independent chat history context array via Streamlit Memory
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display historical lines cleanly using Custom CSS bounds
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">🧑 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai">🤖 <b>AI:</b> {msg["content"]}</div>', unsafe_allow_html=True)
                if "audio" in msg:
                    # Autoplay only the most recent AI message
                    autoplay = (i == len(st.session_state.chat_history) - 1)
                    st.audio(msg["audio"], format="audio/mp3", autoplay=autoplay)

        # Handle both Text and Voice Input
        final_query = None
        
        user_input = st.chat_input("Ask a clinical question about heart health...")
        audio_value = st.audio_input("🎤 Or record your question here:")
        
        if user_input:
            final_query = user_input
        elif audio_value:
            # Prevent infinite loop by checking if we already processed this exact audio recording
            audio_bytes_val = audio_value.getvalue()
            if st.session_state.get("last_processed_audio") != audio_bytes_val:
                st.session_state["last_processed_audio"] = audio_bytes_val
                with st.spinner("Transcribing your voice..."):
                    final_query = transcribe_audio(audio_value, provider, api_key)
            else:
                final_query = None

        # Route User input to LangChain Context Analyzer
        if final_query:
            st.session_state.chat_history.append({"role": "user", "content": final_query})
            with st.spinner("Thinking..."):
                answer = st.session_state.rag.qa_chain.invoke(final_query)
                
                # Convert response to speech and add to history so it plays
                audio_fp = text_to_speech(answer)
                st.session_state.chat_history.append({"role": "ai", "content": answer, "audio": audio_fp.getvalue()})
                st.rerun()

# ─────────────────────────────────────────────────────────────
# 5. FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<center style="color:#475569;font-size:0.8rem;">
⚠️ <b>Medical Disclaimer:</b> This tool is for educational purposes only.
Not a substitute for professional medical advice, diagnosis, or treatment.
Always consult a licensed physician for medical decisions.
</center>
""", unsafe_allow_html=True)
