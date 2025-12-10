import streamlit as st
import sqlite3
import time
import os

# --- IMPORT MODELS ---
try:
    from svm_sentiment import SVMSentimentModel
    from transformer_sentiment import SentimentModel as TransformerModel
except ImportError as e:
    st.error(f"Missing model files: {e}")
    st.stop()

# --- CONFIG ---
DB_LOGS_NAME = "production_logs.db"
DB_WAREHOUSE = "corporate_data_warehouse.db"

# --- DATABASE SETUP (Kept hidden in background) ---
def init_log_db():
    conn = sqlite3.connect(DB_LOGS_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_version TEXT,
            input_text TEXT,
            prediction_label TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(model_name, text, pred):
    conn = sqlite3.connect(DB_LOGS_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO prediction_logs (model_version, input_text, prediction_label) VALUES (?, ?, ?)",
              (model_name, text, pred))
    conn.commit()
    conn.close()

# Initialize DB on load
init_log_db()

# --- PAGE SETUP ---
st.set_page_config(page_title="Sentiment Analyzer")

# --- SIDEBAR ---
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["SVM (Fast)", "DistilBERT (Accurate)"]
)

# --- LOAD MODELS ---
@st.cache_resource
def get_svm_model():
    model = SVMSentimentModel(db_filepath=DB_WAREHOUSE)
    if os.path.exists("svm_model.pkl"):
        model.load_model("svm_model.pkl")
        return model
    return None

@st.cache_resource
def get_transformer_model():
    model = TransformerModel()
    if os.path.exists("./bert_model_saved"):
        model.load_saved_model("./bert_model_saved")
        return model
    return None

# --- MAIN UI ---
st.title(" Sentiment Analyzer")
st.write("Enter text below to detect if the sentiment is Positive or Negative.")

# Input Area
user_text = st.text_area("Input Text", height=150, placeholder="e.g., I absolutely loved this product!")

# Action Button
if st.button("Analyze", type="primary", use_container_width=True):
    if not user_text:
        st.warning("Please type something first.")
    else:
        prediction = None
        
        # Run Inference
        with st.spinner("Analyzing..."):
            if "SVM" in model_choice:
                model = get_svm_model()
                if model: prediction = model.predict(user_text)
                else: st.error("SVM Model not found.")
            
            elif "DistilBERT" in model_choice:
                model = get_transformer_model()
                if model: prediction = model.predict(user_text)
                else: st.error("Transformer Model not found.")

        # Display Result
        if prediction:
            # Log to DB silently
            log_prediction(model_choice, user_text, prediction)

            # Visual Output
            st.divider()
            if prediction == "Positive":
                st.success(f"**Result: {prediction}")
            else:
                st.error(f"**Result: {prediction}")