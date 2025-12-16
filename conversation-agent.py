# pip install streamlit Flask==3.0.3 gunicorn==23.0.0 Werkzeug==3.0.3 python-dotenv numpy pandas scikit-learn matplotlib gensim openai
# pip install tiktoken faiss-cpu datasets sentencepiece google-generativeai unstructured plotly jupyter-dash pydub
# pip install accelerate sentence_transformers feedparser speedtest-cli

import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai

# -------------------------------
# 🔧 Configuration de l'application
# -------------------------------
st.set_page_config(page_title="Multi-Model AI Chat", page_icon="🤖", layout="centered")
load_dotenv(override=True)

# -------------------------------
# 🔑 Récupération des clés API
# -------------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# -------------------------------
# 🧩 Vérification des clés API
# -------------------------------
st.sidebar.title("🔑 Clés API")
if openai_api_key:
    st.sidebar.success(f"OpenAI: {openai_api_key[:8]}...")
else:
    st.sidebar.warning("Clé OpenAI manquante")

if groq_api_key:
    st.sidebar.success(f"Groq: {groq_api_key[:8]}...")
else:
    st.sidebar.warning("Clé Groq manquante")

if google_api_key:
    st.sidebar.success(f"Google: {google_api_key[:8]}...")
else:
    st.sidebar.warning("Clé Google manquante")

# -------------------------------
# ⚙️ Initialisation des API
# -------------------------------
openai = OpenAI(api_key=openai_api_key)
google.generativeai.configure(api_key=google_api_key)

system_message = "You are a helpful and intelligent assistant."

# -------------------------------
# 🧠 Mémoire de chat
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # liste de tuples (role, content)

# -------------------------------
# ⚙️ Fonctions des modèles
# -------------------------------
def stream_gpt(prompt):
    messages = [{"role": "system", "content": system_message}]
    for role, content in st.session_state.chat_history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})

    stream = openai.chat.completions.create(
        model="gpt-5-mini", messages=messages, stream=True
    )

    response = ""
    for chunk in stream:
        text = chunk.choices[0].delta.content or ""
        response += text
        yield text


def stream_gemini(prompt):
    gemini = google.generativeai.GenerativeModel(
        model_name="gemini-2.5-flash", system_instruction=system_message
    )
    response = gemini.generate_content(
        [m[1] for m in st.session_state.chat_history] + [prompt],
        stream=True,
    )
    for chunk in response:
        yield chunk.text or ""


def stream_groq(prompt):
    groq = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
    messages = [{"role": "system", "content": system_message}]
    for role, content in st.session_state.chat_history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})

    stream = groq.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=messages, stream=True
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""

# -------------------------------
# 💬 Interface principale
# -------------------------------
st.title("🤖 Chat Multi-Modèles (GPT | Gemini | Groq)")
st.markdown("Discutez en temps réel avec plusieurs modèles d’IA.")

model_choice = st.selectbox("🧩 Choisissez un modèle :", ["GPT", "Gemini", "Groq"])
prompt = st.text_area("🧠 Votre message :", height=130)
send = st.button("Envoyer")

# Affichage de l'historique
if st.session_state.chat_history:
    for role, content in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").markdown(content)
        else:
            st.chat_message("assistant").markdown(content)

# Si l'utilisateur envoie un message
if send and prompt.strip():
    st.session_state.chat_history.append(("user", prompt))
    st.chat_message("user").markdown(prompt)

    st.chat_message("assistant")
    response_placeholder = st.empty()
    full_response = ""

    if model_choice == "GPT":
        stream = stream_gpt(prompt)
    elif model_choice == "Gemini":
        stream = stream_gemini(prompt)
    else:
        stream = stream_groq(prompt)

    for chunk in stream:
        full_response += chunk
        response_placeholder.markdown(full_response)

    st.session_state.chat_history.append(("assistant", full_response))

# Bouton pour effacer la conversation
if st.sidebar.button("🧹 Effacer la conversation"):
    st.session_state.chat_history = []
    st.rerun()

# Pied de page
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>✨ Propulsé par GPT, Groq et Gemini — "
    "Application Streamlit avec mémoire de chat.</div>",
    unsafe_allow_html=True,
)
