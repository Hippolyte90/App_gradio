# pip install streamlit Flask==3.0.3 gunicorn==23.0.0 Werkzeug==3.0.3 python-dotenv numpy pandas scikit-learn matplotlib gensim openai requests Pillow
# pip install tiktoken faiss-cpu datasets sentencepiece google-generativeai unstructured plotly jupyter-dash pydub google-cloud-aiplatform
# pip install accelerate sentence_transformers feedparser speedtest-cli

import os
import streamlit as st
import requests
import io
import tempfile
import subprocess
import zipfile
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
from PIL import Image, ImageDraw, ImageFont
from video_utils import pil_images_from_uploaded_files, make_mp4_from_pil_images

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

def generate_image_dalle(prompt):
    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        return f"Erreur lors de la génération de l'image : {e}"


def crop_to_aspect(img: Image.Image, target_ratio: float) -> Image.Image:
    """Recadre l'image au centre pour respecter le ratio cible (width/height)."""
    w, h = img.size
    current_ratio = w / h
    if abs(current_ratio - target_ratio) < 1e-6:
        return img

    if current_ratio > target_ratio:
        # trop large => réduire la largeur
        new_w = int(target_ratio * h)
        left = (w - new_w) // 2
        box = (left, 0, left + new_w, h)
    else:
        # trop haut => réduire la hauteur
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        box = (0, top, w, top + new_h)
    return img.crop(box)

# -------------------------------
# 💬 Interface principale
# -------------------------------
st.title("🤖 Agent IA Multi-Tâches")
st.markdown("Discutez avec des modèles de langage ou générez des images.")

task_choice = st.selectbox("Quelle tâche souhaitez-vous effectuer ?", ["💬 Chat", "🎨 Génération d'images", "🎬 Génération de vidéos"])

if task_choice == "💬 Chat":
    st.header("💬 Chat Multi-Modèles (GPT | Gemini | Groq)")
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

elif task_choice == "🎨 Génération d'images":
    st.header("🎨 Génération d'images avec DALL-E 3")
    image_prompt = st.text_area("🖼️ 1. Décrivez l'image à générer :", height=100)
    overlay_text = st.text_input("✍️ 2. (Optionnel) Texte à ajouter sur l'image :")
    ratio_choice = st.selectbox("🖼️ 3. Choisir le format :", ["1:1", "16:9", "9:16"])
    generate_button = st.button("Générer l'image")

    if generate_button and image_prompt.strip():
        with st.spinner("Génération de l'image en cours..."):
            image_url = generate_image_dalle(image_prompt)
            if image_url and "Erreur" not in image_url:
                display_image_data = None
                try:
                    # Récupérer le contenu de l'image depuis l'URL
                    image_data = requests.get(image_url).content

                    # Ouvre l'image avec Pillow et force RGBA pour conserver canaux
                    img = Image.open(io.BytesIO(image_data)).convert("RGBA")

                    # Déterminer le ratio cible
                    if ratio_choice == "1:1":
                        target_ratio = 1.0
                    elif ratio_choice == "16:9":
                        target_ratio = 16.0 / 9.0
                    else:
                        target_ratio = 9.0 / 16.0

                    # Recadrer au ratio choisi (centré)
                    cropped = crop_to_aspect(img, target_ratio)

                    # Redimensionner si nécessaire pour garder une taille raisonnable (max 1024)
                    max_dim = 1024
                    w, h = cropped.size
                    scale = min(max_dim / max(w, h), 1.0)
                    if scale < 1.0:
                        new_size = (int(w * scale), int(h * scale))
                        cropped = cropped.resize(new_size, Image.LANCZOS)

                    # Si l'utilisateur a fourni du texte à ajouter, on l'ajoute maintenant
                    if overlay_text:
                        draw = ImageDraw.Draw(cropped)
                        # Taille de la police relative à la hauteur de l'image
                        try:
                            font_size = max(16, int(cropped.height * 0.06))
                            font = ImageFont.truetype("Arial.ttf", font_size)
                        except Exception:
                            font = ImageFont.load_default()

                        # Calculer position (bas à gauche avec marge relative)
                        margin_x = int(cropped.width * 0.03)
                        margin_y = int(cropped.height * 0.03)
                        text_w, text_h = draw.textsize(overlay_text, font=font)
                        position = (margin_x, cropped.height - text_h - margin_y)

                        # Ombre + texte pour lisibilité
                        draw.text((position[0] + 2, position[1] + 2), overlay_text, font=font, fill="black")
                        draw.text(position, overlay_text, font=font, fill="white")

                    # Sauvegarde l'image modifiée dans un buffer mémoire
                    buf = io.BytesIO()
                    cropped.save(buf, format="PNG")
                    display_image_data = buf.getvalue()

                    st.image(display_image_data, caption=f"Image générée ({ratio_choice}) pour : '{image_prompt}'")
                    filename = f"generated_image_{ratio_choice.replace(':','-')}.png"
                    st.download_button(label="📥 Télécharger l'image", data=display_image_data, file_name=filename, mime="image/png")
                except requests.exceptions.RequestException as e:
                    st.error(f"Erreur lors de la récupération de l'image : {e}")
                except Exception as e:
                    st.error(f"Une erreur est survenue lors du traitement de l'image : {e}")
            else:
                st.error(image_url)

elif task_choice == "🎬 Génération de vidéos":
    st.header("🎬 Génération de vidéos (frames → MP4)")
    video_prompt = st.text_area("🖼️ 1. Décrivez la scène ou le prompt de base :", height=100)
    uploaded_images = st.file_uploader("📁 2. (Option) Importer des images depuis votre ordinateur :", accept_multiple_files=True, type=["png", "jpg", "jpeg", "webp"])
    overlay_text_vid = st.text_input("✍️ 3. (Optionnel) Texte à ajouter sur chaque frame :")
    ratio_choice_vid = st.selectbox("🖼️ 4. Choisir le format :", ["16:9", "1:1", "9:16"])
    duration_value = st.number_input("Durée (valeur)", min_value=1, value=5)
    duration_unit = st.selectbox("Unité", ["secondes", "minutes"]) 
    fps = st.number_input("Images par seconde (fps)", min_value=1, max_value=60, value=12)
    generate_video_button = st.button("Générer la vidéo")

    if generate_video_button and (video_prompt.strip() or uploaded_images):
        with st.spinner("Génération des frames en cours..."):
            try:
                duration_seconds = int(duration_value) * (60 if duration_unit == "minutes" else 1)
                total_frames = max(1, int(duration_seconds * int(fps)))

                pil_images = []
                # Si l'utilisateur a uploadé des images, on les utilise
                if uploaded_images:
                    pil_images = pil_images_from_uploaded_files(uploaded_images)
                else:
                    # Générer frames via DALL·E selon le prompt
                    for i in range(total_frames):
                        frame_prompt = f"{video_prompt} (frame {i+1}/{total_frames})"
                        frame_url = generate_image_dalle(frame_prompt)
                        if not frame_url or "Erreur" in frame_url:
                            raise RuntimeError(f"Erreur lors de la génération de la frame {i+1}: {frame_url}")
                        frame_data = requests.get(frame_url).content
                        img = Image.open(io.BytesIO(frame_data)).convert("RGBA")

                        # Appliquer ratio et redimensionnement
                        if ratio_choice_vid == "1:1":
                            target_ratio = 1.0
                        elif ratio_choice_vid == "16:9":
                            target_ratio = 16.0 / 9.0
                        else:
                            target_ratio = 9.0 / 16.0
                        cropped = crop_to_aspect(img, target_ratio)
                        max_dim = 1024
                        w, h = cropped.size
                        scale = min(max_dim / max(w, h), 1.0)
                        if scale < 1.0:
                            new_size = (int(w * scale), int(h * scale))
                            cropped = cropped.resize(new_size, Image.LANCZOS)
                        pil_images.append(cropped)

                # Si on a des images uploadées, on doit distribuer les frames selon la durée
                if uploaded_images and pil_images:
                    n_imgs = len(pil_images)
                    repeats = -(-total_frames // n_imgs)  # ceil
                    frames_list = []
                    for img in pil_images:
                        for _ in range(repeats):
                            frames_list.append(img)
                    frames_list = frames_list[:total_frames]
                else:
                    frames_list = pil_images

                # Optionnel: ajouter overlay text (numéro de frame si plusieurs)
                if overlay_text_vid:
                    new_frames = []
                    for idx, img in enumerate(frames_list):
                        draw = ImageDraw.Draw(img)
                        try:
                            font_size = max(12, int(img.height * 0.05))
                            font = ImageFont.truetype("Arial.ttf", font_size)
                        except Exception:
                            font = ImageFont.load_default()
                        margin_x = int(img.width * 0.03)
                        margin_y = int(img.height * 0.03)
                        text = overlay_text_vid + (f" ({idx+1})" if len(frames_list) > 1 else "")
                        text_w, text_h = draw.textsize(text, font=font)
                        position = (margin_x, img.height - text_h - margin_y)
                        draw.text((position[0] + 2, position[1] + 2), text, font=font, fill="black")
                        draw.text(position, text, font=font, fill="white")
                        new_frames.append(img)
                    frames_list = new_frames

                # Créer la vidéo via imageio (utilise imageio[ffmpeg])
                try:
                    video_bytes = make_mp4_from_pil_images(frames_list, fps=int(fps))
                    st.video(video_bytes)
                    st.download_button(label="📥 Télécharger la vidéo", data=video_bytes, file_name="generated_video.mp4", mime="video/mp4")
                except Exception as e:
                    # Fallback: fournir un ZIP des frames
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w") as zf:
                        for i, img in enumerate(frames_list):
                            b = io.BytesIO()
                            img.save(b, format="PNG")
                            b.seek(0)
                            zf.writestr(f"frame_{i:04d}.png", b.read())
                    zip_buf.seek(0)
                    st.warning("Assemblée MP4 échouée — fourniture d'un ZIP des frames.")
                    st.download_button(label="📥 Télécharger les frames (ZIP)", data=zip_buf.getvalue(), file_name="frames.zip", mime="application/zip")
            except Exception as e:
                st.error(f"Erreur lors de la génération vidéo : {e}")

# Bouton pour effacer la conversation
if st.sidebar.button("🧹 Effacer la conversation"):
    st.session_state.chat_history = []
    st.rerun()

# Pied de page
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>✨ Propulsé par OpenAI, Groq et Gemini — "
    "Application Streamlit.</div>",
    unsafe_allow_html=True,
)
