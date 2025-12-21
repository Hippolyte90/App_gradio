import os
import io
import tempfile
import numpy as np
from PIL import Image
import imageio


def pil_images_from_uploaded_files(uploaded_files):
    """Convertit une liste de fichiers uploadés Streamlit en PIL Images."""
    images = []
    for f in uploaded_files:
        img = Image.open(f).convert("RGBA")
        images.append(img)
    return images


def make_mp4_from_pil_images(images, fps=12):
    """Crée un MP4 à partir d'une liste de PIL Images et renvoie les bytes.

    Utilise imageio (ffmpeg). Les images sont redimensionnées à la taille de la
    première image si nécessaire.
    """
    if not images:
        raise ValueError("Aucune image fournie")

    base_w, base_h = images[0].size
    frames = []
    for img in images:
        if img.size != (base_w, base_h):
            img = img.resize((base_w, base_h), Image.LANCZOS)
        frames.append(img.convert("RGB"))

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        writer = imageio.get_writer(tmp_path, fps=fps, codec='libx264')
        for frame in frames:
            arr = np.array(frame)
            writer.append_data(arr)
        writer.close()
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return data
