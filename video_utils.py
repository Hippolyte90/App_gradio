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


def animate_single_image(img: Image.Image, total_frames: int, zoom_end: float = 1.12) -> list:
    """Crée une liste de PIL frames animant une seule image (effet Ken Burns simple).

    zoom_end: facteur de zoom final (>1 -> zoom in)
    """
    if total_frames <= 1:
        return [img]

    w, h = img.size
    frames = []
    # start center and end center (slight pan)
    start_cx, start_cy = w * 0.5, h * 0.5
    end_cx, end_cy = w * 0.5 + w * 0.05, h * 0.5 + h * 0.03

    for i in range(total_frames):
        t = i / (total_frames - 1)
        scale = 1.0 + (zoom_end - 1.0) * t
        cx = start_cx + (end_cx - start_cx) * t
        cy = start_cy + (end_cy - start_cy) * t

        crop_w = int(w / scale)
        crop_h = int(h / scale)
        left = int(max(0, min(w - crop_w, cx - crop_w // 2)))
        top = int(max(0, min(h - crop_h, cy - crop_h // 2)))
        box = (left, top, left + crop_w, top + crop_h)
        frame = img.crop(box).resize((w, h), Image.LANCZOS)
        frames.append(frame.convert("RGBA"))

    return frames


def animate_uploaded_images(images: list, total_frames: int) -> list:
    """Anime une liste d'images uploadées.

    - Si une seule image: applique `animate_single_image`.
    - Si plusieurs: crée des transitions par fondu entre images pour remplir `total_frames`.
    """
    if not images:
        return []
    if len(images) == 1:
        return animate_single_image(images[0], total_frames)

    n = len(images)
    # nombre total de transitions = n-1
    transitions = n - 1
    # frames par transition (au moins 1)
    per = max(1, total_frames // transitions)
    frames = []
    # Precompute RGB images of same size
    base_w, base_h = images[0].size
    imgs_rgb = [im.convert("RGBA").resize((base_w, base_h), Image.LANCZOS) for im in images]

    for idx in range(transitions):
        a = imgs_rgb[idx]
        b = imgs_rgb[idx + 1]
        for f in range(per):
            t = f / per
            blended = Image.blend(a, b, t)
            frames.append(blended.convert("RGBA"))

    # If not enough frames, append last image
    while len(frames) < total_frames:
        frames.append(imgs_rgb[-1].convert("RGBA"))

    # Trim if too many
    frames = frames[:total_frames]
    return frames
