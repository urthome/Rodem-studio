import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Rodem Studio Web", page_icon="🧺", layout="centered")
st.title("Rodem Studio Web v0.2")
st.caption("가벼운 시험판 · 발매트 사진을 올리면 배경을 제거합니다.")

def resize_image(pil_img, max_side=1000):
    w, h = pil_img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1:
        pil_img = pil_img.resize((int(w * scale), int(h * scale)))
    return pil_img

def remove_background_fast(pil_img, threshold=42, feather=3):
    pil_img = resize_image(pil_img, 1000)
    rgb = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]

    band = max(5, int(min(h, w) * 0.04))
    border = np.concatenate([
        lab[:band].reshape(-1, 3),
        lab[-band:].reshape(-1, 3),
        lab[:, :band].reshape(-1, 3),
        lab[:, -band:].reshape(-1, 3)
    ], axis=0)

    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    mask = (dist > threshold).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == idx).astype(np.uint8)

    alpha = (mask * 255).astype(np.uint8)
    if feather > 0:
        k = feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)

    return Image.fromarray(np.dstack([rgb, alpha]))

def checker_preview(rgba, size=16):
    arr = np.array(rgba)
    h, w = arr.shape[:2]
    yy, xx = np.indices((h, w))
    checks = ((xx // size + yy // size) % 2) * 35 + 220
    bg = np.dstack([checks, checks, checks]).astype(np.uint8)
    a = arr[:, :, 3:4].astype(np.float32) / 255.0
    out = (arr[:, :, :3] * a + bg * (1 - a)).astype(np.uint8)
    return Image.fromarray(out)

uploaded = st.file_uploader("발매트 사진 선택", type=["jpg", "jpeg", "png", "webp"])

with st.expander("결과 조절"):
    threshold = st.slider("배경 제거 강도", 20, 80, 42, 1)
    feather = st.slider("가장자리 부드럽게", 0, 6, 3, 1)

if uploaded is None:
    st.info("사진을 선택하면 결과가 바로 아래에 나타납니다.")
else:
    try:
        img = Image.open(uploaded)
        img = ImageOps.exif_transpose(img).convert("RGB")
        st.success("사진 업로드 완료")
        st.image(img, caption="원본", use_container_width=True)

        with st.spinner("배경을 제거하고 있습니다..."):
            result = remove_background_fast(img, threshold, feather)

        st.image(checker_preview(result), caption="배경 제거 결과", use_container_width=True)

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.download_button(
            "투명 PNG 저장",
            data=buf.getvalue(),
            file_name="rodem_mat_cutout.png",
            mime="image/png",
            use_container_width=True
        )
    except Exception as e:
        st.exception(e)
