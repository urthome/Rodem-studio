
import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Rodem Studio Web",
    page_icon="🧺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {max-width: 760px; padding-top: 1.2rem; padding-bottom: 2rem;}
.stButton button, .stDownloadButton button {height: 3rem; font-size: 1.05rem;}
[data-testid="stFileUploaderDropzone"] {min-height: 150px;}
</style>
""", unsafe_allow_html=True)

st.title("Rodem Studio Web")
st.caption("휴대폰이나 PC에서 발매트 사진을 올리고 투명 PNG로 저장하세요.")

def resize_for_processing(img_bgr, max_side=1400):
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        img_bgr = cv2.resize(
            img_bgr,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return img_bgr, scale

def largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    if n <= 1:
        return mask
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == idx).astype(np.uint8)

def auto_cutout(pil_img, sensitivity=36, feather=2):
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    small, scale = resize_for_processing(bgr)
    h, w = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)

    band = max(6, int(min(h, w) * 0.035))
    border = np.concatenate([
        lab[:band].reshape(-1, 3),
        lab[-band:].reshape(-1, 3),
        lab[:, :band].reshape(-1, 3),
        lab[:, -band:].reshape(-1, 3),
    ], axis=0)

    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)

    edge = max(4, int(min(h, w) * 0.012))
    gc[:edge, :] = cv2.GC_BGD
    gc[-edge:, :] = cv2.GC_BGD
    gc[:, :edge] = cv2.GC_BGD
    gc[:, -edge:] = cv2.GC_BGD

    gc[dist < sensitivity * 0.55] = cv2.GC_BGD
    gc[(dist >= sensitivity * 0.55) & (dist < sensitivity)] = cv2.GC_PR_BGD
    gc[dist > sensitivity * 1.35] = cv2.GC_PR_FGD

    y1, y2 = int(h * 0.12), int(h * 0.88)
    x1, x2 = int(w * 0.12), int(w * 0.88)
    center = gc[y1:y2, x1:x2]
    center[center == cv2.GC_PR_BGD] = cv2.GC_PR_FGD
    gc[y1:y2, x1:x2] = center

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(small, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        mask = np.where(
            (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1, 0
        ).astype(np.uint8)
    except Exception:
        mask = (dist > sensitivity).astype(np.uint8)

    mask = largest_component(mask)
    k = max(3, int(min(h, w) * 0.006) | 1)
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )

    alpha = (mask * 255).astype(np.uint8)
    if feather > 0:
        blur = int(feather * 2 + 1)
        alpha = cv2.GaussianBlur(alpha, (blur, blur), 0)

    if scale < 1:
        alpha = cv2.resize(
            alpha,
            (bgr.shape[1], bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba)

def checker_preview(rgba, size=18):
    arr = np.array(rgba)
    h, w = arr.shape[:2]
    yy, xx = np.indices((h, w))
    checks = ((xx // size + yy // size) % 2) * 35 + 220
    bg = np.dstack([checks, checks, checks]).astype(np.uint8)
    a = arr[:, :, 3:4].astype(np.float32) / 255.0
    out = (arr[:, :, :3] * a + bg * (1 - a)).astype(np.uint8)
    return Image.fromarray(out)

uploaded = st.file_uploader(
    "발매트 사진 선택",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
    help="사진 전체에 발매트가 다 나오고, 제품과 배경의 색 차이가 클수록 잘 됩니다.",
)

with st.expander("결과 조절"):
    sensitivity = st.slider(
        "배경 제거 강도",
        min_value=18,
        max_value=70,
        value=36,
        step=1,
        help="배경이 남으면 올리고, 발매트가 잘리면 내리세요.",
    )
    feather = st.slider(
        "가장자리 부드럽게",
        min_value=0,
        max_value=6,
        value=2,
        step=1,
    )

if uploaded is not None:
    try:
        img = Image.open(uploaded)
        img = ImageOps.exif_transpose(img).convert("RGB")

        with st.spinner("발매트를 분리하고 있습니다..."):
            result = auto_cutout(
                img,
                sensitivity=sensitivity,
                feather=feather,
            )

        st.subheader("원본")
        st.image(img, use_container_width=True)

        st.subheader("자동 배경 제거 결과")
        st.image(checker_preview(result), use_container_width=True)

        buf = io.BytesIO()
        result.save(buf, format="PNG")

        st.download_button(
            label="투명 PNG 저장",
            data=buf.getvalue(),
            file_name="rodem_mat_cutout.png",
            mime="image/png",
            use_container_width=True,
        )

        st.info(
            "갈색 바닥이 남으면 강도를 조금 올리고, "
            "발매트 털이 잘리면 강도를 낮춰 다시 확인하세요."
        )
    except Exception as exc:
        st.error(f"사진 처리 중 오류가 발생했습니다: {exc}")
else:
    st.write("위 버튼을 눌러 휴대폰 사진을 선택하세요.")

st.divider()
st.caption("v0.1 · 이번 버전은 자동 배경 제거 기능만 포함합니다.")
