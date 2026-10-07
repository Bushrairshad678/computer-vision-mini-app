import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Bean Leaf Disease Detector", page_icon="🌿", layout="wide")

CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
MODEL_PATH = "beans_model.keras"
IMG_SIZE = 224


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


def predict(img_rgb: np.ndarray) -> np.ndarray:
    """img_rgb: uint8 RGB image (H, W, 3). Returns class probabilities."""
    model = load_model()
    x = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype("float32")
    x = np.expand_dims(x, axis=0)  # raw 0-255 pixels; scaling is done inside the model
    return model.predict(x, verbose=0)[0]


# ----------------------------------------------------------------------------
# Image processing helpers
# ----------------------------------------------------------------------------
def to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert grayscale / single-channel images to 3-channel RGB."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def zoom_image(img, factor):
    h, w = img.shape[:2]
    ch, cw = int(h / factor), int(w / factor)
    y1, x1 = (h - ch) // 2, (w - cw) // 2
    crop = img[y1:y1 + ch, x1:x1 + cw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)


def resize_image(img, new_w, new_h):
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def average_blur(img, k):
    return cv2.blur(img, (k, k))


def gaussian_blur(img, k):
    return cv2.GaussianBlur(img, (k, k), 0)


def median_blur(img, k):
    return cv2.medianBlur(img, k)


def hist_equalization(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def clahe_equalization(img, clip):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def canny_edges(img, t1, t2):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, t1, t2)


def sobel_edges(img, k):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
    mag = cv2.magnitude(gx, gy)
    return cv2.convertScaleAbs(mag)


def laplacian_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def sharpen(img, amount):
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def adjust_brightness_contrast(img, alpha, beta):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def negative(img):
    return 255 - img


def binary_threshold(img, t):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return th


def morphology(img, op, k):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((k, k), np.uint8)
    ops = {"Erosion": cv2.MORPH_ERODE, "Dilation": cv2.MORPH_DILATE,
           "Opening": cv2.MORPH_OPEN, "Closing": cv2.MORPH_CLOSE}
    return cv2.morphologyEx(gray, ops[op], kernel)


def plot_histograms(img_rgb, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for i, (c, name) in enumerate(zip(("r", "g", "b"), ("Red", "Green", "Blue"))):
        h = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        axes[0].plot(h, color=c, label=name)
    axes[0].set_title(f"{title} - RGB Histogram")
    axes[0].set_xlim([0, 255])
    axes[0].legend()
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    axes[1].hist(gray.ravel(), bins=256, range=(0, 255), color="gray")
    axes[1].set_title(f"{title} - Grayscale Histogram")
    axes[1].set_xlim([0, 255])
    fig.tight_layout()
    return fig


def odd(n):
    return n if n % 2 == 1 else n + 1


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🌿 Bean Leaf Disease Detector")
st.caption("Computer Vision Lab Assignment | Image Enhancement + MobileNetV2 Classification")

uploaded = st.file_uploader("Upload a bean leaf image", type=["jpg", "jpeg", "png", "bmp", "webp"])

TECHNIQUES = [
    "None (Original)",
    "Zoom",
    "Resize",
    "Average Blur",
    "Gaussian Blur",
    "Median Blur",
    "Histogram Equalization",
    "CLAHE (Adaptive Histogram)",
    "Edge Detection - Canny",
    "Edge Detection - Sobel",
    "Edge Detection - Laplacian",
    "Grayscale",
    "Sharpen",
    "Brightness / Contrast",
    "Negative",
    "Binary Threshold",
    "Morphology",
]

st.sidebar.header("⚙️ Image Processing")
technique = st.sidebar.selectbox("Technique chunein", TECHNIQUES)

if uploaded is None:
    st.info("Upload an image above to get started.")
    st.stop()

original = np.array(Image.open(uploaded).convert("RGB"))
processed = original.copy()
show_hist = False

# ---- Parameters + processing -------------------------------------------------
if technique == "Zoom":
    factor = st.sidebar.slider("Zoom factor", 1.0, 5.0, 2.0, 0.1)
    processed = zoom_image(original, factor)

elif technique == "Resize":
    h0, w0 = original.shape[:2]
    scale = st.sidebar.slider("Scale (%)", 10, 300, 50, 5)
    processed = resize_image(original, max(1, int(w0 * scale / 100)), max(1, int(h0 * scale / 100)))

elif technique == "Average Blur":
    k = odd(st.sidebar.slider("Kernel size", 3, 51, 9, 2))
    processed = average_blur(original, k)

elif technique == "Gaussian Blur":
    k = odd(st.sidebar.slider("Kernel size", 3, 51, 9, 2))
    processed = gaussian_blur(original, k)

elif technique == "Median Blur":
    k = odd(st.sidebar.slider("Kernel size", 3, 31, 5, 2))
    processed = median_blur(original, k)

elif technique == "Histogram Equalization":
    processed = hist_equalization(original)
    show_hist = True

elif technique == "CLAHE (Adaptive Histogram)":
    clip = st.sidebar.slider("Clip limit", 1.0, 10.0, 3.0, 0.5)
    processed = clahe_equalization(original, clip)
    show_hist = True

elif technique == "Edge Detection - Canny":
    t1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
    t2 = st.sidebar.slider("Threshold 2", 0, 255, 200)
    processed = canny_edges(original, t1, t2)

elif technique == "Edge Detection - Sobel":
    k = st.sidebar.select_slider("Kernel size", options=[1, 3, 5, 7], value=3)
    processed = sobel_edges(original, k)

elif technique == "Edge Detection - Laplacian":
    processed = laplacian_edges(original)

elif technique == "Grayscale":
    processed = to_gray(original)

elif technique == "Sharpen":
    amt = st.sidebar.slider("Sharpen amount", 0.5, 5.0, 1.5, 0.1)
    processed = sharpen(original, amt)

elif technique == "Brightness / Contrast":
    alpha = st.sidebar.slider("Contrast (alpha)", 0.5, 3.0, 1.2, 0.1)
    beta = st.sidebar.slider("Brightness (beta)", -100, 100, 20)
    processed = adjust_brightness_contrast(original, alpha, beta)

elif technique == "Negative":
    processed = negative(original)

elif technique == "Binary Threshold":
    t = st.sidebar.slider("Threshold", 0, 255, 127)
    processed = binary_threshold(original, t)

elif technique == "Morphology":
    op = st.sidebar.selectbox("Operation", ["Erosion", "Dilation", "Opening", "Closing"])
    k = st.sidebar.slider("Kernel size", 1, 15, 3)
    processed = morphology(original, op, k)

# ---- Side by side view -------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original Image")
    st.image(original, use_container_width=True)
    st.caption(f"Size: {original.shape[1]} x {original.shape[0]}")
with col2:
    st.subheader(f"Processed: {technique}")
    st.image(processed, use_container_width=True, clamp=True)
    st.caption(f"Size: {processed.shape[1]} x {processed.shape[0]}")

# ---- Histograms --------------------------------------------------------------
if technique.startswith("Histogram") or show_hist:
    st.subheader("📊 Histograms (Before vs After)")
    st.pyplot(plot_histograms(original, "Original"))
    st.pyplot(plot_histograms(to_rgb(processed), "Processed"))
else:
    with st.expander("📊 View histogram of the original image"):
        st.pyplot(plot_histograms(original, "Original"))

# ---- Download ----------------------------------------------------------------
ok, buf = cv2.imencode(".png", cv2.cvtColor(to_rgb(processed), cv2.COLOR_RGB2BGR))
if ok:
    st.download_button("⬇️ Download processed image", buf.tobytes(),
                       file_name="processed.png", mime="image/png")

# ---- Prediction --------------------------------------------------------------
st.divider()
st.subheader("🔍 Model Prediction")
source = st.radio("Run prediction on:",
                  ["Original image", "Processed image"], horizontal=True)

if st.button("Predict", type="primary"):
    target = original if source == "Original image" else to_rgb(processed)
    try:
        probs = predict(target)
    except Exception as e:
        st.error(f"Could not load the model or run the prediction. Make sure `{MODEL_PATH}` is in the same folder as app.py.\n\n{e}")
        st.stop()

    top = int(np.argmax(probs))
    label = CLASS_NAMES[top].replace("_", " ").title()
    conf = probs[top] * 100

    if CLASS_NAMES[top] == "healthy":
        st.success(f"Prediction: **{label}** ({conf:.2f}% confidence)")
    else:
        st.warning(f"Prediction: **{label}** ({conf:.2f}% confidence)")

    st.write("Class probabilities:")
    for name, p in zip(CLASS_NAMES, probs):
        st.progress(float(p), text=f"{name.replace('_', ' ').title()}: {p * 100:.2f}%")
