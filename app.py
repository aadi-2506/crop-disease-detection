from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from utils.disease_info import get_disease_info, get_prediction_details


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "crop_disease_model.keras"
RESULTS_PATH = PROJECT_DIR / "models" / "evaluation_results.json"
DATASET_PATH = PROJECT_DIR / "dataset"

st.set_page_config(
    page_title="CropCare | Crop Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
        --green: #1f6b48;
        --deep: #123c2b;
        --lime: #dff2ce;
        --ink: #19352c;
        --muted: #536961;
        --canvas: #edf5f2;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: var(--canvas); color: var(--ink); }
    [data-testid="stAppViewContainer"] { background: var(--canvas); }
    [data-testid="stAppViewContainer"] .block-container {
        padding-top: 2.25rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }
    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: #163d2c;
        letter-spacing: -0.025em;
    }
    [data-testid="stAppViewContainer"] h1 {
        font-size: clamp(2.35rem, 4vw, 3.2rem);
        line-height: 1.15;
        margin-bottom: .6rem;
    }
    [data-testid="stAppViewContainer"] h2 { font-size: clamp(1.65rem, 2.8vw, 2.15rem); }
    [data-testid="stAppViewContainer"] h3 { font-size: 1.35rem; }
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] li,
    [data-testid="stAppViewContainer"] label {
        color: #253c34;
        font-size: 1.05rem;
        line-height: 1.65;
    }
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {
        color: var(--muted);
        font-size: .95rem;
    }
    [data-testid="stHeader"] {
        background: var(--canvas) !important;
    }
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] [data-testid="stDeployButton"] {
        color: var(--ink) !important;
    }
    [data-testid="stSidebarCollapsedControl"] button {
        color: var(--green) !important;
    }
    [data-testid="stSidebarCollapsedControl"] button svg {
        color: var(--green) !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f6b48 0%, #123c2b 100%);
        border-right: 1px solid #1d5c42;
        min-width: 260px !important;
        width: 260px !important;
        padding: 1.2rem 0.9rem 0.8rem 0.9rem;
        color: white !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
        color: white !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] svg {
        color: white !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] strong {
        color: white !important;
    }
    [data-testid="stSidebar"] .stRadio > div { gap: 0.3rem; }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 1.05rem;
        padding: 0.45rem 0.6rem;
        border-radius: 10px;
        color: white !important;
        display: flex;
        align-items: center;
        min-height: 2.4rem;
        background: rgba(255,255,255,0.03);
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.09);
    }
    [data-testid="stSidebar"] .stRadio input[type="radio"] {
        accent-color: #ffffff;
        width: 0.9rem;
        height: 0.9rem;
        margin-right: 0.6rem;
    }
    [data-testid="stSidebar"] .stSuccess,
    [data-testid="stSidebar"] .stWarning,
    [data-testid="stSidebar"] .stInfo {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        color: white !important;
    }
    [data-testid="stSidebar"] .stCaption {
        color: rgba(255,255,255,0.8) !important;
    }
    .hero {
        padding: 3.4rem 3.2rem;
        border: 1px solid rgba(255,255,255,.18);
        border-radius: 26px;
        color: white;
        box-shadow: 0 16px 42px rgba(18, 60, 43, .16);
        background: radial-gradient(circle at 90% 20%, #8bbd68 0, transparent 30%),
        linear-gradient(120deg, #143d2b 0%, #286b45 62%, #73a75c 100%); }
    .hero h1 {
        color: white !important;
        font-size: clamp(2.6rem, 5vw, 4rem);
        line-height: 1.1;
        margin-bottom: .9rem;
    }
    .hero p { color: #f0f7f2; max-width: 760px; font-size: 1.2rem; line-height: 1.75; }
    .eyebrow { text-transform: uppercase; letter-spacing: .15em; font-size: .9rem; font-weight: 700; color: #d8f0c6; }
    .home-cta-copy { text-align: center; padding: .5rem 0 1rem; }
    .home-cta-copy h2 { margin-bottom: .35rem; }
    .home-cta-copy p { color: #536961; font-size: 1.12rem; }
    .st-key-home_cta {
        background: white;
        border: 1px solid #e4ece3;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(28, 65, 43, .045);
        padding: 1.2rem 1.5rem;
    }
    .st-key-home_cta div.stButton > button {
        min-height: 3.5rem;
        font-size: 1.15rem;
        border-radius: 12px;
    }
    .panel {
        background: #ffffff;
        padding: 1.5rem 1.65rem;
        border: 1px solid #d7e7df;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(28, 65, 43, .06);
    }
    .panel h3 { color: #174832; }
    .soft-note { color: #536961; font-size: 1.05rem; line-height: 1.7; }
    [data-testid="stAppViewContainer"] .stRadio label,
    [data-testid="stAppViewContainer"] .stRadio label p,
    [data-testid="stAppViewContainer"] [data-testid="stFileUploader"] label,
    [data-testid="stAppViewContainer"] [data-testid="stFileUploader"] label p,
    [data-testid="stAppViewContainer"] [data-testid="stFileUploader"] small {
        color: #183326 !important;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio label p,
    [data-testid="stSidebar"] .stRadio label span {
        color: #ffffff !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stFileUploader"] section {
        background: #ffffff !important;
        border: 1px solid #dce8dc !important;
        border-radius: 12px;
    }
    [data-testid="stAppViewContainer"] [data-testid="stFileUploader"] section > div {
        color: #183326 !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stFileUploader"] button {
        background: #ffffff !important;
        color: #1f6b48 !important;
        border: 1px solid #b9d2c1 !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stFileUploader"] button:hover {
        background: #f0f7f1 !important;
        color: #164e34 !important;
        border-color: #1f6b48 !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stCameraInput"] {
        color: #183326 !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stCameraInput"] button {
        background: #ffffff !important;
        color: #1f6b48 !important;
        border: 1px solid #b9d2c1 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stCameraInput"] button:hover {
        background: #f0f7f1 !important;
        color: #164e34 !important;
        border-color: #1f6b48 !important;
    }
    .result-card { background: #fff; border: 1px solid #dce9dd; border-left: 5px solid #43a365;
        border-radius: 16px; padding: 1.3rem 1.5rem; }
    .status-healthy { color: #1e7748; font-weight: 700; }
    .status-disease { color: #b54e33; font-weight: 700; }
    .footer { margin-top: 3rem; padding: 1.2rem 0; border-top: 1px solid #dce8dc; color: #738078; font-size: .85rem; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d7e7df;
        border-top: 4px solid #67b989;
        border-radius: 14px;
        padding: 1rem 1.15rem;
        box-shadow: 0 8px 20px rgba(28, 65, 43, .05);
    }
    [data-testid="stMetricLabel"] { color: #536961 !important; font-size: 1rem !important; }
    [data-testid="stMetricValue"] { color: #19352c !important; font-size: 2rem !important; }
    [data-testid="stAlert"] {
        border-radius: 12px;
        font-size: 1.05rem;
    }
    div.stButton > button[kind="primary"] {
        background: #216c48;
        border: 0;
        border-radius: 10px;
        font-size: 1.05rem;
        font-weight: 700;
        min-height: 2.9rem;
        color: white !important;
    }
    div.stButton > button[kind="primary"] p { color: white !important; }
    div.stButton > button[kind="primary"]:hover { background: #164e34; color: white !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained MobileNetV2 model...")
def load_model(model_path: str, modified_time: float):
    del modified_time
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def read_model_metadata() -> dict | None:
    if not RESULTS_PATH.is_file():
        return None
    try:
        with RESULTS_PATH.open("r", encoding="utf-8") as result_file:
            return json.load(result_file)
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read model evaluation results: {exc}")
        return None


def render_footer() -> None:
    st.markdown(
        '<div class="footer">CropCare · Crop Disease Detection Using Machine Learning'
        " · College PBL demonstration project</div>",
        unsafe_allow_html=True,
    )


def render_home() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Smart farming · Computer vision</div>
          <h1>Healthy crops start<br>with an early diagnosis.</h1>
          <p>Upload a leaf photo or take one with your camera. CropCare uses a
          transfer-learned MobileNetV2 model to recognize common crop diseases
          and share practical next steps.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    with st.container(key="home_cta"):
        st.markdown(
            """
            <div class="home-cta-copy">
              <h2>Ready to check a leaf?</h2>
              <p>Upload a photo to get an instant crop health estimate.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _, button_column, _ = st.columns([1, 2, 1])
        with button_column:
            if st.button(
                "Start detection →",
                type="primary",
                key="home_start",
                use_container_width=True,
            ):
                st.session_state["next_navigation"] = "Disease Detection"
                st.rerun()
    st.write("")
    with st.container():
        st.markdown(
            """
            <div class="panel">
              <h3>How it works</h3>
              <p class="soft-note"><b>01 · Add a leaf image</b><br>Choose a photo
              from your device or capture one with a camera.</p>
              <p class="soft-note"><b>02 · Run the model</b><br>MobileNetV2
              analyzes visual patterns in the leaf.</p>
              <p class="soft-note"><b>03 · Understand the result</b><br>Review
              the prediction, confidence and simple care guidance.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.write("")
    c1, c2, c3 = st.columns(3)
    for column, title, body in [
        (c1, "Image based", "Works with a gallery upload or a fresh camera capture."),
        (c2, "Transfer learning", "Uses ImageNet-pretrained MobileNetV2."),
        (c3, "Measured results", "Reports held-out test metrics and a confusion matrix."),
    ]:
        with column:
            st.markdown(
                f'<div class="panel"><h3>{title}</h3><p class="soft-note">{body}</p></div>',
                unsafe_allow_html=True,
            )


def render_detection() -> None:
    st.title("Disease detection")
    st.markdown(
        '<p class="soft-note">Choose a clear, well-lit image with the leaf filling most of the frame.</p>',
        unsafe_allow_html=True,
    )
    image_source = st.radio(
        "Choose image source",
        ["Upload image", "Use camera"],
        horizontal=True,
        label_visibility="collapsed",
    )
    selected_image = None
    if image_source == "Upload image":
        selected_image = st.file_uploader(
            "Select a leaf image",
            type=["jpg", "jpeg", "png"],
            help="Accepted formats: JPG, JPEG and PNG.",
        )
    else:
        selected_image = st.camera_input(
            "Capture a leaf photo",
            help="Allow camera access if prompted, then press Take photo to capture the image.",
        )

    if selected_image is None:
        if image_source == "Use camera":
            st.info("Allow camera access, then press Take photo to capture a leaf image.")
        else:
            st.info("Add an image to preview it and enable disease detection.")
        return

    image_bytes = selected_image.getvalue()
    image_signature = hashlib.sha256(image_bytes).hexdigest()
    if st.session_state.get("image_signature") != image_signature:
        st.session_state.pop("last_prediction", None)
        st.session_state["image_signature"] = image_signature

    try:
        image = Image.open(selected_image)
        image.verify()
        selected_image.seek(0)
        image = Image.open(selected_image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("This image could not be opened. Please choose a valid JPG, JPEG or PNG file.")
        return

    preview, instructions = st.columns([1, 1], gap="large")
    with preview:
        st.image(image, caption="Selected leaf image", use_container_width=True)
    with instructions:
        st.markdown(
            """
            <div class="panel">
              <h3>Ready to analyze</h3>
              <p class="soft-note">The image will be resized to 224 × 224 pixels
              and analyzed by the trained model. The result is an estimate,
              not a laboratory diagnosis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Detect disease", type="primary", use_container_width=True):
            if not MODEL_PATH.is_file():
                st.error(
                    f"Trained model not found at {MODEL_PATH}. "
                    "Train locally with `python train.py`, then include the generated "
                    "model file in the deployment."
                )
                return
            try:
                metadata = read_model_metadata()
                if not metadata or not metadata.get("class_names"):
                    st.error(
                        "Model class labels are missing. Run `python train.py` "
                        "to create the model and its metadata."
                    )
                    return
                model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
                model_input = np.expand_dims(np.asarray(image.resize((224, 224)), dtype=np.float32), axis=0)
                probabilities = model.predict(model_input, verbose=0)[0]
                class_names = metadata["class_names"]
                if len(probabilities) != len(class_names):
                    st.error("The model output does not match the saved class labels. Retrain the model.")
                    return
                order = np.argsort(probabilities)[::-1]
                st.session_state["last_prediction"] = {
                    "class_name": class_names[int(order[0])],
                    "confidence": float(probabilities[int(order[0])]),
                    "top_predictions": [
                        {"class_name": class_names[int(index)], "confidence": float(probabilities[int(index)])}
                        for index in order[: min(5, len(order))]
                    ],
                }
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                st.error(f"Prediction could not be completed: {exc}")

    prediction = st.session_state.get("last_prediction")
    if prediction:
        st.divider()
        st.subheader("Prediction")
        crop, disease, status = get_prediction_details(prediction["class_name"])
        status_label = "Healthy" if status == "healthy" else "Diseased"
        status_class = "status-healthy" if status == "healthy" else "status-disease"
        st.markdown(
            f"""
            <div class="result-card">
              <div class="{status_class}">{status_label} leaf</div>
              <h2>{crop}</h2>
              <p><b>Disease:</b> {disease}<br>
              <b>Confidence:</b> {prediction["confidence"]:.1%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if status == "healthy":
            st.success("The model classified this leaf as healthy. Continue regular crop monitoring.")
        else:
            st.warning("The model detected a disease pattern. Confirm the result with a local agriculture expert.")
        info = get_disease_info(prediction["class_name"])
        st.subheader("Disease information")
        st.markdown(f"**{info['name']}** — {info['description']}")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown("**Common symptoms**")
            st.write(info["symptoms"])
        with info_col2:
            st.markdown("**Prevention and control**")
            st.write(info["control"])
        with st.expander("Other likely predictions"):
            for alternative in prediction["top_predictions"][1:]:
                st.write(
                    f"{get_prediction_details(alternative['class_name'])[1]} "
                    f"— {alternative['confidence']:.1%}"
                )


def render_performance() -> None:
    st.title("Model performance")
    if not MODEL_PATH.is_file():
        st.warning(
            "The trained model is not available in this deployment. Train it locally "
            "with `python train.py`, then include `models/crop_disease_model.keras` "
            "when deploying."
        )
    has_dataset_images = DATASET_PATH.is_dir() and any(
        path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        for path in DATASET_PATH.rglob("*")
    )
    if not has_dataset_images:
        st.info(
            "Dataset folder not found or empty. Download PlantVillage, extract it under "
            "`dataset/`, and run `python train.py`."
        )
    results = read_model_metadata()
    if results is None:
        st.info("Evaluation results appear here after `python evaluate.py` has been run.")
        return

    metrics = results.get("metrics", {})
    metric_columns = st.columns(4)
    for column, (label, key) in zip(
        metric_columns,
        [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1 score", "f1_score")],
    ):
        with column:
            value = metrics.get(key)
            st.metric(label, f"{value:.2%}" if isinstance(value, (int, float)) else "N/A")
    st.caption(
        f"Test samples: {results.get('test_samples', 'N/A')} · "
        f"Classes: {len(results.get('class_names', []))} · "
        f"Image size: {results.get('image_size', 224)} × {results.get('image_size', 224)}"
    )

    matrix = results.get("confusion_matrix")
    labels = results.get("class_names", [])
    if matrix and labels:
        st.subheader("Confusion matrix")
        figure_width = max(8, min(20, len(labels) * 0.45))
        fig, axis = plt.subplots(figsize=(figure_width, figure_width * 0.8))
        image = axis.imshow(matrix, interpolation="nearest", cmap="Greens")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set(
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=[label.replace("___", "\n").replace("_", " ") for label in labels],
            yticklabels=[label.replace("___", "\n").replace("_", " ") for label in labels],
            ylabel="True label",
            xlabel="Predicted label",
            title="Held-out test set",
        )
        plt.setp(axis.get_xticklabels(), rotation=65, ha="right", rotation_mode="anchor", fontsize=8)
        plt.setp(axis.get_yticklabels(), fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    else:
        st.info("Confusion matrix data is not available. Run `python evaluate.py`.")


def render_about() -> None:
    st.title("About the project")
    sections = [
        ("Problem statement", "Crop diseases reduce yield and income. Identifying a disease early from symptoms can be difficult without timely access to expert support."),
        ("Existing problem", "Visual inspection is manual and can be slow or inconsistent, especially when many plants need monitoring."),
        ("Proposed solution", "A Streamlit application accepts a leaf photo and uses transfer learning with MobileNetV2 to predict one of the classes learned from the dataset."),
        ("Machine learning model", "MobileNetV2 initialized with ImageNet weights, a frozen feature extractor, global average pooling, dropout and a dataset-sized softmax output layer."),
        ("Dataset", "PlantVillage leaf images downloaded from Kaggle. Class names are detected from the extracted class-folder names; the model does not assume a fixed number of classes."),
        ("Technologies", "Python, TensorFlow/Keras, MobileNetV2, NumPy, Pandas, Pillow, scikit-learn, Matplotlib and Streamlit."),
        ("Future scope", "Collect field images, validate predictions with agricultural experts, support regional languages, and deploy with ongoing monitoring and model updates."),
    ]
    for heading, description in sections:
        with st.expander(heading, expanded=heading == "Problem statement"):
            st.write(description)


with st.sidebar:
    st.markdown("# 🌿 CropCare")
    st.caption("AI-assisted crop health")
    pages = ["Home", "Disease Detection", "Model Performance", "About Project"]
    if "navigation" not in st.session_state:
        st.session_state["navigation"] = "Home"
    if "next_navigation" in st.session_state:
        st.session_state["navigation"] = st.session_state["next_navigation"]
        del st.session_state["next_navigation"]
    selected_page = st.radio(
        "Navigation",
        pages,
        key="navigation",
        label_visibility="collapsed",
    )
    st.caption("PlantVillage dataset · MobileNetV2")

if selected_page == "Home":
    render_home()
elif selected_page == "Disease Detection":
    render_detection()
elif selected_page == "Model Performance":
    render_performance()
else:
    render_about()

render_footer()
