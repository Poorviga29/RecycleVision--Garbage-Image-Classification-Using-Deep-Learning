import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# PAGE CONFIG

st.set_page_config(
    page_title="RecycleVision AI",
    page_icon="♻️",
    layout="wide"
)

# LOAD MODEL

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("RecycleVision_BestModel.h5")

model = load_model()

NUM_CLASSES = model.output_shape[-1]


# CLASS NAMES (12 REAL CLASSES)

DISPLAY_NAMES = [
    "Battery", "Biological Waste", "Brown Glass", "Cardboard", "Clothes", "Green Glass",
    "Metal", "Paper", "Plastic", "Shoes", "Trash", "White Glass"
]

if len(DISPLAY_NAMES) != NUM_CLASSES:
    DISPLAY_NAMES = DISPLAY_NAMES[:NUM_CLASSES]

# DISPOSAL TIPS (USED ONCE)

DISPOSAL_GUIDE = {
    "Battery": " Hazardous waste. Dispose at certified battery recycling centers.",
    "Biological Waste": "Use yellow bio-medical waste bins.",
    "Brown Glass": "Recycle through brown-glass containers.",
    "Cardboard": "Flatten and put in paper/cardboard recycling.",
    "Clothes": "Donate if usable; recycle textile waste.",
    "Green Glass": "Recycle under green-glass containers.",
    "Metal": "Rinse and place in metal recycling.",
    "Paper": "Recycle clean, dry paper only.",
    "Plastic": "Recycle based on plastic code; avoid contaminated plastic.",
    "Shoes": "Donate wearable pairs; recycle rubber soles.",
    "Trash": "Non-recyclable waste. Send to landfill.",
    "White Glass": "Recycle separately from colored glass."
}

# PREDICTION FUNCTION

def predict(image):
    img = image.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)[0]
    top_idx = preds.argsort()[-3:][::-1]
    return [(DISPLAY_NAMES[i], float(preds[i])) for i in top_idx]

# MENU

menu = st.sidebar.radio(
    "☰ MENU",
    ["🏠 Home", "🖼️ Classify Waste", "📚 Waste Categories & Disposal Guide", "ℹ️ About Project"]
)

# HOME PAGE

if menu == "🏠 Home":
    st.title("♻️ RecycleVision – AI Waste Classification System")

    st.write("""
    **RecycleVision** is a Deep Learning powered garbage classification system  
    that identifies waste types from an uploaded image.

    ### 🌟 Features
    - Upload garbage images  
    - AI predicts the waste category  
    - Displays top-3 predictions with confidence  
    - Single combined page for waste learning + disposal instructions  
    - Modern, simple, and clean UI  
    """)

# CLASSIFICATION PAGE

elif menu == "🖼️ Classify Waste":
    st.title("🖼️ Garbage Image Classification")

    file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing image with AI..."):
            top3 = predict(img)

        label, score = top3[0]

        st.subheader("📌 Prediction")
        st.success(f"Category: **{label}** ({score*100:.2f}% confidence)")

        st.subheader("📊 Top-3 Confidence Chart")
        st.bar_chart({name: conf for name, conf in top3})

        st.subheader("🏆 Top-3 Predictions")
        for name, conf in top3:
            st.write(f"➡ **{name}** — {conf*100:.2f}%")


# COMBINED: LEARN + DISPOSAL GUIDE

elif menu == "📚 Waste Categories & Disposal Guide":
    st.title("📚 Waste Categories & Smart Disposal Guide")

    st.write("Click any category to learn what it means and how to dispose it properly:")

    for name in DISPLAY_NAMES:
        with st.expander(f" {name}"):
            st.write(f"**Disposal Method:** {DISPOSAL_GUIDE[name]}")

# ABOUT PROJECT PAGE

elif menu == "ℹ️ About Project":
    st.title("ℹ️ About This Project")

    st.write("""
    **RecycleVision AI**  
    A Deep Learning project classifying waste into 12 categories using  
    **CNN + Transfer Learning (TensorFlow/Keras)**.

    ### 🔧 Technologies Used
    - Python  
    - TensorFlow / Keras  
    - Transfer Learning  
    - Streamlit  
    - Image Processing (PIL)

    ### 🎯 Objective
    Automate waste classification and promote correct disposal habits.

    ### 📦 Deliverables
    - Processed dataset  
    - Trained model  
    - Streamlit app with real-time prediction  
    """)

st.markdown("---")
st.caption("RecycleVision - Garbage Image Classification")
