import streamlit as st
import numpy as np
import anthropic
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image


st.set_page_config(page_title="TrafficVision", page_icon="🚦", layout="centered")
st.title("🚦 TrafficVision")
st.write("AI-Powered Hand Traffic Signal Recognition")


@st.cache_resource
def load_traffic_model():
    return load_model("xception_model.h5")

model    = load_traffic_model()
img_size = model.input_shape[1:3]  # (299, 299)
class_names = ['Go', 'Stop', 'Turn Left', 'Turn Right']

# ── Anthropic client ──────────────────────────────────────────────
client = anthropic.Anthropic(api_key="YOUR_API_KEY_HERE")  # paste your key here

def get_genai_explanation(signal: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=60,
        messages=[{
            "role": "user",
            "content": (
                f"A traffic police officer is showing the '{signal}' hand signal. "
                f"Give a single short instruction for drivers in one sentence."
            )
        }]
    )
    return response.content[0].text.strip()

# ── Prediction ────────────────────────────────────────────────────
def predict(image: Image.Image):
    img = image.resize(img_size)
    img = np.array(img.convert("RGB"), dtype=np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    prediction  = model.predict(img, verbose=0)
    class_index = np.argmax(prediction)
    confidence  = prediction[0][class_index] * 100
    return class_names[class_index], confidence

# ── UI ────────────────────────────────────────────────────────────
st.markdown("---")
option = st.radio("Choose Input Method", ["📷 Webcam", "🖼️ Upload Image"], horizontal=True)

if option == "📷 Webcam":
    camera_image = st.camera_input("Point your camera at a hand signal")

    if camera_image:
        image = Image.open(camera_image)
        st.image(image, caption="Captured Frame", use_column_width=True)

        with st.spinner("Detecting signal..."):
            label, confidence = predict(image)

        # Signal result
        color_map = {"Go": "🟢", "Stop": "🔴", "Turn Left": "🟡", "Turn Right": "🟣"}
        emoji = color_map.get(label, "⚪")
        st.success(f"{emoji} **Detected Signal: {label}**  —  Confidence: {confidence:.1f}%")

        # GenAI explanation
        with st.spinner("Getting AI instruction..."):
            explanation = get_genai_explanation(label)
        st.info(f"💬 **AI Instruction:** {explanation}")

elif option == "🖼️ Upload Image":
    uploaded = st.file_uploader("Upload a hand signal image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Detecting signal..."):
            label, confidence = predict(image)

        color_map = {"Go": "🟢", "Stop": "🔴", "Turn Left": "🟡", "Turn Right": "🟣"}
        emoji = color_map.get(label, "⚪")
        st.success(f"{emoji} **Detected Signal: {label}**  —  Confidence: {confidence:.1f}%")

        with st.spinner("Getting AI instruction..."):
            explanation = get_genai_explanation(label)
        st.info(f"💬 **AI Instruction:** {explanation}")