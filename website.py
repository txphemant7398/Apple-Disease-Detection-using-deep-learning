import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Run via terminal: streamlit run app.py
# cd "D:\All Study Stuff\python\tf-env\python\HPCA Crop"
# python -m streamlit run website.py
st.set_page_config(
    page_title="Apple Leaf & Fruit Disease Recognition",
    page_icon="🍎",
    layout="centered"
)

# Exact alphabetical class order matching your 5 training folders (indices 0 to 4)
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blotch_Apple"
]

# Readable names for clean UI display
DISPLAY_NAMES = {
    "Apple___Apple_scab": "Apple Scab (Leaf Fungal Disease)",
    "Apple___Black_rot": "Apple Black Rot",
    "Apple___Cedar_apple_rust": "Cedar Apple Rust",
    "Apple___healthy": "Healthy Apple Leaf / Fruit",
    "Blotch_Apple": "Apple Blotch (Fruit Disease)"
}

# Treatment recommendations for all 5 Apple classes
TREATMENT_METHODS = {
    "Apple___Apple_scab": [
        "Remove and destroy fallen leaves to reduce overwintering fungal spores.",
        "Apply fungicides (e.g., sulfur or myclobutanil) during bud break and early leaf development.",
        "Prune trees to improve air circulation and speed up foliage drying.",
        "Plant scab-resistant apple varieties such as 'Enterprise' or 'Freedom'."
    ],
    "Apple___Black_rot": [
        "Prune and destroy infected branches, cankers, and mummified fruits.",
        "Apply protective fungicides (e.g., captan) from bloom through petal fall.",
        "Remove dead wood from the orchard to reduce fungal inoculum.",
        "Maintain strict orchard sanitation throughout the growing season."
    ],
    "Apple___Cedar_apple_rust": [
        "Remove nearby cedar or juniper trees (alternate host) within a few hundred yards if feasible.",
        "Apply fungicides (e.g., myclobutanil) from pink bud stage through petal fall.",
        "Plant rust-resistant apple varieties such as 'Liberty' or 'Redfree'.",
        "Improve canopy air circulation through regular pruning."
    ],
    "Apple___healthy": [
        "No treatment needed. Maintain regular watering, balanced fertilization, and seasonal pruning to prevent future infections."
    ],
    "Blotch_Apple": [
        "Prune dead twigs, cankers, and infected branches during dormancy to remove overwintering spores.",
        "Apply protective fungicides (e.g., captan or mancozeb) starting at petal fall and continuing through summer.",
        "Thin fruit clusters and open the tree canopy to improve airflow and drying after rain.",
        "Collect and destroy infected fallen fruit from the orchard floor."
    ]
}

# Cache the trained MobileNetV2 model in RAM so it loads only once
@st.cache_resource
def load_disease_model():
    model_path = "apple_disease_model_v2.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in {os.getcwd()}. Place app.py in the same folder as your saved .keras model.")
        st.stop()
    return tf.keras.models.load_model(model_path)

model = load_disease_model()

# Single-image inference function
def model_prediction(test_image):
    # Convert to RGB to prevent 4-channel RGBA errors on PNG uploads
    image = Image.open(test_image).convert("RGB").resize((128, 128))
    input_arr = np.array(image, dtype=np.float32)
    input_arr = np.expand_dims(input_arr, axis=0)  # Shape: (1, 128, 128, 3)

    # Note: mobilenet_v2.preprocess_input is already built inside the saved model
    predictions = model.predict(input_arr, verbose=0)[0]
    result_index = int(np.argmax(predictions))
    confidence = float(predictions[result_index] * 100)
    return result_index, confidence, predictions

# Sidebar Navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# -------------------- HOME PAGE --------------------
if app_mode == "Home":
    st.header("🍎 APPLE LEAF & FRUIT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)

    st.markdown("""
    Welcome to the **Apple Leaf & Fruit Disease Recognition System**! 🌿🍎

    Our system uses deep learning (**MobileNetV2 Transfer Learning**) to inspect both **apple leaves and apple fruits**, identifying common diseases early so you can take timely action and protect orchard yield.

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page in the sidebar and upload a photo of an apple leaf or fruit.
    2. **AI Analysis:** The model processes the image and classifies it across 5 unified leaf and fruit conditions.
    3. **Actionable Guidance:** View the predicted condition, confidence breakdown, and recommended orchard treatment steps.

    ### Supported Conditions
    * **Apple Scab** (`Apple___Apple_scab`)
    * **Apple Black Rot** (`Apple___Black_rot`)
    * **Cedar Apple Rust** (`Apple___Cedar_apple_rust`)
    * **Apple Blotch** (`Blotch_Apple`)
    * **Healthy Leaf & Fruit** (`Apple___healthy`)
    """)

# -------------------- ABOUT PAGE --------------------
elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
    #### Dataset & Unified Architecture
    This system is trained on a unified **Apple Leaf & Fruit Disease Dataset** covering **5 classes** across both foliage and fruit surfaces (`Apple___Apple_scab`, `Apple___Black_rot`, `Apple___Cedar_apple_rust`, `Apple___healthy`, and `Blotch_Apple`), merging healthy leaf and healthy fruit samples into a single unified healthy category.

    #### Model Pipeline
    * **Backbone:** Pretrained `MobileNetV2` (`128 × 128` RGB input) with on-the-fly data augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`).
    * **Training Strategy:** Two-phase transfer learning—Phase 1 trains the 5-class classification head, and Phase 2 fine-tunes the top 30 convolutional layers with a low learning rate (`1e-4`) to capture fine fungal lesion and fruit blotch textures.
    * **Deployment:** Lightweight Streamlit web interface with cached model loading (`@st.cache_resource`) for fast CPU inference.
    """)

# -------------------- DISEASE RECOGNITION PAGE --------------------
elif app_mode == "Disease Recognition":
    st.header("🔍 Apple Disease Recognition")
    test_image = st.file_uploader("Upload an Apple Leaf or Fruit Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict Disease"):
            with st.spinner("Analyzing leaf/fruit texture..."):
                try:
                    result_index, confidence, all_preds = model_prediction(test_image)
                    raw_label = CLASS_NAMES[result_index]
                    readable_label = DISPLAY_NAMES[raw_label]

                    if raw_label == "Apple___healthy":
                        st.success(f"**Prediction:** {readable_label} ({confidence:.2f}% confidence)")
                    else:
                        st.warning(f"**Detected Condition:** {readable_label} ({confidence:.2f}% confidence)")

                    # Display Treatment Methods
                    st.subheader("🩺 Recommended Treatment Methods")
                    treatments = TREATMENT_METHODS.get(
                        raw_label,
                        ["No specific treatments available. Consult a local agricultural extension officer."]
                    )
                    for i, treatment in enumerate(treatments, 1):
                        st.markdown(f"{i}. {treatment}")

                    # Display Confidence Breakdown across all 5 classes
                    with st.expander("View Confidence Scores for All 5 Classes"):
                        for idx, cls in enumerate(CLASS_NAMES):
                            st.write(f"**{DISPLAY_NAMES[cls]}**: {all_preds[idx] * 100:.2f}%")
                            st.progress(float(all_preds[idx]))

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    else:
        st.info("Please upload an image above to begin diagnosis.")