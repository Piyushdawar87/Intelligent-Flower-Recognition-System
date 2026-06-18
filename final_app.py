
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="🌸 AI Flower Recognition Dashboard",
                   page_icon="🌸",
                   layout="wide")

st.markdown("""
<style>
.block-container {padding-top:1rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_flower_model():
    model_path = "flower_model_trained.hdf5"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    return tf.keras.models.load_model(model_path)

model = load_flower_model()

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


FLOWER_COLORS = {
    "rose": "#FF1744",
    "sunflower": "#FFD600",
    "daisy": "#FF80AB",
    "tulip": "#2979FF",
    "dandelion": "#AA00FF"
}


flower_info = {
    "daisy":"Symbol of innocence and purity.The daisy is a charming flower known for its white petals and bright yellow center. It is commonly used in gardens, floral arrangements, decorations, and gifting due to its simple yet elegant appearance. Daisies symbolize innocence, purity, happiness, and new beginnings, making them a popular choice for expressing positivity and joy",
    "dandelion":"Medicinal flowering plant.The dandelion is a hardy flowering plant recognized for its bright yellow flowers and fluffy seed heads. It is commonly found in gardens, fields, and grasslands and has been traditionally used in herbal remedies, teas, and natural health products. Dandelions symbolize resilience, hope, and the ability to thrive in challenging conditions, making them a powerful representation of strength and perseverance.",
    "rose":"Popular ornamental flower.The rose is one of the most popular and admired flowers, known for its beauty, fragrance, and variety of colors. It is widely used in bouquets, decorations, perfumes, cosmetics, and special occasions such as weddings and celebrations. Roses symbolize love, passion, beauty, and admiration, making them a timeless flower cherished around the world.",
    "sunflower":"Known for heliotropism.The sunflower is a bright and cheerful flower known for its large yellow petals and its tendency to turn toward the sun. It is widely used in gardens, landscaping, floral decorations, and for producing sunflower seeds and oil. Sunflowers symbolize happiness, positivity, loyalty, and strength, making them one of the most admired flowers worldwide",
    "tulip":'''Colorful spring blooming flower.The tulip is a beautiful flowering plant known for its vibrant colors and elegant shape. It is widely used in gardens, floral arrangements, home decorations, and as a gift to express love and appreciation. Tulips symbolize perfect love, happiness, and new beginnings, making them one of the most popular ornamental flowers in the world'''
}

def predict_class(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [180,180])
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image, verbose=0)[0]
    return tf.nn.softmax(prediction).numpy()

# ===== USER PROFILE HEADER =====
col_title, col_profile = st.columns([6, 1])

with col_title:
    st.markdown("# 🌸 Intelligent Flower Recognition System")
    st.caption("Deep Learning • TensorFlow • Streamlit Analytics Dashboard")

with col_profile:
    if os.path.exists("assets/profile.jpg"):
        st.image("assets/profile.jpg", width=80)
    else:
        st.markdown("👤")

    st.markdown(
        "<center><b>Piyush Dawar</b></span></center>",
        unsafe_allow_html=True
    )

c1,c2,c3,c4 = st.columns(4)
c1.metric("Classes", "5")
c2.metric("Model", "CNN")
c3.metric("Input Size", "180x180")
c4.metric("Status", "Active")

tab1, tab2, tab3 = st.tabs(["🫡 Prediction","📔 Analytics","😘 About"])

with tab1:

    st.markdown("## 🌺 Interactive Flower Gallery")
    st.write("Click any flower below or upload your own image.")

    sample_images = {
        "Daisy": "assets/daisy.jpg",
        "Dandelion": "assets/dandelion.jpg",
        "Rose": "assets/rose.jpg",
        "Sunflower": "assets/sunflower.jpg",
        "Tulip": "assets/tulip.jpg"
    }

    selected_sample = None
    cols = st.columns(5)

    for i,(name,path) in enumerate(sample_images.items()):
        with cols[i]:
            if os.path.exists(path):
                st.image(path, use_container_width=True)
                if st.button(name):
                    selected_sample = path

    uploaded = st.file_uploader(
        "Upload a flower image",
        type=["jpg","jpeg","png"]
    )

    image_source = None

    if uploaded:
        image_source = Image.open(uploaded)
    elif selected_sample:
        image_source = Image.open(selected_sample)

    if image_source is not None:

        col1,col2 = st.columns([1,1])

        with col1:
            st.image(image_source, caption="Selected Flower", use_container_width=True)

        probs = predict_class(np.asarray(image_source))

        pred_idx = int(np.argmax(probs))
        pred_name = class_names[pred_idx]
        confidence = float(np.max(probs)*100)

        with col2:
            st.success(f"Prediction: {pred_name.title()}")
            st.metric("Confidence", f"{confidence:.2f}%")
            st.info(flower_info[pred_name])

        df = pd.DataFrame({
            "Flower": class_names,
            "Probability": probs*100
        })

        st.subheader("📊 Prediction Distribution")
        fig = px.bar(
            df,
            x="Flower",
            y="Probability",
            text="Probability",
            color="Flower",
            color_discrete_map=FLOWER_COLORS
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        pie = px.pie(
            df,
            names="Flower",
            values="Probability",
            title="🌸 Class Probability Share",
            color="Flower",
            color_discrete_map=FLOWER_COLORS
        )
        pie.update_traces(textinfo="percent+label")
        st.plotly_chart(pie, use_container_width=True)

        st.subheader("🏆 Top 3 Predictions")
        top3 = np.argsort(probs)[::-1][:3]

        cols = st.columns(3)
        for col, idx in zip(cols, top3):
            with col:
                flower = class_names[idx]
                st.markdown(
                    f"""
                    <div style="background:{FLOWER_COLORS[flower]};padding:20px;border-radius:15px;text-align:center;color:white;">
                    <h3>{flower.title()}</h3>
                    <h1>{probs[idx]*100:.2f}%</h1>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

with tab2:
    metrics = pd.DataFrame({
        "Metric":["Accuracy","Precision","Recall","F1 Score"],
        "Value":[95.8,95.1,94.8,95.0]
    })

    st.dataframe(metrics, use_container_width=True)
    st.plotly_chart(
        px.bar(metrics,x="Metric",y="Value",
               title="Performance Metrics"),
        use_container_width=True
    )

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=95.8,
        title={"text":"Model Accuracy"}
    ))
    st.plotly_chart(gauge, use_container_width=True)

with tab3:
    st.markdown("""
### Features
- Interactive Flower Gallery
- Upload & Predict
- Top 3 Predictions
- Probability Charts
- Analytics Dashboard
- Deep Learning Classification
""")
