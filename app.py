import streamlit as st
import tensorflow as tf
import numpy as np
import os
import tempfile
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import gdown  # ⬅️ added for Google Drive download

# App title
st.title("Twi Image Classifier")
st.write("🔁 App reloaded!")

# ✅ Load model from Google Drive if not present
@st.cache_resource
def load_model():
    model_path = "fine_tuned_model_3.0.keras"
    file_id = "1Zt6Fg4PeQx9WPIXXWzwQTP4FhczpZ3L9"
    gdown_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(gdown_url, model_path, quiet=False)

    return tf.keras.models.load_model(model_path)

model = load_model()

# Class names
class_names = [  
    'Accidents and disaster', 'Agriculture', 'Architecture', 'Arts and crafts', 'Automobile',
    'Construction', 'Culture', 'Disabilities', 'Economy', 'Education', 'Energy', 'Engineering',
    'Entertainment', 'Ethnicity people and race', 'Family and society', 'Fashion and clothing',
    'Fauna and flora', 'Food and drink', 'Funeral', 'Furniture', 'Geography', 'Governance',
    'Health and medicine', 'History', 'Home and housing', 'Hospitality', 'Immigration',
    'Justice and law enforcement', 'Languages and Communication', 'Leisure', 'Lifestyle',
    'Love and romance', 'Marine', 'Mining', 'Movie cinema and theatre', 'Music and dance',
    'Nature', 'News', 'Politics', 'Religion', 'Sanitation', 'Science', 'Security', 'Sports',
    'Technology', 'Trading and commerce', 'Transportation', 'Travel and tourism',
    'Weather and climate'
]

# Twi translations
akan_twi_translations = {
    'Accidents and disaster': 'Asiane ne Amanehunu',
    'Agriculture': 'Kuadwuma',
    'Architecture': 'Adan mu nhyehyɛe',
    'Arts and crafts': 'Adwinneɛ ne nsaanodwuma',
    'Automobile': 'Kaa/ Kwan so nnwuma deɛ',
    'Construction': 'Adesie',
    'Culture': 'Amammerɛ',
    'Disabilities': 'Dɛmdi ahorow',
    'Economy': 'Sikasɛm ne Ahonya ho nsɛm',
    'Education': 'Nwomasua/Adesua',
    'Energy': 'Ahoɔden',
    'Engineering': 'Mfiridwuma',
    'Entertainment': 'Anigyedeɛ',
    'Ethnicity people and race': 'Mmusuakuw mu nnipa ne abusuakuw',
    'Family and society': 'Abusua ne Ɔmanfoɔ',
    'Fashion and clothing': 'Ahosiesie ne Ntadeɛ',
    'Fauna and flora': 'Mmoa ne Nnua',
    'Food and drink': 'Aduane ne Nsa',
    'Funeral': 'Ayie',
    'Furniture': 'Efie adeɛ / Efie hyehyeɛ',
    'Geography': 'Asase ho nimdeɛ',
    'Governance': 'Nniso nhyehyɛe',
    'Health and medicine': 'Apɔmuden ne Nnuro',
    'History': 'Abakɔsɛm',
    'Home and housing': 'Efie ne Tenabea',
    'Hospitality': 'Ahɔhoyɛ',
    'Immigration': 'Atubrafo ho nsɛm',
    'Justice and law enforcement': 'Atɛntenenee ne Mmara banbɔ',
    'Languages and Communication': 'Kasa ne Nkitahodie',
    'Leisure': 'Ahomegyeɛ',
    'Lifestyle': 'Abrateɛ',
    'Love and romance': 'Ɔdɔ ne Ɔdɔ ho nsɛm',
    'Marine': 'Ɛpo mu nsɛm',
    'Mining': 'Awuto fagude',
    'Movie cinema and theatre': 'Sinima ne Agorɔhwɛbea',
    'Music and dance': 'Nnwom ne Asaw',
    'Nature': 'Abɔdeɛ',
    'News': 'Kaseɛbɔ',
    'Politics': 'Amammuisɛm',
    'Religion': 'Gyidi ne Nsom',
    'Sanitation': 'Ahoteɛ',
    'Science': 'Saense',
    'Security': 'Banbɔ',
    'Sports': 'Agodie',
    'Technology': 'Tɛknɔlɔgyi',
    'Trading and commerce': 'Dwadie ne Nsesaguoɔ',
    'Transportation': 'Akwantuo',
    'Travel and tourism': 'Akwantuɔ ne Ahɔhoɔ',
    'Weather and climate': 'Ewiem tebea ne Ewiem nhyehyɛeɛ'
}

# Process single image
def process_single_image_streamlit(uploaded_image, img_size=(224, 224), batch_size=1):
    with tempfile.TemporaryDirectory() as temp_dir:
        class_dir = os.path.join(temp_dir, 'class0')
        os.makedirs(class_dir, exist_ok=True)

        image_path = os.path.join(class_dir, 'uploaded.jpg')
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        dataset = tf.keras.utils.image_dataset_from_directory(
            temp_dir,
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False
        )

        for images, _ in dataset.take(1):
            return images

# Upload and predict
uploaded_file = st.file_uploader("Fa mfonini (Upload Image)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Mfoni a wo de baeɛ", use_column_width=True)

    img_tensor = process_single_image_streamlit(uploaded_file)

    predictions = model.predict(img_tensor)[0]
    top_indices = predictions.argsort()[::-1][:3]
    top_preds = [(class_names[i], float(predictions[i])) for i in top_indices]

    if top_preds[0][1] >= 0.5:
        st.markdown("### Top 3 Nkyerɛaseɛ")
        for rank, (label, confidence) in enumerate(top_preds, start=1):
            twi_label = akan_twi_translations.get(label, "❓")
            st.write(f"{rank}. {label} ({twi_label}) - Gyidie: {confidence*100:.2f}%")
    else:
        st.warning("Gyidie no nsɔ 50%, enti yɛrentumi nka")
