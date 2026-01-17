import streamlit as st
from fastai.vision.all import *

# streamlit is a web application to easily do a rapid prototyping

# labeling mechanism / function
# labeling function
def extract_pet_breed(filename):
    parts = filename.split("_")
    return "_".join(parts[0:-1])

cat_vs_dog_model = load_learner("pet_breed_model_fastai_284.pkl")

st.markdown("<h1 style='color: yellow;'>Pet Breed Classifier</h1>", unsafe_allow_html=True)
#st.title("Cat or Dog")
st.text("Created by Gamas Chang")

uploaded_file = st.file_uploader("Choose as image...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    real_img = PILImage.create(uploaded_file)
    resized_img = real_img.resize((224, 224), Image.NEAREST)
    prediction = cat_vs_dog_model.predict(resized_img)
    print(prediction)
    index = int(prediction[1])
    confident_level = prediction[2][index] * 100

    if confident_level > 90:
        label = f"I am {confident_level:.2f} % sure that it is a {prediction[0]}"
    else:
        label = f"WARNING. It am {confident_level:.2f} % sure that it is a {prediction[0]}"
    st.text(label)
    st.image(uploaded_file)