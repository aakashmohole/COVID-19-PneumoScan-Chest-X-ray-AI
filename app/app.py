import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.badges import badge
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

datapath = '\\app\\sample\\'

@st.cache_data 
def load_sample_data1():
    img1 = Image.open(datapath + '\\data1n.jpeg')
    return img1

@st.cache_data 
def load_sample_data2():
    img2 = Image.open(datapath + '\\data1p.jpeg')
    return img2

path = "D:\\DL Projects\\Covid XRAY\\app\model\\resnet_model.h5"  # Update this with the actual path to your model
model = load_model(path)

# Load and preprocess an image
def load_and_preprocess_image(img):
    img = img.resize((256, 256))  # Resize to the target size expected by your model
    if img.mode != 'RGB':
        img = img.convert('RGB')  # Ensure the image is in RGB mode
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension
    img_arr = img_arr / 255.0  # Normalize the image to the range [0, 1]
    return img_arr

# STREAMLIT CODE

top_image = Image.open('D:\\DL Projects\\Covid XRAY\\app\\trippyPattern.png')
st.sidebar.image(top_image)

st.sidebar.title("COVID-19 PneumoScan")

st.sidebar.markdown("> An inference API for pre-screening upper-respiratory infectious diseases based on Chest X-ray (CXR) images.")
st.sidebar.header("Choose a page to proceed!")
page = st.sidebar.selectbox("", ["Upload Your Image", "Sample Data"])

home_tab, working_tab, devloper_tab = st.tabs(["Home", "Working", "Devloper"])

with home_tab:
    st.title("COVID-19 PneumoScan: Chest X-ray AI")
        
    st.markdown("> Disclaimer : I do not claim this application as a highly accurate COVID Diagnosis Tool. This Application has not been professionally or academically Vetted. This is purely for Educational Purpose to demonstrate the Potential of AI's help in Medicine.")
    st.markdown("Developed by : [Aakash Mohole](https://github.com/aakashmohole)")

    st.markdown("**Note:** You should upload atmost one Chest Xray Image of either class (COVID19 Infected or Normal). Since this application is a Classification Task not a Segmentation.")

    if page == 'Sample Data':
        st.header("Sample Data for Detecting COVID-19")
        st.markdown("""
        Here you can choose Sample Data
        """)
        sample_option = st.selectbox('Please Select Sample Data',
                                     ('Sample Data I', 'Sample Data II'))

        st.write('You selected:', sample_option)
        sample_image = None
        if sample_option == 'Sample Data I':
            if st.checkbox('Show Sample Data I'):
                st.info("Loading Sample data I.......")
                sample_image = load_sample_data1()
                st.image(sample_image, caption='Sample Data I', use_column_width=True)
        else:
            if st.checkbox('Show Sample Data II'):
                st.info("Loading Sample data II..........")
                sample_image = load_sample_data2()
                st.image(sample_image, caption='Sample Data II', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        if sample_image is not None:
            sample_img_arr = load_and_preprocess_image(sample_image)
            sample_prediction = model.predict(sample_img_arr)

            if sample_prediction[0][0] > 0.5:
                st.success("The Patient has Positive X-Ray: COVID-19 Positive")
            else:
                st.warning("The Patient has Normal X-Ray: COVID-19 Negative")

    else:
        uploaded_image_file = st.file_uploader("Choose a chest X-ray image...", type="jpg")

        if uploaded_image_file is not None:
            uploaded_image = Image.open(uploaded_image_file)
            st.image(uploaded_image, caption='Uploaded Chest X-ray', use_column_width=True)
            st.write("")
            st.write("Classifying...")

            uploaded_img_array = load_and_preprocess_image(uploaded_image)
            uploaded_prediction = model.predict(uploaded_img_array)

            if uploaded_prediction[0][0] > 0.5:
                st.success("The Patient has Positive X-Ray: COVID-19 Positive")
            else:
                st.warning("The Patient has Normal X-Ray: COVID-19 Negative")

with working_tab :
    st.title('COVID-19 PneumoScan Working!')
    st.image('D:\\DL Projects\\Covid XRAY\\app\\ae-cnn-final.png')
    st.header('Encoder, Decoder, and Autoencoder')
    st.write("Overview of AE-CNN: Our proposed framework consists of three main blocks namely encoder, decoder, and classifier. The figure shows the autoencoder based convolutional neural network (AE-CNN) model for disease classification. Here, autoencoder reduces the spatial dimension of the imput image of size 1024 × 1024. The encoder produces a latent code tensor of size 256 × 256 and decoder reconstructs back the image. This latent code tensor is passed through a CNN classifier for classifying the chest x-rays. The final loss is the weighted sum of the resconstruction loss by decoder and classification loss by the CNN classifier.")
    st.write("Encoder:")
    st.markdown("""> Function: The encoder compresses the input data into a latent-space representation. In the context of an image, it extracts important features and reduces the dimensionality.
                    Architecture: Typically consists of convolutional layers followed by pooling layers. For example, in ResNet-50, the initial layers can be considered part of the encoder as they extract features from the input image.""")
    st.write("")
    st.write("Decoder:")
    st.markdown("""> Function: The decoder reconstructs the input data from the latent-space representation. This is used in tasks where output images are required (e.g., image generation or segmentation).
                    Architecture: Typically consists of upsampling layers (like transposed convolutions) that increase the dimensionality of the latent representation back to the original input size.""")
    st.write("")
    st.write("Autoencoder:")
    st.markdown("""> Function: An autoencoder is a type of neural network used to learn efficient codings of input data. It consists of two parts: the encoder and the decoder. The goal is to compress the input into a latent-space representation and then reconstruct the output as closely as possible to the original input.
                    Architecture: Combines both the encoder and decoder. The encoder reduces the input to a latent space, and the decoder reconstructs the input from this latent space.""")
    
    st.subheader("ResNet-50 Model for COVID-19 Detection Using Chest X-Ray")
    st.write("ResNet-50 (Residual Network):")
    st.markdown("ResNet-50 is a deep convolutional neural network that is 50 layers deep. It is well-known for its ability to handle the vanishing gradient problem, which is common in very deep networks. This is achieved through the introduction of residual blocks.")
    

    # Feature Extraction with ResNet-50 Encoder
    st.subheader("Feature Extraction with ResNet-50 Encoder")
    st.markdown("""
    The **ResNet-50** model's convolutional layers act as an encoder, extracting high-level features from chest X-ray images.
    """)

    # Classification Head
    st.subheader("Classification Head")
    st.markdown("""
    After feature extraction, a global average pooling layer and a fully connected layer are added to classify the image as **COVID-19 positive** or **negative**.
    """)

    # Training
    st.subheader("Training")
    st.markdown("""
    The model is trained on a labeled dataset of chest X-ray images, with labels indicating the presence or absence of COVID-19.
    """)

    # Evaluation
    st.subheader("Evaluation")
    st.markdown("""
    The model is evaluated on a separate test set to measure its accuracy, sensitivity, specificity, and other relevant metrics.
    """)

with devloper_tab:
    badge(type="github", name="aakashmohole")
        
    st.title("""
                Hello! I am Aakash Mohole!
                """)
    st.write("""
                I'm a dedicated B-Tech student with a profound interest in Data Science. I am committed to mastering the art of transforming raw data into actionable insights. 
            My academic pursuits have equipped me with a robust foundation in data analysis, machine learning, and data visualization.
            """)
    st.write("""
                With an unwavering belief that every data point holds valuable information, I am actively seeking opportunities to apply my expertise to real-world challenges. 
            Whether it involves developing predictive models, facilitating data-driven decision-making, or crafting persuasive content, I am poised to make a substantial impact.
            I invite professionals and peers to connect, collaborate, and explore opportunities where data science and storytelling converge to drive innovation and positive change. 
            Together, we can navigate the dynamic landscape of data science and collectively shape a future driven by data-driven insights.
            """)

