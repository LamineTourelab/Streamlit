#structural libraries
import streamlit as st
import io
from PIL import Image
from io import BytesIO
import requests

#model specific libraries
import tensorflow as tf 
import tensorflow_hub as hub
import numpy as np
import pandas as pd 



img_path = 'https://github.com/LamineGith/Streamlit/blob/main/logo.png?raw=true'
#https://github.com/SalvatoreRa/StreamStyle/blob/main/img/robot_painting.png?raw=true
capt = 'An android painting. Image created by the author with DALL-E'
img_logo = 'https://github.com/LamineGith/Streamlit/blob/main/robot_painting.png?raw=true'


def load_images():
    content_img = st.file_uploader("Choose the image to paint!")
    style_img = st.file_uploader("Choose the style!")
    if content_img:
            cont = content_img.getvalue()
            content_img = Image.open(io.BytesIO(cont))
            print('p')
    if style_img: 
            styl = style_img.getvalue()   
            style_img = Image.open(io.BytesIO(styl))
            print('p')
    
    return content_img, style_img


def process_input(img):
  
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = tf.image.convert_image_dtype(img, tf.float32)
  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  
  scale = 1024 / max(shape)
  new_shape = tf.cast(shape * scale, tf.int32)
  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  img /= 255.0
  return img

def process_output(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)

def load_model():
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    return model

def NST(model, content, style):
    t_content = process_input(content)
    t_style = process_input(style)
    out = model(tf.constant(t_content), tf.constant(t_style))[0]
    result = process_output(out)
    return result

def outputs(style, content, styled_img):
    col1, col2, col3 = st.columns([0.25, 0.25, 0.25])
    with col1:
        st.write('Content image')
        st.image(content)
    with col2:
        st.write('Style image')
        st.image(style)
    with col3:
        st.write('Stylized image')
        st.image(styled_img)

# Create the main app
def main():
    #Title and column
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #000000;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">StreamStyle</p>', unsafe_allow_html=True)
        
        
        
    with col2:
        response = requests.get(img_logo)
        logo = Image.open(BytesIO(response.content))               
        st.image(logo,  width=150)
    
    response = requests.get(img_path)
    img_screen = Image.open(BytesIO(response.content))
    st.image(img_screen, caption=capt, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.subheader('Transform the style of your image with AI')
    
    st.sidebar.image(logo,  width=150)
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made by [Lamine TOURE](https://www.linkedin.com/feed/)")
    st.sidebar.markdown("---")
    with st.sidebar.expander("About this App"):
     st.write("""
        This simple app is using **Neural Style Transfer** to apply the style of an image to another.
     """)
    with st.sidebar.expander("About Neural Style Transfer"):
        st.write("""
        Neural style transfer is a technique in computer vision that allows for the creation of new images or videos by combining the content of one image or video with the style of another image or video. It is based on the idea of using deep neural networks to separate the content and style representations of an image and then recombining them in a new image that combines the content of the first with the style of the second. This technique has been used to create a wide range of applications, including the generation of artistic images and the creation of realistic virtual environments. It has also been used to improve the performance of machine learning models by providing a way to transfer knowledge from one domain to another. Neural style transfer has the potential to revolutionize the field of computer vision and has already had a significant impact on the way we think about and interact with images and videos.
        (this is written by ChatGPT)
        """)

    content, style = load_images()
    if content and style:
        model = load_model()
        styled_img = NST(model, content, style)
        outputs(style, content, styled_img)
        buf = BytesIO()
        styled_img.save(buf, format="JPEG")
        byte_im =buf.getvalue()
        st.download_button(
            label="Download Image",
            data=byte_im,
            file_name="styled_img"+".jpg",
            mime="image/jpg"
            )


if __name__ == "__main__":
    main()
