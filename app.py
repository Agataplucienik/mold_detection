from PIL import Image
import cv2
import numpy as np

import tempfile
import streamlit as st
import pandas as pd

from prediction import predict
# In[2]:

import os
import requests
import numpy as np
from rembg import remove
from PIL import Image
from io import BytesIO
import time

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import cv2
from sklearn.cluster import KMeans
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76


path = r'C:\Users\agata\Downloads\apple.png'
uploaded_file = rf'{path}'

def remove_background_with_rembg(input_image):
    # Use PIL to open the image from BytesIO
    pil_image = Image.open(input_image)

    # Save the PIL image to a temporary file in PNG format
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        pil_image.save(temp_file, format="PNG")
        temp_file_path = temp_file.name

    # Use rembg to remove the background
    with open(temp_file_path, "rb") as file:
        output_data = remove(file.read())

    # Convert the output data to a NumPy array
    output_array = np.frombuffer(output_data, dtype=np.uint8)

    # Decode the NumPy array into an image using OpenCV
    output_image = cv2.imdecode(output_array, cv2.IMREAD_UNCHANGED)

    # Remove the temporary file
    os.unlink(temp_file_path)

    return output_image

def remove_black_color(output_image):


    # Create a binary mask for black pixels
    mask = (output_image[:, :, 0] == 0) & (output_image[:, :, 1] == 0) & (output_image[:, :, 2] == 0)

    # Set the alpha channel based on the mask
    output_image[mask, 3] = 0

    return output_image





def clustering(image, k_value):
    flat_image = image.reshape((-1, image.shape[-1]))

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k_value, random_state=42)
    kmeans.fit(flat_image)

    # Get cluster centers and labels
    cluster_centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_




    counts = np.bincount(labels)
    max_count_cluster = np.argmax(counts)
    top_clusters = np.argsort(counts)[1:]
    # Get unique colors in hexadecimal format
    hex_colors = ['#%02x%02x%02x' % (c[2], c[1], c[0]) for c in cluster_centers[top_clusters]]
    number_of_colors=len(hex_colors)
    filtered_counts = counts.copy()
    filtered_counts[max_count_cluster] = 0
    return number_of_colors, hex_colors, filtered_counts




def plot_pie_chart(hex_colors, filtered_counts):
    # Plot pie chart for dominant colors
    fig, ax = plt.subplots(figsize=(3, 3))

    # Use a distinct color for each slice
    colors = hex_colors[:len(filtered_counts)]
    labels = hex_colors[:len(filtered_counts)]

    ax.pie(x=filtered_counts, colors=colors, startangle=90)
    ax.axis('equal')

    return fig

def mold(non_background_pixels):
    flattened_image = non_background_pixels.reshape((-1, non_background_pixels.shape[-1]))

    set_pixels=[]
    for i in flattened_image:
        if len(set(i))<3:
            set_pixels.append(set(i))

    mold = []
    no_mold = []

    for pixel_set in set_pixels:
        if len(pixel_set) == 2:
            a, b = pixel_set
            if abs(a - b) < 15:
                mold.append(pixel_set)
            else:
                no_mold.append(pixel_set)


    moldy = len(mold) / len(no_mold)

    if moldy > 0.1:
        message = "ACHTUNG! MOLD!"
    else:
        message = "NO MOLD DETECTED"

    return message

def main():
    st.set_page_config(page_title="MOLD DETECTOR APP",
                   page_icon=":gun:",
                   layout="wide")
    col1, col2, col3 = st.columns(spec=[1,2,1])
    with col2:
        st.title("MOLD DETECTOR APP")

        st.markdown("# :rainbow[Hunting the Silent Killer ]"   ":gun:")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)

        # Remove background with rembg
        processed_image = remove_background_with_rembg(uploaded_file)

        # Remove black color
        processed_image_without_black = remove_black_color(processed_image)
        col1, col2 = st.columns(2)

        with col1:
            st.image(original_image, caption="Original Image", use_column_width=True)

        with col2:
            st.image(processed_image, caption="Image after Background Removal", use_column_width=True)



        # K-means clustering for dominant colors
        k_value = st.slider("Select the number of dominant colors:", min_value=1, max_value=10, value=6)
        with st.spinner('Wait for it...'):
            time.sleep(9)
        st.success('Done!')
        number_of_colors, hex_colors, filtered_counts = clustering(processed_image_without_black, k_value)
        title = 'Color Detection'
            
        styled_subheader = f'<div style="font-size: 46px; text-align: center; background: linear-gradient(45deg, violet, indigo, blue, green, yellow, orange, red); -webkit-background-clip: text; color: transparent;">{title}</div>'
        st.markdown(styled_subheader, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(spec=[1, 2, 1])


        with col2:
            
            st.pyplot(plot_pie_chart(hex_colors, filtered_counts))

        

        moldish=mold(processed_image_without_black)
        styled_text = f'<div style="font-size: 56px; text-align: center; border: 2px solid red; padding: 10px;">{moldish}</div>'
        st.markdown(styled_text, unsafe_allow_html=True)
        test_text="Let's find out what it is! "
        styled_subheaders = f'##<div style="font-size: 46px; text-align: center;">{test_text} :mag:</div>'
        st.markdown(styled_subheaders, unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(spec=[1, 1, 1, 1, 1])
        with col3:
            if st.button('Predicting type of fruit :crystal_ball:'):
                #result = predict(processed_image)
                result = predict(original_image)
                
                styled_subheaders = f'##<div style="font-size: 46px; background-color: #0066cc, text-align: center;">We are sure it is not {result.upper()}</div>'
                st.markdown(styled_subheaders, unsafe_allow_html=True)
                st.success('Hooray!', icon='🎉') 


if __name__ == "__main__":
    main()
