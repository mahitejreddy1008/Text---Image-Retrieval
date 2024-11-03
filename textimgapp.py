import streamlit as st
import torch
import clip
import torchvision.utils as vutils
from PIL import Image
import os
from lib.utils import load_model_weights, mkdir_p
from models.GALIP import NetG, CLIP_TXT_ENCODER

# Set device
device = 'cpu'  # Change to 'cuda:0' if using a GPU

# Load CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.eval()

# Initialize the generator network
text_encoder = CLIP_TXT_ENCODER(clip_model).to(device)
netG = NetG(64, 100, 512, 256, 3, False, clip_model).to(device)

# Load pre-trained weights
path = 'pre_coco.pth'
checkpoint = torch.load(path, map_location=device)
netG = load_model_weights(netG, checkpoint['model']['netG'], multi_gpus=False)

# Create a directory for samples
mkdir_p('./samples')

# Streamlit UI
st.title("Image Generator from Text")
user_input = st.text_input("Enter your caption:")

if st.button("Generate Image"):
    if user_input:
        # Generate noise and process text
        batch_size = 2
        noise = torch.randn((batch_size, 100)).to(device)
        
        tokenized_text = clip.tokenize([user_input]).to(device)
        with torch.no_grad():
            sent_emb, _ = text_encoder(tokenized_text)
            sent_emb = sent_emb.repeat(batch_size, 1)
            fake_imgs = netG(noise, sent_emb, eval=True).float()

            # Save and display the generated image
            name = f'{user_input.replace(" ", "-")}'
            vutils.save_image(fake_imgs.data, f'samples/{name}.png', nrow=8, value_range=(-1, 1), normalize=True)

            # Load and display the image
            img_path = f'samples/{name}.png'
            img = Image.open(img_path)
            st.image(img, caption=f'Generated Image for: {user_input}')
    else:
        st.warning("Please enter a caption to generate an image.")
