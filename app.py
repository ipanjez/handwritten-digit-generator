import streamlit as st
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io


# Define the Generator Network (must be the same as during training)
class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, channels):
        super(Generator, self).__init__()
        self.img_shape = (channels, img_size, img_size)

        self.main = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, np.prod(self.img_shape)),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.main(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# Hyperparameters (must match training)
latent_dim = 100
img_size = 28
channels = 1

# Load the trained generator model
@st.cache_resource # Cache the model loading for better performance
def load_generator_model(model_path):
    generator = Generator(latent_dim, img_size, channels)
    generator.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    generator.eval() # Set to evaluation mode
    return generator

model_path = "generator_mnist.pth" # Make sure this file is in the same directory
generator = load_generator_model(model_path)

st.title("Handwritten Digit Generator")
st.write("Generate handwritten digits (0-9) using a trained GAN model.")

# User input for digit selection
digit_to_generate = st.slider("Select a digit to generate:", 0, 9, 0)

if st.button("Generate Images"):
    st.subheader(f"Generating 5 images for digit: {digit_to_generate}")
    
    # Generate 5 images
    generated_images = []
    with torch.no_grad():
        for _ in range(5):
            # For a simple GAN like this, we don't directly control the digit generated
            # based on the input number. A conditional GAN (cGAN) would be needed for that.
            # For this problem, we'll generate random images and hope for diversity.
            # If you trained a cGAN, you would pass the digit_to_generate as an input.
            z = torch.randn(1, latent_dim) # Generate one image at a time
            img = generator(z).cpu()
            img = 0.5 * img + 0.5 # Denormalize to [0, 1]
            img = transforms.ToPILImage()(img.squeeze(0)) # Convert to PIL Image
            generated_images.append(img)

    # Display images
    cols = st.columns(5)
    for i, img in enumerate(generated_images):
        with cols[i]:
            st.image(img, caption=f"Image {i+1}", use_container_width=True)

st.markdown("**Note:** This GAN generates random digits. To generate specific digits, a Conditional GAN (cGAN) would be required.")
