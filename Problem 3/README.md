# Handwritten Digit Generator

A Streamlit web application that uses a trained GAN (Generative Adversarial Network) model to generate handwritten digit images.

## Features

- Generate handwritten digit images using a trained GAN model
- Interactive web interface built with Streamlit
- Generate multiple images at once
- Easy-to-use slider interface

## Files

- `app.py` - Main Streamlit application
- `generator_mnist.pth` - Trained GAN generator model
- `requirements.txt` - Python dependencies

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your web browser. Use the slider to select a digit and click "Generate Images" to create handwritten digit images.

## Model Details

- Architecture: Simple GAN with fully connected layers
- Training data: MNIST dataset
- Latent dimension: 100
- Output: 28x28 grayscale images

## Note

This GAN generates random digits. For generating specific digits, a Conditional GAN (cGAN) would be required.
