import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
import os


# Load the ESRGAN model
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model


# Preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transforms.ToTensor()(img).unsqueeze(0)
    return img


# Postprocess the image
def postprocess_image(tensor):
    img = tensor.squeeze(0).clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255.0).astype(np.uint8)
    return img


# Upscale the image
def upscale_image(model, image_path, output_path):
    img = preprocess_image(image_path)
    with torch.no_grad():
        sr_img = model(img)
    sr_img = postprocess_image(sr_img)
    cv2.imwrite(output_path, cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))
    print(f"Upscaled image saved to {output_path}")


# Load the model
model_path = 'path/to/your/models/RRDB_ESRGAN_x4.pth'  # Adjust this path
model = load_model(model_path)

# Path to the input image and output image
input_image_path = 'path/to/your/input_image.jpg'  # Adjust this path
output_image_path = 'path/to/your/output_image.jpg'  # Adjust this path

# Upscale the image
upscale_image(model, input_image_path, output_image_path)

import matplotlib.pyplot as plt


def show_images(original_path, upscaled_path):
    original = cv2.cvtColor(cv2.imread(original_path), cv2.COLOR_BGR2RGB)
    upscaled = cv2.cvtColor(cv2.imread(upscaled_path), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Upscaled Image')
    plt.imshow(up
