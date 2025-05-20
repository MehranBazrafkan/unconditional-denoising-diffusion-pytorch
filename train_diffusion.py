import os
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

# Configuration
IMAGE_SIZE = 64
BATCH_SIZE = 16
TIMESTEPS = 20
NUM_EPOCHS = 200
LEARNING_RATE = 1e-5
DATASET_PATH = './train-images'
MODEL_SAVE_DIR = './trained-model'
SAMPLES_SAVE_DIR = './generated-images'

# Create output directories if not exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(SAMPLES_SAVE_DIR, exist_ok=True)

# Transformations
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset
train_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and diffusion process
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
).to(device)
diffusion = GaussianDiffusion(
    model=model,
    image_size=IMAGE_SIZE,
    timesteps=TIMESTEPS
).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    for step, (images, _) in enumerate(train_loader):
        images = images.to(device)

        loss = diffusion(images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{step + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Save model and generate samples
    model_path = os.path.join(MODEL_SAVE_DIR, f'diffusion-model-{epoch + 1}.pth')
    torch.save(model.state_dict(), model_path)

    # Remove previous epoch model if exists
    prev_model_path = os.path.join(MODEL_SAVE_DIR, f'diffusion-model-{epoch}.pth')
    if os.path.exists(prev_model_path):
        os.remove(prev_model_path)

    model.eval()
    with torch.no_grad():
        samples = diffusion.sample(batch_size=BATCH_SIZE)
        sample_path = os.path.join(SAMPLES_SAVE_DIR, f'Epoch-{epoch + 1}.png')
        save_image(samples, sample_path, nrow=int(math.sqrt(BATCH_SIZE)), normalize=True)

print("Training completed.")