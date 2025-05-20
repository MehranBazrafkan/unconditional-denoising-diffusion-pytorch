# Load dataset
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

image_size = 64
model_time_steps = 20
batch_size = 16
dataset_path = './train-images'
learning_rate = 0.00001

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define model
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
unet = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8, 4, 2, 1)
).to(device=device)
diffusion = GaussianDiffusion(model=unet, image_size=image_size, timesteps=model_time_steps).to(device=device)

# Train model
import os
import math
import torch.optim as optim
from torchvision.utils import save_image

optimizer = optim.Adam(unet.parameters(), lr=learning_rate)
num_epochs = 200
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataLoader):
        images = images.to(device=device)
        diffusion.train()
        loss = diffusion(images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataLoader)}], Loss: {loss.item():.4f}')

    if epoch % 1 == 0:
        torch.save(unet.state_dict(), f'./trained-model/diffusion-model-{epoch+1}.pth')
        if os.path.exists(f'./trained-model/diffusion-model-{epoch}.pth'):
            os.remove(f'./trained-model/diffusion-model-{epoch}.pth')
        diffusion.eval()
        generated_images = diffusion.sample(batch_size=batch_size)
        image_filename = f'./generated-images/Epoch-{epoch+1}.png'
        save_image(generated_images.data, image_filename, nrow=int(math.sqrt(batch_size)), normalize=True)