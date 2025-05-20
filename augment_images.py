import os
from PIL import Image
import torchvision.transforms as transforms

number_of_augments = 2

# Paths
input_folder = "./train-images/original-dataset-folder-path"  # Folder with original images
output_folder = "./train-images/augmented-dataset-folder-path"  # Folder to save augmented images
os.makedirs(output_folder, exist_ok=True)

# Define transformation (same as before)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    transforms.ToTensor()
])

# Loop through each image and create new augmented versions
for i, filename in enumerate(os.listdir(input_folder)):
    img_path = os.path.join(input_folder, filename)
    img = Image.open(img_path).convert("RGB")

    for j in range(number_of_augments):  # Generate 64 augmented versions per image
        augmented_img = transform(img)  # Apply transformation
        augmented_img = transforms.ToPILImage()(augmented_img)  # Convert back to PIL image
        augmented_img.save(os.path.join(output_folder, f"aug_{i}_{j}.png"))
        print(os.path.join(output_folder, f"aug_{i}_{j}.png"))

print(f"Generated {number_of_augments * len(os.listdir(input_folder))} augmented images!")