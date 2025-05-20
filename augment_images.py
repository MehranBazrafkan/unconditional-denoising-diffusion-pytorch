# augment_images.py (Refactored)

import os
from PIL import Image
from torchvision import transforms

# Configuration
NUM_AUGMENTS = 2
INPUT_FOLDER = "./train-images/original-dataset-folder-path"
OUTPUT_FOLDER = "./train-images/augmented-dataset-folder-path"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Define augmentation transformations
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    transforms.ToTensor()
])

# Process images
image_filenames = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
total_augmented = 0

for idx, filename in enumerate(image_filenames):
    img_path = os.path.join(INPUT_FOLDER, filename)
    img = Image.open(img_path).convert("RGB")

    for j in range(NUM_AUGMENTS):
        augmented = augmentation_transforms(img)
        augmented_img = transforms.ToPILImage()(augmented)

        save_path = os.path.join(OUTPUT_FOLDER, f"aug_{idx}_{j}.png")
        augmented_img.save(save_path)
        total_augmented += 1

        print(f"Saved: {save_path}")

print(f"Total augmented images: {total_augmented}")