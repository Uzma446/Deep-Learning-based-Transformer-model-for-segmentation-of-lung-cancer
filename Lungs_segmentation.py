import os
import zipfile
import gdown
zip_path = "/content/drive/My Drive/Lung_data.zip"
extract_path = "/content/Lung_data"
import shutil, os
os.makedirs(extract_path, exist_ok=True)
shutil.unpack_archive(zip_path, extract_path)

# Verify extraction
!ls "/content/Lung_data"
!pip install nibabel
!pip install torch torchvision
import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
!pip install monai
import os
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm
from monai.transforms import Compose, RandFlipd, RandRotate90d, RandAdjustContrastd

#  Define Paths
image_tr_path = "/content/Lung_data/imagesTr"
mask_tr_path = "/content/Lung_data/labelsTr"

aug_image_dir = "/content/Lung_data/augmented_imagesTr"
aug_mask_dir = "/content/Lung_data/augmented_masksTr"

#  Ensure Augmented Directories Exist
os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_mask_dir, exist_ok=True)

#  Correct MONAI Transformations with Dictionary Format
augmentation_transforms = Compose([
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
    RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3),
    RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.3), prob=0.5)
])

#  Collect nii Files from Subfolders
image_paths = []
mask_paths = []

for folder in sorted(os.listdir(image_tr_path)):
    folder_path = os.path.join(image_tr_path, folder)
    if os.path.isdir(folder_path):
        nii_files = [f for f in os.listdir(folder_path) if f.endswith(".nii")]
        if nii_files:
            image_paths.append(os.path.join(folder_path, nii_files[0]))

for folder in sorted(os.listdir(mask_tr_path)):
    folder_path = os.path.join(mask_tr_path, folder)
    if os.path.isdir(folder_path):
        nii_files = [f for f in os.listdir(folder_path) if f.endswith(".nii")]
        if nii_files:
            mask_paths.append(os.path.join(folder_path, nii_files[0]))

#  Batch-wise Processing with Dictionary Input
for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc="Augmenting Data"):
    # Load nii files
    nii_image = nib.load(img_path)
    nii_mask = nib.load(mask_path)

    image_data = nii_image.get_fdata()
    mask_data = nii_mask.get_fdata()

    # Convert to MONAI dictionary format
    sample = {
        "image": torch.tensor(image_data, dtype=torch.float32).unsqueeze(0),
        "mask": torch.tensor(mask_data, dtype=torch.float32).unsqueeze(0)
    }

    # Apply Augmentations
    augmented = augmentation_transforms(sample)

    aug_image = augmented["image"].squeeze(0).numpy()
    aug_mask = augmented["mask"].squeeze(0).numpy()

    # Generate Augmented File Name
    file_name = os.path.basename(img_path)
    aug_file_name = f"aug_{file_name}"

    # Save Augmented nii Files
    nib.save(nib.Nifti1Image(aug_image, nii_image.affine), os.path.join(aug_image_dir, aug_file_name))
    nib.save(nib.Nifti1Image(aug_mask, nii_mask.affine), os.path.join(aug_mask_dir, aug_file_name))

print("Batch-wise Augmentation Completed & Saved!")
import os

# Define Augmented Paths
aug_image_dir = "/content/Lung_data/augmented_imagesTr"
aug_mask_dir = "/content/Lung_data/augmented_masksTr"

# Check if Directories Exist
if not os.path.exists(aug_image_dir):
    raise FileNotFoundError(f" Augmented images directory NOT found: {aug_image_dir}")
if not os.path.exists(aug_mask_dir):
    raise FileNotFoundError(f" Augmented masks directory NOT found: {aug_mask_dir}")

# List Augmented Files
aug_images = os.listdir(aug_image_dir)
aug_masks = os.listdir(aug_mask_dir)

print(f" Augmented Images Found: {len(aug_images)} files")
print(f" Augmented Masks Found: {len(aug_masks)} files")

# Print Some Sample Files
print(f" Sample Augmented Images: {aug_images[:5]}")
print(f" Sample Augmented Masks: {aug_masks[:5]}")
import random
from glob import glob

# Load All Augmented Image Paths
aug_image_paths = sorted(glob(os.path.join(aug_image_dir, "*.nii")))
aug_mask_paths = sorted(glob(os.path.join(aug_mask_dir, "*.nii")))

# Ensure Matching Number of Images & Masks
assert len(aug_image_paths) == len(aug_mask_paths), "Mismatch in images & masks count!"

# Shuffle Data Randomly
random.shuffle(aug_image_paths)

# Split into 80% Training & 20% Validation
split_ratio = 0.8
split_idx = int(len(aug_image_paths) * split_ratio)

train_image_paths = aug_image_paths[:split_idx]
val_image_paths = aug_image_paths[split_idx:]

train_mask_paths = aug_mask_paths[:split_idx]
val_mask_paths = aug_mask_paths[split_idx:]

print(f"Training Samples: {len(train_image_paths)}")
print(f"Validation Samples: {len(val_image_paths)}")
import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os

# Fixed Dataset Class with Depth Resizing
class LungDataset(Dataset):
    def __init__(self, image_paths, mask_paths, target_depth=322, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_depth = target_depth  # Standardized depth
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Convert to Float32 & Add Channel Dimension
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, 512, 512, depth)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # Resize Depth to (512, 512, 322) (Keep Width/Height Unchanged)
        image = F.interpolate(image.unsqueeze(0), size=(322, 512, 512), mode="trilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(322, 512, 512), mode="nearest").squeeze(0)

        return image, mask
# Define Paths for Augmented Data
aug_image_dir = "/content/Lung_data/augmented_imagesTr"
aug_mask_dir = "/content/Lung_data/augmented_masksTr"

# List All Images & Masks
all_images = sorted(os.listdir(aug_image_dir))
all_masks = sorted(os.listdir(aug_mask_dir))

# Ensure Same Number of Images & Masks
assert len(all_images) == len(all_masks), "Mismatch between images & masks count!"

# Generate Full Paths
image_paths = [os.path.join(aug_image_dir, img) for img in all_images]
mask_paths = [os.path.join(aug_mask_dir, img) for img in all_masks]

# Train-Validation Split (80% Train, 20% Validation)
split_idx = int(0.8 * len(image_paths))

train_image_paths = image_paths[:split_idx]
train_mask_paths = mask_paths[:split_idx]

val_image_paths = image_paths[split_idx:]
val_mask_paths = mask_paths[split_idx:]

# Create Train & Validation Datasets
train_dataset = LungDataset(train_image_paths, train_mask_paths)
val_dataset = LungDataset(val_image_paths, val_mask_paths)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

print(f"Training Samples: {len(train_loader)} Batches")
print(f"Validation Samples: {len(val_loader)} Batches")
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ✅ Fix CUDA Memory Fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ✅ Free VRAM Before Training
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# ✅ Define Optimized 3D UNet Model
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[8, 16, 32]):  # 🔹 Reduced Depth
        super(UNet3D, self).__init__()

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for feature in features:
            self.encoders.append(self.conv_block(in_channels, feature))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            in_channels = feature

        self.bottleneck = self.conv_block(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoders.append(self.conv_block(feature * 2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []

        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for i in range(len(self.decoders)):
            x = self.upconvs[i](x)
            skip = skip_connections[i]

            if x.shape != skip.shape:
                diffD = skip.shape[2] - x.shape[2]
                diffH = skip.shape[3] - x.shape[3]
                diffW = skip.shape[4] - x.shape[4]
                x = F.pad(x, [diffW // 2, diffW - diffW // 2,
                              diffH // 2, diffH - diffH // 2,
                              diffD // 2, diffD - diffD // 2])

            x = torch.cat((skip, x), dim=1)
            x = self.decoders[i](x)

        return torch.sigmoid(self.final_conv(x))

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet3D(in_channels=1, out_channels=1).to(device)

# Loss & Optimizer
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets, smooth=1):
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        return 1 - (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

criterion = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss()

# Try Different Learning Rates
learning_rate = 1e-5  # Change to 1e-3 or 1e-5 if needed

# Try AdamW Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

# Training Loop (With Validation)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        # Ensure Scaling (0-1)
        images = images / images.max()
        masks = masks / masks.max()

        with torch.amp.autocast("cuda"):
            preds = model(images)
            loss = criterion(preds, masks) + dice_loss(preds, masks)  # 🔹 Try dice_loss(preds, masks) alone

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            images = images / images.max()
            masks = masks / masks.max()

            preds = model(images)
            loss = criterion(preds, masks) + dice_loss(preds, masks)
            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

# Free GPU Memory After Training
torch.cuda.empty_cache()


