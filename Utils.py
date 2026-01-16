import os
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image

import torch
import torch.nn.functional as F

# Define supported image extensions as a constant
SUPPORTED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}


def load_image(
    image_path: str,
    image_size: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Load and preprocess a single image for diffusion-based purification.

    Args:
        image_path (str): Path to the input image file.
        image_size (int): Target image size (height = width).
        blurrer (v2.Transform): A transform (e.g., GaussianBlur) applied after resizing.
        device (torch.device, optional): Device to store the tensor. Not used here but kept for interface consistency.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (C, H, W) with values in [-1, 1].
    """
    # Load image: returns (C, H, W) uint8 tensor
    image = read_image(image_path)  # Shape: [C, H, W], dtype: uint8

    # Ensure only RGB channels (drop alpha if present)
    if image.shape[0] > 3:
        image = image[:3, :, :]
    elif image.shape[0] == 1:
        # Grayscale to RGB (optional; depends on use case)
        image = image.repeat(3, 1, 1)

    # Convert to float and normalize to [-1, 1]
    image = image.float() / 127.5 - 1.0  # Now in [-1, 1]

    # Add batch dimension, resize, apply blur, then remove batch dim
    image = image.unsqueeze(0)  # [1, C, H, W]
    image = F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return image.squeeze(0)  # [C, H, W]


class ImageData(Dataset):
    """Dataset class to load images from a directory."""

    def __init__(self, root_path: str, image_size: int):
        """
        Initialize the dataset.

        Args:
            root_path (str): Root directory containing images (searched recursively).
            image_size (int): Target size for all images.
        """
        self.root_path = root_path
        self.image_size = image_size

        # Collect all valid image paths
        self.image_paths: List[str] = []
        for dirpath, _, filenames in os.walk(self.root_path):
            for filename in filenames:
                ext = filename.split(".")[-1].lower()
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    self.image_paths.append(os.path.join(dirpath, filename))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        """
        Load an image by index.

        Returns:
            Tuple of (filename, preprocessed_image_tensor)
        """
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        image_tensor = load_image(img_path, self.image_size)
        return filename, image_tensor


def torch_to_pil(images: torch.Tensor) -> List[Image.Image]:
    """
    Convert a batch of PyTorch tensors to PIL Images.

    Args:
        images (torch.Tensor): Tensor of shape (B, C, H, W) or (C, H, W) with values in [-1, 1].

    Returns:
        List[PIL.Image.Image]: List of PIL images in RGB or grayscale.
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)  # Add batch dim if missing

    # Denormalize from [-1, 1] to [0, 1]
    images = (images / 2 + 0.5).clamp(0, 1)

    # Convert to numpy in [0, 255]
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()  # [B, H, W, C]
    images = (images * 255).round().astype("uint8")

    pil_images = []
    for img in images:
        if img.shape[-1] == 1:
            pil_images.append(Image.fromarray(img.squeeze(), mode="L"))
        else:
            pil_images.append(Image.fromarray(img))
    return pil_images


def save_images(images: torch.Tensor, image_names: List[str], args) -> None:
    """
    Save a batch of images to disk.

    Args:
        images (torch.Tensor): Batch of images in [-1, 1] format.
        image_names (List[str]): Corresponding filenames.
        args: Argument namespace containing `save_dir`.
    """
    pil_images = torch_to_pil(images)
    os.makedirs(args.save_dir, exist_ok=True)
    for img, name in zip(pil_images, image_names):
        img.save(os.path.join(args.save_dir, name))