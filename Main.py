import os
import argparse
from pytorch_lightning import seed_everything
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader
import torch

# Local imports
import Utils
import UDAP


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Purify adversarial images using diffusion inversion.")
    parser.add_argument('--seed', type=int, default=42, help='Global random seed.')
    parser.add_argument('--save_dir', type=str, default="Outputs/Purified/demo", help='Directory to save purified images.')
    parser.add_argument('--images_root', type=str, default="PoisonedImages/demo", help='Root directory of input (adversarial) images.')
    parser.add_argument('--image_size', type=int, default=512, help='Image size for processing.')
    parser.add_argument('--diffusion_path', type=str, default="stabilityai/stable-diffusion-2-1-base", help='HuggingFace model path.')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs for purification.')
    parser.add_argument('--total_steps', type=int, default=20, help='Total DDIM steps.')
    parser.add_argument('--max_depth', type=int, default=10, help='Max depth for DDIM inversion.')
    parser.add_argument('--threshold', type=float, default=4e-3, help='Reconstruction loss threshold for early stopping.')
    return parser.parse_args()


def main(args):
    """Main purification pipeline."""
    # Set global seed
    seed_everything(args.seed)

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Output directory: {args.save_dir}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load diffusion model
    print("Loading Stable Diffusion model...")
    model = StableDiffusionPipeline.from_pretrained(
        args.diffusion_path,
        safety_checker=None,  # Disable safety checker
        torch_dtype=torch.float32
    ).to(device)

    # Prepare dataset and dataloader
    dataset = Utils.ImageData(args.images_root, args.image_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Process each image
    for file_names, images in dataloader:
        print(f"Processing: {file_names[0]}")
        images = images.to(device)

        # Purify using UDAP method
        purified_images = UDAP.purifying(model, images, args)

        # Save results
        Utils.save_images(purified_images, file_names, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)