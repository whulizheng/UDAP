import random
import torch
from typing import Tuple, Any
import torch.nn.functional as F
from torchvision.transforms import v2

def next_step(
    model: Any,
    noise_pred: torch.FloatTensor,
    timestep: int,
    x: torch.FloatTensor
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Performs a single step in DDIM sampling.
    
    Args:
        model: The diffusion model (contains scheduler and alphas_cumprod)
        noise_pred: Predicted noise from the UNet
        timestep: Current diffusion timestep index
        x: Current latent representation
        
    Returns:
        Tuple containing:
        - x_next: Updated latent representation after the step
        - pred_x0: Predicted original image (x0)
    """
    # Calculate cumulative alpha products for current and next timesteps
    next_step = timestep
    timestep = min(timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps, 999)
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep] if timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_prod_t_next = model.scheduler.alphas_cumprod[next_step]
    beta_prod_t = 1 - alpha_prod_t
    pred_x0 = (x - beta_prod_t**0.5 * noise_pred) / alpha_prod_t**0.5
    pred_dir = (1 - alpha_prod_t_next)**0.5 * noise_pred
    x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
    return x_next, pred_x0

def drloss(model, latents, depth, images, prompt_embeds, total_steps):

    def create_noise_pred(latents, t_in):
        return model.unet(latents, t_in, encoder_hidden_states=prompt_embeds, return_dict=False)[0]

    # forward
    for t in range(depth):
        t_in = torch.tensor((t+1)*int(model.scheduler.config.num_train_timesteps/total_steps),
                            device=model.device)
        t_in = t_in.clamp(max=model.scheduler.config.num_train_timesteps-1)

        noise_pred = torch.utils.checkpoint.checkpoint(
            create_noise_pred,
            latents,
            t_in,
            use_reentrant=False
        )
        latents = next_step(model, noise_pred, t_in, latents)[0]

    # backward
    for t in range(depth):
        t_in = torch.tensor((depth-t-1)*int(model.scheduler.config.num_train_timesteps/total_steps),
                            device=model.device)
        t_in = t_in.clamp(max=model.scheduler.config.num_train_timesteps-1)

        noise_pred = torch.utils.checkpoint.checkpoint(
            create_noise_pred,
            latents,
            t_in,
            use_reentrant=False
        )
        latents = model.scheduler.step(noise_pred, t_in, latents, False)[0]

  
    with torch.enable_grad():
        images_out = model.vae.decode(
            1 / model.vae.config.scaling_factor * latents,
            return_dict=False
        )[0]

    return F.mse_loss(images, images_out, reduction='mean')

def purifying(
    model: Any,
    images: torch.FloatTensor,
    args: Any
) -> torch.FloatTensor:
    """
    Applies purification process to enhance image quality.
    
    Args:
        model: The diffusion model (with VAE, UNet, scheduler)
        images: Input images to purify
        args: Configuration arguments (threshold, epochs, max_depth, total_steps)
        
    Returns:
        Purified images
    """
    # Set threshold tensor
    th = torch.tensor(args.threshold, device=model.device)
    
    blurrer = v2.GaussianBlur(kernel_size=9, sigma=12)
    # Encode images to latent space
    latents = model.vae.encode(blurrer(images)).latent_dist.sample()
    latents = latents * model.vae.config.scaling_factor
    latents = latents.detach()
    latents.requires_grad_(True)
    
    # Prepare prompt embeddings
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = model.encode_prompt(
            "", model.device, latents.shape[0], True, ""
        )
        negative_prompt_embeds = negative_prompt_embeds.detach()
        prompt_embeds = prompt_embeds.detach()
    
    # Optimization loop
    optimizer = torch.optim.AdamW([latents], lr=5e-2)
    for epoch in range(args.epochs):
        
        # Set scheduler timesteps
        model.scheduler.set_timesteps(args.total_steps, device=model.device)
        
        # Reset optimizer
        optimizer.zero_grad()
        
        # Start with initial latents
        tmp = latents.clone()
        
        
        loss_rec = drloss(model,tmp,random.randint(1,args.max_depth),images,prompt_embeds,args.total_steps)
        if loss_rec < th:
            print(f"Early stopping at epoch {epoch}/{args.epochs}, reconstruction loss: {loss_rec.item():.6f}")
            break
        
        # Apply loss and optimize
        loss = loss_rec * 1e3
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch + 1}/{args.epochs}, reconstruction loss: {loss_rec.item():.6f}")
    
    # Return final result
    with torch.no_grad():
        images_out = model.vae.decode(1 / model.vae.config.scaling_factor * latents, return_dict=False)[0]
    return images_out