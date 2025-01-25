import torch

def generate_bandmasks(
    positions: torch.Tensor,
    num_context_patches: int,
    num_target_patches: int,
    sigma1: float = 1.0,
    sigma2: float = 3.0,
    patch_size: int = 16,
    max_dim: int = 512
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate two independent masks for context and target patches based on separate
    band-pass filtering distributions, ensuring no overlap between them.
    Each distribution is generated using mid-frequency noise: Ng = Gσ1 * W - Gσ2 * W.
    Uses top-k selection.
    
    Args:
        positions: [B, N] tensor of patch positions
        num_context_patches: Number of patches to select for context
        num_target_patches: Number of patches to select for targets
        sigma1: Standard deviation for weak blur (default: 1.0)
        sigma2: Standard deviation for strong blur (default: 3.0)
        patch_size: Size of each patch
        max_dim: Maximum dimension of reference grid
        
    Returns:
        context_mask: [B, num_context_patches] tensor of indices
        target_mask: [B, num_target_patches] tensor of indices
    """
    assert sigma1 < sigma2, "sigma1 must be less than sigma2 for proper band-pass filtering"
    
    B, N = positions.shape
    patches_per_dim = max_dim // patch_size
    device = positions.device
    
    # Convert positions to 2D coordinates for each batch
    row_coords = positions // patches_per_dim
    col_coords = positions % patches_per_dim
    coords = torch.stack([row_coords, col_coords], dim=-1).float()  # [B, N, 2]
    
    context_masks = []
    target_masks = []
    
    for b in range(B):
        # Generate two independent white noise distributions
        W_context = torch.randn(N, device=device)
        W_target = torch.randn(N, device=device)
        
        # Compute pairwise distances for this batch's coordinates
        dists = torch.cdist(coords[b], coords[b])  # [N, N]
        
        # Generate kernels for both blurs
        kernel1 = torch.exp(-dists**2 / (2 * sigma1**2))
        kernel1 = kernel1 / kernel1.sum(dim=1, keepdim=True)
        
        kernel2 = torch.exp(-dists**2 / (2 * sigma2**2))
        kernel2 = kernel2 / kernel2.sum(dim=1, keepdim=True)
        
        # Generate band-pass noise for context
        weak_blur_context = torch.matmul(kernel1, W_context)
        strong_blur_context = torch.matmul(kernel2, W_context)
        band_pass_context = weak_blur_context - strong_blur_context
        
        # Get top-k indices for context
        _, context_indices = torch.topk(band_pass_context, num_context_patches)
        
        # Generate band-pass noise for targets
        weak_blur_target = torch.matmul(kernel1, W_target)
        strong_blur_target = torch.matmul(kernel2, W_target)
        band_pass_target = weak_blur_target - strong_blur_target
        
        # Zero out values at context positions
        band_pass_target[context_indices] = float('-inf')
        
        # Get top-k indices for targets
        _, target_indices = torch.topk(band_pass_target, num_target_patches)
        
        context_masks.append(context_indices)
        target_masks.append(target_indices)
    
    return torch.stack(context_masks, dim=0), torch.stack(target_masks, dim=0)
