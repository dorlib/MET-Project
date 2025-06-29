import torch
import logging
import os

# Try to import the correct UNETR implementation
try:
    from self_attention_cv import UNETR
    print("Using self_attention_cv UNETR implementation")
except ImportError:
    print("self_attention_cv not found, falling back to MONAI UNETR")
    from monai.networks.nets import UNETR

logger = logging.getLogger(__name__)

def get_model(checkpoint_path: str, device: torch.device):
    """
    Initialize and load the UNETR model for MET segmentation
    Copy of the working inference approach from the standalone script
    
    Args:
        checkpoint_path: Path to the model weights file
        device: torch.device for model execution
        
    Returns:
        Loaded and initialized model in eval mode
    """
    logger.info(f"Loading model from {checkpoint_path} on {device}")
    
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        logger.error(f"Model file not found: {checkpoint_path}")
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
    
    # Check if file is empty
    if os.path.getsize(checkpoint_path) <= 1024:  # Likely our placeholder
        logger.warning(f"Model file may be a placeholder: {checkpoint_path}")
    
    # Import self_attention_cv UNETR - this must work for the trained model
    try:
        from self_attention_cv import UNETR
    except ImportError as e:
        logger.error(f"self_attention_cv not available: {e}")
        logger.error("Installing self-attention-cv...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "self-attention-cv"])
        from self_attention_cv import UNETR
        logger.info("self-attention-cv installed successfully")
    
    # Use exact parameters from the training script and working local inference
    model = UNETR(
        img_shape=(128, 128, 128),
        input_dim=1,
        output_dim=4,  # NUM_CLASSES = 4
        embed_dim=128,
        patch_size=16,
        num_heads=8,
        ext_layers=[3, 6, 9, 12, 15, 18],
        norm='instance',
        dropout=0.2,
        base_filters=16,
        dim_linear_block=1024
    ).to(device)
    logger.info("Using self_attention_cv UNETR implementation")

    try:
        state = torch.load(checkpoint_path, map_location=device)
        
        # Check if state dict is valid
        if not isinstance(state, dict) and hasattr(state, 'state_dict'):
            # If we have a full model object instead of just the state dict
            state = state.state_dict()
        elif not isinstance(state, dict):
            # If state is not a dictionary at all, raise error
            raise ValueError("Invalid model state format")
            
        model.load_state_dict(state)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")
        
    model.eval()
    return model
