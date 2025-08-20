"""Device utility functions for cross-platform compatibility."""

import torch


def get_device():
    """
    Get the best available device for PyTorch operations.
    
    Returns:
        torch.device: The best available device (MPS on Mac, CUDA on GPU machines, CPU otherwise)
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(tensor_or_model, device=None):
    """
    Move tensor or model to the specified device.
    
    Args:
        tensor_or_model: Tensor or model to move
        device: Target device (if None, uses get_device())
        
    Returns:
        Tensor or model moved to the target device
    """
    if device is None:
        device = get_device()
    return tensor_or_model.to(device)


def is_cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mps_available():
    """Check if MPS is available."""
    return torch.backends.mps.is_available()


def synchronize():
    """Synchronize the current device."""
    if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def empty_cache():
    """Empty the cache for the current device."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_device_count():
    """Get the number of available devices."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        return 1  # MPS typically has one device
    else:
        return 1  # CPU


def set_device(device_id):
    """Set the current device."""
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    # MPS doesn't need explicit device setting


def get_device_name():
    """Get a string representation of the current device."""
    device = get_device()
    if device.type == 'mps':
        return 'mps'
    elif device.type == 'cuda':
        return 'cuda'
    else:
        return 'cpu'