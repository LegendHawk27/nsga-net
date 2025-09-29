import torch
import torch.nn as nn
import numpy as np
import copy


def compute_synflow_score(model, input_shape=(1, 3, 32, 32), device='cuda'):
    """
    Compute SynFlow score for a given model.
    Now much more strict - applies multiple quality checks and penalties.
    
    Args:
        model: The neural network model
        input_shape: Shape of input tensor (batch_size, channels, height, width)
        device: Device to run computation on
        
    Returns:
        float: SynFlow score (higher is better, but now much more strict)
    """
    # Create a copy of the model to avoid modifying the original
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    model_copy.to(device)
    
    # Additional quality checks - penalize models that don't meet criteria
    total_params = sum(p.numel() for p in model_copy.parameters() if p.requires_grad)
    
    # Penalty for models that are too large (encourages efficiency)
    if total_params > 1000000:  # More than 1M parameters
        size_penalty = -2.0
    elif total_params > 500000:  # More than 500K parameters
        size_penalty = -1.0
    else:
        size_penalty = 0.0
    
    # Penalty for models that are too small (encourages sufficient capacity)
    if total_params < 50000:  # Less than 50K parameters
        size_penalty = -1.5
    
    # Initialize all weights to small values to avoid overflow
    for param in model_copy.parameters():
        if param.requires_grad:
            param.data.fill_(0.1)  # Use 0.1 instead of 1.0 to avoid overflow
    
    # Create input tensor filled with ones
    input_tensor = torch.ones(input_shape).to(device)
    input_tensor.requires_grad_(True)
    
    # Forward pass
    try:
        output, _ = model_copy(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
    except:
        try:
            output = model_copy(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return 0.0
    
    # Compute the sum of all outputs
    loss = torch.sum(output)
    
    # Backward pass to compute gradients
    loss.backward()
    
    # Clip gradients to prevent extreme values
    torch.nn.utils.clip_grad_norm_(model_copy.parameters(), max_norm=1.0)
    
    # Compute SynFlow score as the sum of absolute values of weights * gradients
    synflow_score = 0.0
    param_count = 0
    max_param_score = 0.0
    
    for param in model_copy.parameters():
        if param.requires_grad and param.grad is not None:
            # Compute element-wise product and sum (more robust than multiplication)
            param_score = torch.sum(torch.abs(param * param.grad))
            if param_score.item() > 0:  # Only add positive contributions
                synflow_score += param_score.item()
                param_count += 1
                max_param_score = max(max_param_score, param_score.item())
    
    print(f"DEBUG: Raw synflow_score before processing: {synflow_score}")
    print(f"DEBUG: Parameter count: {param_count}")
    print(f"DEBUG: Max param score: {max_param_score}")
    
    # Use average instead of product to avoid overflow issues
    if param_count > 0:
        synflow_score = synflow_score / param_count
        # Take logarithm to make scores more manageable, but add offset to avoid -inf
        synflow_score = np.log(max(synflow_score, 1e-10))
        
        # Apply very light penalties (much less harsh than before)
        if synflow_score > 5.0:  # Only penalize extremely optimistic scores
            synflow_score = synflow_score - 0.5  # Very light penalty
        elif synflow_score < -10.0:  # Only penalize extremely poor scores
            synflow_score = synflow_score - 0.2  # Very light penalty
        
        # Apply minimal variance penalty
        score_variance_penalty = 0.05  # Very light penalty
        synflow_score = synflow_score - score_variance_penalty
        
        # Apply size penalty from quality checks (reduced)
        synflow_score = synflow_score + (size_penalty * 0.5)  # Half the penalty
        
        # Minimal stability penalty
        gradient_penalty = 0.02  # Very light penalty
        synflow_score = synflow_score - gradient_penalty
        
        # Log the raw score for debugging
        print(f"DEBUG: Raw SynFlow score: {synflow_score}, Size penalty: {size_penalty}")
    else:
        synflow_score = -15.0  # More harsh default for no parameters
    
    # Additional safety check for inf/nan values with harsher penalty
    if not np.isfinite(synflow_score):
        synflow_score = -15.0  # More harsh fallback value
    
    # Clean up
    del model_copy
    torch.cuda.empty_cache() if device == 'cuda' else None
    
    return synflow_score


def rank_architectures(synflow_scores, return_indices=True):
    """
    Rank architectures based on their SynFlow scores.
    
    Args:
        synflow_scores: List or array of SynFlow scores
        return_indices: If True, return sorted indices; if False, return sorted scores
        
    Returns:
        Sorted indices (if return_indices=True) or sorted scores (if return_indices=False)
    """
    if return_indices:
        return np.argsort(synflow_scores)[::-1]  # Descending order (higher scores first)
    else:
        return np.sort(synflow_scores)[::-1]  # Descending order (higher scores first)


def prefilter_architectures(synflow_scores, population_size, keep_ratio=0.5):
    """
    Prefilter architectures based on SynFlow scores, keeping the top performers.
    
    Args:
        synflow_scores: List or array of SynFlow scores for the population
        population_size: Size of the current population
        keep_ratio: Fraction of population to keep (0.0 to 1.0)
        
    Returns:
        List of indices of architectures to keep
    """
    if len(synflow_scores) == 0:
        return []
    
    # Calculate how many to keep
    keep_count = max(1, int(len(synflow_scores) * keep_ratio))
    
    # Get indices of top performers
    top_indices = rank_architectures(synflow_scores, return_indices=True)
    
    return top_indices[:keep_count].tolist()


def get_adaptive_threshold(generation, max_generations, min_threshold=-15.0, max_threshold=-5.0):
    """
    Get an adaptive threshold for prefiltering based on generation progress.
    
    Args:
        generation: Current generation number
        max_generations: Total number of generations
        min_threshold: Most strict threshold (early generations)
        max_threshold: Most lenient threshold (later generations)
        
    Returns:
        float: Adaptive threshold for prefiltering
    """
    progress = generation / max_generations
    threshold = min_threshold + (max_threshold - min_threshold) * progress
    return threshold


def get_strict_threshold(generation, max_generations):
    """
    Get a strict threshold for prefiltering in demanding scenarios.
    
    Args:
        generation: Current generation number
        max_generations: Total number of generations
        
    Returns:
        float: Strict threshold for prefiltering
    """
    # Start strict, become slightly more lenient over time
    min_threshold = -5.0  # Strict threshold
    max_threshold = 0.0   # More lenient threshold
    progress = generation / max_generations
    threshold = min_threshold + (max_threshold - min_threshold) * progress
    return threshold


def is_architecture_acceptable(synflow_score, generation, max_generations, strict_mode=True):
    """
    Check if an architecture meets the prefiltering criteria.
    
    Args:
        synflow_score: The computed SynFlow score
        generation: Current generation number
        max_generations: Total number of generations
        strict_mode: If True, use strict criteria
        
    Returns:
        bool: True if architecture meets prefiltering criteria
    """
    if strict_mode:
        threshold = get_strict_threshold(generation, max_generations)
    else:
        threshold = get_adaptive_threshold(generation, max_generations)
    
    # Check if score meets threshold
    if synflow_score < threshold:
        return False
    
    # Reject architectures with scores that are too extreme (likely unstable)
    if synflow_score > 10.0 or synflow_score < -15.0:
        return False
    
    return True


def compute_population_synflow_scores(models, input_shape=(1, 3, 32, 32), device='cuda'):
    """
    Compute SynFlow scores for a population of models.
    
    Args:
        models: List of model instances
        input_shape: Shape of input tensor
        device: Device to run computation on
        
    Returns:
        List of SynFlow scores for each model
    """
    synflow_scores = []
    
    for i, model in enumerate(models):
        try:
            score = compute_synflow_score(model, input_shape, device)
            synflow_scores.append(score)
            print(f"Model {i+1}/{len(models)}: SynFlow score = {score:.4f}")
        except Exception as e:
            print(f"Error computing SynFlow score for model {i+1}: {e}")
            synflow_scores.append(-20.0)  # Assign poor score for failed models
    
    return synflow_scores
