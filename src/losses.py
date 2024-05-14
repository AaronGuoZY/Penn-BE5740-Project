import torch
import torch.nn.functional as F

def gradient_penalty(D, real_samples, fake_samples, real_health_states, real_age_vectors, device):
    # Random weight term for interpolation between real and fake samples
    epsilon = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    epsilon = epsilon.expand_as(real_samples)
    
    # Interpolate between real and fake samples
    interpolates = (epsilon * real_samples + ((1 - epsilon) * fake_samples)).requires_grad_(True)
    
    # Calculate discriminator output for interpolates
    d_interpolates = D(interpolates, real_health_states, real_age_vectors)
    
    # Define a tensor for gradient calculation that matches the d_interpolates shape
    grad_outputs = torch.ones_like(d_interpolates, device=device)
    
    # Compute gradients
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=grad_outputs, create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]
    
    # Compute the L2 norm of the gradients
    gradients_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()

    return gradient_penalty
    
    # Compute the L2 norm of the gradient
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def wasserstein_loss(critic, real_samples, fake_samples,  real_age_vectors, real_health_states, lambda_gp=10, device="cuda"):
   # lgan = wasserstein_loss(critic, real_images, fake_images, target_ages, health_states, lambda_gp, device)
   # Calculate critic scores for real and fake samples
    real_scores = critic(real_samples, real_health_states, real_age_vectors)
    fake_scores = critic(fake_samples, real_health_states, real_age_vectors)
    critic_loss = -(torch.mean(real_scores) - torch.mean(fake_scores))

    # Gradient penalty
    gp = gradient_penalty(critic, real_samples, fake_samples, real_health_states, real_age_vectors, device) * lambda_gp

    return critic_loss + gp

# Assuming critic is your discriminator model which should return a score given an input image
# real_samples, fake_samples, real_ages, health_states are tensors containing the respective data
# lambda_gp is the weighting factor for the gradient penalty (in the text it is mentioned as 10)
# device is the device on which the computation will be performed (e.g., "cuda" or "cpu")

# Example usage:
# real_images, fake_images, real_ages, health_states = get_data()  # You'll need to implement this
# lambda_gp = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# critic_loss = wasserstein_loss(critic, real_images, fake_images, real_ages, health_states, lambda_gp, device)




def identity_preservation_loss(input_images, output_images, age_difference_encoded, age_max, age_min):
    # Compute L1 loss between input and output images
    l1_loss = F.l1_loss(output_images, input_images, reduction='none').mean([1, 2, 3])
    
    # Sum over the age difference vector to obtain a single value
    age_difference = age_difference_encoded.sum(1)  # Assuming age_difference_encoded is of shape [batch_size, encoding_size]
    
    # Compute the age difference factor for the exponential term
    age_range = age_max - age_min
    exponent_factor = -age_difference / age_range
    
    # Calculate the identity preservation loss
    id_loss = l1_loss * torch.exp(exponent_factor)
    
    return id_loss.mean()


# Example usage:
# Assuming input_images, output_images, original_age, target_age are torch Tensors
# age_max, age_min are the maximum and minimum ages in the dataset

# input_images = ...  # Tensor of input images
# output_images = ... # Tensor of output images generated by the network
# original_age = ...  # Tensor of original ages of the subjects
# target_age = ...    # Tensor of target ages for the generated images
# age_max = ...       # Maximum age in the dataset
# age_min = ...       # Minimum age in the dataset

# identity_loss = identity_preservation_loss(input_images, output_images, original_age, target_age, age_max, age_min)

def self_reconstruction_loss(input_images, reconstructed_images):
    """
    Calculate the self-reconstruction loss between the input images
    and the reconstructed images produced by the network.

    Parameters:
    input_images (torch.Tensor): The input images to the network.
    reconstructed_images (torch.Tensor): The images reconstructed by the network.

    Returns:
    torch.Tensor: The calculated self-reconstruction loss.
    """
    # Compute L1 loss (mean absolute error) between input and reconstructed images
    loss = F.l1_loss(input_images, reconstructed_images)

    return loss


# Total loss function
def total_loss(critic, generator, real_images, age_difference, target_ages, health_states, age_max, age_min, lambda_gp, device):
    lambda_1 = 1    # Weight for LGAN
    lambda_2 = 100  # Weight for LID
    lambda_3 = 10   # Weight for Lrec

    # Generate fake images
    fake_images = generator(real_images, health_states, target_ages).to(device)

    # Adversarial loss
    lgan = wasserstein_loss(critic, real_images, fake_images, target_ages, health_states, lambda_gp, device)

    # Identity-preservation loss
    lid = identity_preservation_loss(real_images, fake_images, age_difference, age_max, age_min)

    # Self-reconstruction loss
    lrec = self_reconstruction_loss(real_images, fake_images)

    # Combine losses
    total_gen_loss = lambda_1 * lgan + lambda_2 * lid + lambda_3 * lrec

    return total_gen_loss
