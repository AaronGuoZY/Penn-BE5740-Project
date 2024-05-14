import torch
import torch.optim as optim
from models import Generator, Discriminator
from data_loader import CustomImageDataset
from torch.utils.data import DataLoader
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from losses import wasserstein_loss, total_loss

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Setup device and move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = generator.to(device)
discriminator = discriminator.to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Hyperparameters
lambda_gp = 10
age_max = 95
age_min = 55
epochs = 100
discriminator_update_ratio = 5
initial_discriminator_updates = 50
batch_size = 3

# DataLoader
dataset = CustomImageDataset("/home/aarongzy/class/src/data")
train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
model_path = "/home/aarongzy/class/src/trained_models"

# Training loop
for epoch in range(epochs):
    total_d_loss = 0
    total_g_loss = 0
    for batch_idx, (xi, yo, ad, ao, ho) in enumerate(train_loader):
        xi = xi.float().to(device)
        yo = yo.float().to(device)
        ad = ad.float().to(device)
        ao = ao.float().to(device)
        ho = ho.float().to(device).squeeze(-1)

        current_batch_size = xi.size(0)
        xi = xi.view(current_batch_size, 1, 208, 160)
        yo = yo.view(current_batch_size, 1, 208, 160)

        # Discriminator update steps
        for _ in range(initial_discriminator_updates if epoch < 20 else discriminator_update_ratio):
            optimizer_D.zero_grad()
            fake_images = generator(xi, ho, ad).detach()
            d_loss = wasserstein_loss(discriminator, yo, fake_images, ao, ho, lambda_gp, device)
            d_loss.backward()
            optimizer_D.step()
            total_d_loss += d_loss.item()

        # Generator update steps
        optimizer_G.zero_grad()
        fake_images = generator(xi, ho, ad)
        g_loss = total_loss(
            critic=discriminator,
            generator=generator,
            real_images=yo,
            target_ages=ao,
            age_difference=ad,
            health_states=ho,
            age_max=age_max,
            age_min=age_min,
            lambda_gp=lambda_gp,
            device=device
        )
        g_loss.backward()
        optimizer_G.step()
        total_g_loss += g_loss.item()

    # Print average losses for the current epoch
    print(f'Epoch {epoch + 1}/{epochs}, Discriminator Loss: {total_d_loss / len(train_loader)}, Generator Loss: {total_g_loss / len(train_loader)}')

    # Save models every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(generator.state_dict(), os.path.join(model_path, f'generator_epoch{epoch + 1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(model_path, f'discriminator_epoch{epoch + 1}.pth'))

# Save final models
torch.save(generator.state_dict(), os.path.join(model_path, 'generator_final.pth'))
torch.save(discriminator.state_dict(), os.path.join(model_path, 'discriminator_final.pth'))

print("Training completed and models saved.")

print("Training completed and models saved.")
