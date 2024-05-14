import torch
import torch.optim as optim
from models import Generator, Discriminator
from data_loader import CustomImageDataset
from torch.utils.data import Dataset, DataLoader
# Assuming the following functions and models are already defined:
# generator - the generative model
# discriminator - the discriminative model (critic)
# total_loss - the function for calculating the generator's total loss (as defined previously)
# wasserstein_loss - the function for calculating the discriminator's loss (as defined previously)
# get_young_old_pairs - a function to retrieve pairs of young and old images along with their health states and ages

# declare Generator and Discriminator object
generator = Generator()
discriminator = Discriminator()

# Set up the models and optimizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = generator.to(device)
discriminator = discriminator.to(device)

print(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Hyperparameters and settings
lambda_gp = 10
age_max = 95
age_min = 55
epochs = 2
discriminator_update_ratio = 5
initial_discriminator_updates = 50

# instantiate data loader
dataset = CustomImageDataset("/home/aarongzy/class/data")
train_loader = DataLoader(dataset, batch_size = 2, num_workers = 4)

# Training loop
for epoch in range(epochs):
    for batch_idx, (xi, yo, ad, ao, ho) in enumerate(train_loader):
        for _ in range(initial_discriminator_updates if epoch < 20 else discriminator_update_ratio):
            # Discriminator update steps
            # xi, yo, ao, ho = get_young_old_pairs()
            # xi, yo, ad, ao, ho = CustomImageDataset()
            
            # xi = xi.to(device)
            # yo = yo.to(device)
            # ad = ad.to(device)
            # ao = ao.to(device)
            # ho = ho.to(device)
            
            optimizer_D.zero_grad()
            fake_images = generator(xi, ad, ho).detach()
            d_loss = wasserstein_loss(discriminator, yo, fake_images, ao, ho, lambda_gp, device)
            d_loss.backward()
            optimizer_D.step()

        # Generator update steps
        optimizer_G.zero_grad()
        fake_images = generator(xi, ad, ho)
        g_loss = total_loss(discriminator, generator, yo, xi, ao, ho, age_max, age_min, lambda_gp, device)
        g_loss.backward()
        optimizer_G.step()

# Save models, if needed
# torch.save(generator.state_dict(), 'path_to_save_generator.pth')
# torch.save(discriminator.state_dict(), 'path_to_save_discriminator.pth')
