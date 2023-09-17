from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from pathlib import Path
from helpers import num_to_groups
from torch.optim import Adam
import torch
from unet import Unet
from torchvision.utils import save_image
from sampler import Sampler
from loss import p_losses
import matplotlib.pyplot as plt

results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000


# define image transformations (e.g. using torchvision)
train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])

# load dataset from the hub
fashion_mnist_dataset = FashionMNIST('data', transform=train_tf, download=True) 
# dataset[0] = (img, label) # ((1, 28, 28), i)
image_size = 28
channels = 1
batch_size = 128

# create dataloader
dataloader = DataLoader(fashion_mnist_dataset, batch_size=batch_size, shuffle=True)

# batch = next(iter(dataloader))

device = "cuda" if torch.cuda.is_available() else "cpu"

# define model
model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)

# define optimizer
optimizer = Adam(model.parameters(), lr=1e-3)

sampler = Sampler(1000)
epochs = 6

for epoch in range(epochs):
    for step, (batch, _) in enumerate(dataloader):
      optimizer.zero_grad()

      batch_size = batch.shape[0]
      batch = batch.to(device)

      # Algorithm 1 line 3: sample t uniformally for every example in the batch
      t = torch.randint(0, sampler.timesteps, (batch_size,), device=device).long()

      loss = p_losses(model, sampler, batch, t, loss_type="huber")

      if step % 100 == 0:
        print("Loss:", loss.item())

      loss.backward()
      optimizer.step()

      # save generated images
      if step != 0 and step % save_and_sample_every == 0:
        milestone = step // save_and_sample_every
        batches = num_to_groups(4, batch_size)
        all_images_list = list(map(lambda n: sampler.sample(model, batch_size=n, channels=channels), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)


# inference
# sample 64 images
samples = sampler.sample(model, image_size=image_size, batch_size=64, channels=channels)

# show a random one
random_index = 5
plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")

import matplotlib.animation as animation

random_index = 53

fig = plt.figure()
ims = []
for i in range(sampler.timesteps):
    im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
    ims.append([im])

animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
animate.save('diffusion.gif')
plt.show()
