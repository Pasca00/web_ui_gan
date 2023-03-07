import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms, datasets
from torchvision.utils import save_image

from Generator import Generator
from Discriminator import Discriminator

ROOT_DIR = 'images/'
BATCH_SIZE = 8
NUM_EPOCHS = 10
lr_gen = 0.002
lr_disc = 0.002
latent_dim = 100

if __name__ == '__main__':
    denormalize = transforms.Normalize((-1, -1, -1), (2, 2, 2))
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(root=ROOT_DIR, transform=image_transforms)
    dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    G = Generator(4, 512, latent_dim=latent_dim).to('cuda')
    D = Discriminator(4, 512).to('cuda')

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=lr_gen, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_disc, betas=(0.5, 0.999))

    if not os.path.exists('./images_out/'):
        os.makedirs('./images_out/')

    save_out = './images_out/'

    for curr_epoch in range(1, NUM_EPOCHS + 1):
        for index, (real_images, labels) in enumerate(dataloader):
            optimizer_D.zero_grad()
            real_images = real_images.to('cuda')
            labels = labels.to('cuda')
            labels = labels.unsqueeze(1).long()

            real_target = torch.ones(real_images.size(0), 1).to('cuda')
            fake_target = torch.zeros(real_images.size(0), 1).to('cuda')

            D_real_loss = criterion(
                D((real_images, labels)), real_target
            )

            noise_vector = torch.randn(real_images.size(0), latent_dim).to('cuda')

            generated_images = G((noise_vector, labels))
            output = D((generated_images.detach(), labels))

            D_fake_loss = criterion(output, fake_target)

            D_total_loss = (D_real_loss + D_fake_loss) / 2

            D_total_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            G_loss = criterion(
                D((generated_images, labels), real_target)
            )

            G_loss.backward()
            optimizer_G.step()

            if (index % 10 == 0 and index != 0):
                image_name = os.path.join(save_out, 'e_{}_b_{}.jpg'.format(curr_epoch, index))
                save_image(denormalize(generated_images.detach().cpu()), image_name)