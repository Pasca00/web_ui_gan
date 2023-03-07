from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self, n_classes, embedding_dim, latent_dim) -> None:
        super(Generator, self).__init__()

        self.label_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=n_classes, embedding_dim=embedding_dim),
            nn.Linear(embedding_dim, 16)
        )

        self.latent = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(513, 64*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*8, momentum=0.1,eps=0.8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*8, 64*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*8, momentum=0.1, eps=0.8),
            nn.ReLU(True), 
            nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1,bias=False),
            nn.BatchNorm2d(64*4, momentum=0.1, eps=0.8),
            nn.ReLU(True), 
            nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1,bias=False),
            nn.BatchNorm2d(64*2, momentum=0.1, eps=0.8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*2, 64*1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*1, momentum=0.1, eps=0.8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*1, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, inputs):
        noise_vector, label = inputs
        label_output = self.label_embedding(label)
        label_output = label_output.view(-1, 1, 4, 4)
        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 512, 4, 4)
        concat = torch.cat((latent_output, label_output), dim=1)
        image = self.model(concat)

        return image