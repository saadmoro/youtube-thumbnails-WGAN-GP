import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.serialization import normalize_storage_type
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from Model import Discriminator, Generator, initialize_weights
from Utils import gradient_penalty


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 100
NUM_EPOCHS = 250
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
#WEIGHT_CLIP = 0.01
LAMBDA_GP = 10


transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range (CHANNELS_IMG)]),
    ],
)

#dataset = datasets.MNIST(root = "dataset/", train = True, transform = transforms, download=True)
dataset = datasets.ImageFolder(root = "one_class", transform = transforms)
loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=True)
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr = LEARNING_RATE, betas = (0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr = LEARNING_RATE, betas = (0.0, 0.9))
#criterion = nn.BCELoss()

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        #noise = torch.randn((BATCH_SIZE, NOISE_DIM, 1, 1)).to(device)
        #fake = gen(noise)

        #print('Batch no.: ' + str(batch_idx))

        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE, NOISE_DIM, 1, 1)).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP*gp
            ) #Original WGAN loss + LAMBDA_GP times gradient penalty
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            #for p in critic.parameters():
            #    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)


        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        ###Discriminator: max log(D(x)) + log(1- D(G(z)))
        #disc_real = disc(real).reshape(-1)
        #loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        #disc_fake = disc(fake.detach()).reshape(-1)
        #loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        #loss_disc = (loss_disc_real + loss_disc_fake) / 2
        #disc.zero_grad()
        #loss_disc.backward(retain_graph = True)
        #opt_disc.step()

        ##Train Generator
        #output = disc(fake).reshape(-1)
        #loss_gen = criterion(output, torch.ones_like(output))
        #gen.zero_grad()
        #loss_gen.backward()
        #opt_gen.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                    Loss C: {loss_critic:.4f}, Loss G: {loss_gen:.4f}"
            )
            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                img_name = 'fake_epoch' + str(epoch) + '_batch_' + str(batch_idx) + '.png'
                save_image(img_grid_fake, img_name)


                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                writer_real.add_image("Real", img_grid_real, global_step=step)

            step += 1

torch.save(gen.state_dict(), 'mytraining_gen.pth')
torch.save(critic.state_dict(), 'mytraining_critic.pth')