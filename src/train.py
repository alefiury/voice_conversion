import os
import argparse
import itertools

import torch
import wandb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable

from models import *
from data_proc import DataProc


def plot_batch_train(modelname, direction, curr_epoch, SRC, cyclic_SRC, fake_TRGT, real_TRGT):
    SRC, cyclic_SRC, fake_TRGT, real_TRGT = to_numpy(SRC), to_numpy(cyclic_SRC), to_numpy(fake_TRGT), to_numpy(real_TRGT)
    i = 1
    for src, cyclic_src, fake_target, real_target in zip(SRC, cyclic_SRC, fake_TRGT, real_TRGT):
        fname = "out_train/%s/%s_%02d_%s.png"%(modelname, direction, curr_epoch, i)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plot_mel_transfer_train(fname, curr_epoch, src, cyclic_src, fake_target, real_target)
        wandb.log({direction: wandb.Image(fname)})
        i += 1


def to_numpy(batch):
    batch = batch.detach().cpu().numpy()
    batch = np.squeeze(batch)
    return batch


def plot_mel_transfer_train(save_path, curr_epoch, mel_in, mel_cyclic, mel_out, mel_target):
    """Visualises melspectrogram style transfer in training, with target specified"""
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))

    ax[0,0].imshow(mel_in, interpolation="None")
    ax[0,0].invert_yaxis()
    ax[0,0].set(title='Input')
    ax[0,0].set_ylabel('Mels')
    ax[0,0].axes.xaxis.set_ticks([])
    ax[0,0].axes.xaxis.set_ticks([])

    ax[1,0].imshow(mel_cyclic, interpolation="None")
    ax[1,0].invert_yaxis()
    ax[1,0].set(title='Cyclic Reconstruction')
    ax[1,0].set_xlabel('Frames')
    ax[1,0].set_ylabel('Mels')

    ax[0,1].imshow(mel_out, interpolation="None")
    ax[0,1].invert_yaxis()
    ax[0,1].set(title='Output')
    ax[0,1].axes.yaxis.set_ticks([])
    ax[0,1].axes.xaxis.set_ticks([])

    ax[1,1].imshow(mel_target, interpolation="None")
    ax[1,1].invert_yaxis()
    ax[1,1].set(title='Target')
    ax[1,1].set_xlabel('Frames')
    ax[1,1].axes.yaxis.set_ticks([])

    fig.suptitle('Epoch ' + str(curr_epoch))
    plt.savefig(save_path)
    plt.close()


wandb.login()

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--model_name", type=str, help="name of the model")
parser.add_argument("--dataset", type=str, help="path to dataset")
parser.add_argument("--n_spkrs", type=int, default=2, help="size of the batches")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=50, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--dim", type=int, default=32, help="number of filters in first encoder layer")
parser.add_argument("--logging", type=bool, default=True, help="Wandb Logging")
parser.add_argument("--plot_interval", type=int, default=-1, help="interval between saving mel spectogram samples")

opt = parser.parse_args()
print(opt)

run = wandb.init(
    project="voice_conversion",
    config=opt,
    mode=None if opt.logging else "disabled"
)

wandb.run.name = opt.model_name
wandb.run.save()

cuda = True if torch.cuda.is_available() else False

# Create sample and checkpoint directories
os.makedirs("saved_models/%s" % opt.model_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_pixel = torch.nn.L1Loss()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Dimensionality (channel-wise) of image embedding
shared_dim = opt.dim * 2 ** opt.n_downsample

# Initialize generator and discriminator
shared_E = ResidualBlock(features=shared_dim)
encoder = Encoder(dim=opt.dim, in_channels=opt.channels, n_downsample=opt.n_downsample)
shared_G = ResidualBlock(features=shared_dim)
G1 = Generator(dim=opt.dim, out_channels=opt.channels, n_upsample=opt.n_downsample, shared_block=shared_G)
G2 = Generator(dim=opt.dim, out_channels=opt.channels, n_upsample=opt.n_downsample, shared_block=shared_G)
D1 = Discriminator(input_shape)
D2 = Discriminator(input_shape)

if cuda:
    encoder = encoder.cuda()
    G1 = G1.cuda()
    G2 = G2.cuda()
    D1 = D1.cuda()
    D2 = D2.cuda()
    criterion_GAN.cuda()
    criterion_pixel.cuda()

if opt.epoch != 0:
    # Load pretrained models
    encoder.load_state_dict(torch.load("saved_models/%s/encoder_%d.pth" % (opt.model_name, opt.epoch)))
    G1.load_state_dict(torch.load("saved_models/%s/G1_%d.pth" % (opt.model_name, opt.epoch)))
    G2.load_state_dict(torch.load("saved_models/%s/G2_%d.pth" % (opt.model_name, opt.epoch)))
    D1.load_state_dict(torch.load("saved_models/%s/D1_%d.pth" % (opt.model_name, opt.epoch)))
    D2.load_state_dict(torch.load("saved_models/%s/D2_%d.pth" % (opt.model_name, opt.epoch)))
else:
    # Initialize weights
    encoder.apply(weights_init_normal)
    G1.apply(weights_init_normal)
    G2.apply(weights_init_normal)
    D1.apply(weights_init_normal)
    D2.apply(weights_init_normal)

# Loss weights
lambda_0 = 10   # GAN
lambda_1 = 0.1  # KL (encoded spect)
lambda_2 = 100  # ID pixel-wise
lambda_3 = 0.1  # KL (encoded translated spect)
lambda_4 = 100  # Cycle pixel-wise
lambda_5 = 10   # latent space L1

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), G1.parameters(), G2.parameters()),
    lr=opt.lr,
    betas=(opt.b1, opt.b2),
)
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Prepare dataloader
dataloader = torch.utils.data.DataLoader(
	DataProc(opt),
	batch_size=opt.batch_size,
	shuffle=True,
	num_workers=opt.n_cpu)

def compute_kl(mu):
    mu_2 = torch.pow(mu, 2)
    loss = torch.mean(mu_2)
    return loss


# ----------
#  Training
# ----------

wandb.watch(G1)
wandb.watch(G2)

wandb.watch(D1)
wandb.watch(D2)

wandb.watch(encoder)

for epoch in range(opt.epoch, opt.n_epochs):
    losses = {'G': [],'D': []}
    progress = tqdm(enumerate(dataloader),desc='',total=len(dataloader))
    for i, batch in progress:

        # Set model input
        X1 = Variable(batch["A"].type(Tensor))
        X2 = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((X1.size(0), *D1.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((X1.size(0), *D1.output_shape))), requires_grad=False)

        # -------------------------------
        #  Train Encoders and Generators
        # -------------------------------

        optimizer_G.zero_grad()

        # Get shared latent representation
        mu1, Z1 = encoder(X1)
        mu2, Z2 = encoder(X2)

        # Latent space feat
        feat_1 = mu1.view(mu1.size()[0], -1).mean(dim=0)
        feat_2 = mu2.view(mu2.size()[0], -1).mean(dim=0)

        # Reconstruct speech
        recon_X1 = G1(Z1)
        recon_X2 = G2(Z2)

        # Translate speech
        fake_X1 = G1(Z2)
        fake_X2 = G2(Z1)

        # Cycle translation
        mu1_, Z1_ = encoder(fake_X1)
        mu2_, Z2_ = encoder(fake_X2)
        cycle_X1 = G1(Z2_)
        cycle_X2 = G2(Z1_)

        # Losses
        loss_GAN_1 = lambda_0 * criterion_GAN(D1(fake_X1), valid)
        loss_GAN_2 = lambda_0 * criterion_GAN(D2(fake_X2), valid)
        loss_KL_1 = lambda_1 * compute_kl(mu1)
        loss_KL_2 = lambda_1 * compute_kl(mu2)
        loss_ID_1 = lambda_2 * criterion_pixel(recon_X1, X1)
        loss_ID_2 = lambda_2 * criterion_pixel(recon_X2, X2)
        loss_KL_1_ = lambda_3 * compute_kl(mu1_)
        loss_KL_2_ = lambda_3 * compute_kl(mu2_)
        loss_cyc_1 = lambda_4 * criterion_pixel(cycle_X1, X1)
        loss_cyc_2 = lambda_4 * criterion_pixel(cycle_X2, X2)
        loss_feat = lambda_5 * criterion_pixel(feat_1, feat_2)

        # Total loss
        loss_G = (
            loss_KL_1
            + loss_KL_2
            + loss_ID_1
            + loss_ID_2
            + loss_GAN_1
            + loss_GAN_2
            + loss_KL_1_
            + loss_KL_2_
            + loss_cyc_1
            + loss_cyc_2
            + loss_feat
        )

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator 1
        # -----------------------

        optimizer_D1.zero_grad()

        loss_D1 = criterion_GAN(D1(X1), valid) + criterion_GAN(D1(fake_X1.detach()), fake)

        loss_D1.backward()
        optimizer_D1.step()

        # -----------------------
        #  Train Discriminator 2
        # -----------------------

        optimizer_D2.zero_grad()

        loss_D2 = criterion_GAN(D2(X2), valid) + criterion_GAN(D2(fake_X2.detach()), fake)

        loss_D2.backward()
        optimizer_D2.step()

        # --------------
        #  Log Progress
        # --------------

        losses['G'].append(loss_G.item())
        losses['D'].append((loss_D1 + loss_D2).item())

        wandb.log({"G_loss": loss_G.item()})
        wandb.log({"D_loss": (loss_D1 + loss_D2).item()})

        # update progress bar
        progress.set_description("[Epoch %d/%d] [D loss: %f] [G loss: %f] "
            % (epoch,opt.n_epochs,np.mean(losses['D']), np.mean(losses['G'])))

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D1.step()
    lr_scheduler_D2.step()

    wandb.log({"epoch": epoch})

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(encoder.state_dict(), "saved_models/%s/encoder_%d.pth" % (opt.model_name, epoch))
        torch.save(G1.state_dict(), "saved_models/%s/G1_%d.pth" % (opt.model_name, epoch))
        torch.save(G2.state_dict(), "saved_models/%s/G2_%d.pth" % (opt.model_name, epoch))
        torch.save(D1.state_dict(), "saved_models/%s/D1_%d.pth" % (opt.model_name, epoch))
        torch.save(D2.state_dict(), "saved_models/%s/D2_%d.pth" % (opt.model_name, epoch))

    # Plot first batch every epoch or few epochs
    if opt.plot_interval != -1 and (epoch+1) % opt.plot_interval == 0:
        plot_batch_train(opt.model_name, "A-B", epoch, X1, cycle_X1, fake_X2, X2)
        plot_batch_train(opt.model_name, "B-A", epoch, X2, cycle_X2, fake_X1, X1)
