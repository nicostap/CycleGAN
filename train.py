import torch
from dataset import BatikImageDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def train_fn(
    disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    use_amp = torch.cuda.is_available()

    for idx, (image, batik) in enumerate(loop):
        image = image.to(config.DEVICE)
        batik = batik.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast() if use_amp else torch.enable_grad():
            fake_batik = gen_H(image)
            D_H_real = disc_H(batik)
            D_H_fake = disc_H(fake_batik.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_image = gen_Z(batik)
            D_Z_real = disc_Z(image)
            D_Z_fake = disc_Z(fake_image.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it together
            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        if d_scaler:
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()
        else:
            D_loss.backward()
            opt_disc.step()

        # Train Generators H and Z
        with torch.cuda.amp.autocast() if use_amp else torch.enable_grad():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_batik)
            D_Z_fake = disc_Z(fake_image)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_image = gen_Z(fake_batik)
            cycle_batik = gen_H(fake_image)
            cycle_image_loss = l1(image, cycle_image)
            cycle_batik_loss = l1(batik, cycle_batik)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            # identity_image = gen_Z(image)
            # identity_batik = gen_H(batik)
            # identity_image_loss = l1(image, identity_image)
            # identity_batik_loss = l1(batik, identity_batik)

            # add all together
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_image_loss * config.LAMBDA_CYCLE
                + cycle_batik_loss * config.LAMBDA_CYCLE
                # + identity_batik_loss * config.LAMBDA_IDENTITY
                # + identity_image_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        if g_scaler:
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()
        else:
            G_loss.backward()
            opt_gen.step()

        if idx % 200 == 0:
            save_image(fake_batik * 0.5 + 0.5, f"saved_images/batik_{idx}.png")
            save_image(fake_image * 0.5 + 0.5, f"saved_images/image_{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H,
            gen_H,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z,
            gen_Z,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H,
            disc_H,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z,
            disc_Z,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = BatikImageDataset(
        root_batik=config.TRAIN_DIR + "/batik",
        root_image=config.TRAIN_DIR + "/images",
        transform=config.transforms,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    d_scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_H,
            disc_Z,
            gen_Z,
            gen_H,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)

if __name__ == "__main__":
    main()
