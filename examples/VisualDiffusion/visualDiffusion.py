"""
A demo that used to test the diffusion model.

The main purpose is to view how diffusion model works.

Most of the code derives from one github repo with address
"https://github.com/tcapelle/Diffusion-Models-pytorch".

"""

import os
import logging

import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from PIL import Image
from matplotlib import pyplot as plt

from torch.utils.tensorboard import SummaryWriter


from vggbase.models.diffusions import generalized_diffusion

from vggbase.config import Config


def one_batch(dl):
    return next(iter(dl))


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat(
            [
                torch.cat([i for i in images.cpu()], dim=-1),
            ],
            dim=-2,
        )
        .permute(1, 2, 0)
        .cpu()
    )
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(
    dataset_path, train_folder, val_folder, batch_size, image_size, slice_size=-1
):
    train_transforms = torchvision.transforms.Compose(
        [
            T.Resize(
                image_size + int(0.25 * image_size)
            ),  # image_size + 1/4 *image_size
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    val_transforms = torchvision.transforms.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(dataset_path, train_folder), transform=train_transforms
    )
    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(dataset_path, val_folder), transform=val_transforms
    )

    if slice_size > 1:
        train_dataset = torch.utils.data.Subset(
            train_dataset, indices=range(0, len(train_dataset), slice_size)
        )
        val_dataset = torch.utils.data.Subset(
            val_dataset, indices=range(0, len(val_dataset), slice_size)
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_dataset = DataLoader(
        val_dataset,
        batch_size=2 * batch_size,
        shuffle=False,
        num_workers=1,
    )
    return train_dataloader, val_dataset


def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def main():
    """Main function to train a ViBertTransformer model for the VG task."""

    ## obtain configurations
    env_config = Config().environment
    model_config = Config().model
    data_config = Config().data
    train_config = Config().train
    loss_config = Config().loss
    logging_config = Config().logging

    env_config = Config.items_to_dict(env_config._asdict())
    env_device = env_config["device"]

    ## define the diffusion model
    diffusion_model = generalized_diffusion.GeneralizedDiffusionModel(
        chain_steps=model_config.chain_steps,
        noise_variance_schedule_config=model_config.noise,
        diffusion_head_config=model_config.diffusion_head_config,
        out_weights_config=model_config.out_weight_config,
        normalization_config=model_config.normalization_config,
        reverse_sampling_config=model_config.reverse_sampling_config,
        device=env_device,
    )

    ## get the dataloader
    train_dataloader, val_dataloader = get_data(
        image_size=data_config.image_size,
        dataset_path=data_config.data_path,
        train_folder="train",
        val_folder="test",
        batch_size=train_config.batch_size,
    )

    optimizer = optim.AdamW(
        diffusion_model.parameters(), lr=train_config.parameters.optimizer.lr
    )

    mse_loss = nn.MSELoss()
    data_length = len(train_dataloader)
    logger = SummaryWriter(os.path.join("runs", "DiffusionDemo"))
    images = None
    for epoch in range(train_config.epochs):
        logging.info("Starting epoch %s:", epoch)
        pbar = tqdm(train_dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(env_device)
            diffusion_outputs = diffusion_model(images)

            predicted_noise = diffusion_outputs.predictions
            targets = diffusion_outputs.diffusion_targets

            loss = mse_loss(predicted_noise, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * data_length + i)

        sampled_images = diffusion_model.diffusion_reverse_sampling(
            target_shape=images.shape
        )
        save_images(
            sampled_images, os.path.join("results", "DiffusionDemo", f"{epoch}.jpg")
        )
        torch.save(
            diffusion_model.state_dict(),
            os.path.join("models", "DiffusionDemo", "ckpt.pt"),
        )


if __name__ == "__main__":
    main()

    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
