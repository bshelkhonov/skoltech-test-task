import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM

from dataset import EuroSAT


def train_maps(
    n_epoch: int = 100, device: str = "cuda:0"
) -> None:

    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)
    ddpm.to(device)

    dataset = EuroSAT("./data", transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=16)
    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x in pbar:
            optim.zero_grad()
            x = tf(x.to(device))
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(8, (3, 64, 64), device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            save_image(grid, f"./contents/ddpm_sample_maps{i}.png")

            # save model
            torch.save(ddpm.state_dict(), "./ddpm_maps.pth")


if __name__ == "__main__":
    train_maps()
