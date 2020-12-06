from argparse import ArgumentParser
from multiprocessing import cpu_count

import pytorch_lightning as pl
import torch
import torch.optim as optim
from pynache.data import load_content, load_style
from pynache.data.datasets import get_coco
from pynache.data.transforms import denormalize
from pynache.models import TransformationNetwork, VGGFeatures, losses
from pynache.training.logger import ARTIFACTS_PATH
from pynache.utils import to_numpy
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from wandb import Image

SEED = 42999


class FastStyleTransfer(pl.LightningModule):
    def __init__(
        self,
        style: str,
        batch_size: int = 4,
        alpha: int = 5.25e-07,
        tv_strength: float = 7e-09,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters("style", "batch_size", "alpha", "tv_strength")
        self.style_image = (
            load_style(style, resize=[256, 256])
            .unsqueeze(dim=0)
            .repeat(self.hparams.batch_size, 1, 1, 1)
        )  # Shape: (B, C, H, W)

        # Loss functions and networks
        self.transformation_network = TransformationNetwork()
        self.vgg_features = VGGFeatures(config="recommended")
        self.compute_style_loss = losses.StyleLossChained(
            num_layers=len(self.vgg_features.style_layers),
            weights="recommended",
            activation_shift=True,
        )
        self.compute_content_loss = losses.ContentLoss()
        self.compute_total_variation = losses.TotalVariation()

    def forward(self, x):
        return self.transformation_network(x)

    def on_fit_start(self):
        """Pre-computing feature maps of style image"""
        super().on_fit_start()
        self.style_features, _ = self.vgg_features(self.style_image.to(self.device))

    def training_step(self, batch, batch_idx):
        _, content_features = self.vgg_features(batch)

        generated_images = self(batch)
        generated_style, generated_content = self.vgg_features(generated_images)

        style_loss = (
            self.compute_style_loss(generated_style, self.style_features)
            / self.hparams.batch_size
        )
        content_loss = (
            self.compute_content_loss(generated_content, content_features)
            / self.hparams.batch_size
        )
        total_variation = (
            0
            if self.hparams.tv_strength == 0
            else self.compute_total_variation(generated_images)
        ) / self.hparams.batch_size

        total_loss = (
            (self.hparams.alpha * content_loss)
            + style_loss
            + (self.hparams.tv_strength * total_variation)
        )

        self.log_dict(
            {
                "style_loss": style_loss,
                "content_loss": content_loss,
                "total_loss": total_loss,
            },
            on_step=True,
            on_epoch=False,
        )

        return total_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        return DataLoader(
            get_coco(),
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=True,
            drop_last=True,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--style", type=str, default="starry_night", help="Style image"
        )
        parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
        parser.add_argument(
            "--alpha", type=float, default=5.25e-07, help="The content-style ratio"
        )
        parser.add_argument(
            "--tv_strength",
            type=float,
            default=7e-09,
            help="Strength of the total variation loss",
        )

        return parser


class ExamplesLoggingCallback(Callback):
    """Callback to upload sample predictions to W&B."""

    def __init__(self, content="kinkaku_ji", seed=None) -> None:
        super().__init__()
        self.log_every_n_steps = 2000
        self.content_image = load_content(content, resize=[256, 256]).unsqueeze(0)

    def on_fit_start(self, trainer, pl_module):
        self.style_image = pl_module.style_image[:1, :, :, :]  # Shape: (1, C, H, W)
        self.content_image = self.content_image.to(pl_module.device)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if (batch_idx + 1) % self.log_every_n_steps == 0:
            pl_module.eval()
            with torch.no_grad():
                generated_image = pl_module(self.content_image)
                self._log_image(
                    pl_module,
                    images=[self.content_image, self.style_image, generated_image],
                    captions=["Content image", "Style image", "Generated Image"],
                )
            pl_module.train()

    def _prepare_image(self, image: torch.Tensor):
        assert image.ndim == 4, "Expected input of shape (1, C, H, W)"
        with torch.no_grad():
            image = to_numpy(denormalize(image[0]).detach().cpu())
        return image

    def _log_image(self, pl_module, images, captions=None, section="samples"):
        pl_module.logger.experiment.log(
            {
                section: [
                    Image(self._prepare_image(image), caption=caption)
                    for (image, caption) in zip(images, captions)
                ]
            },
            step=pl_module.trainer.global_step,
        )


def main(args):
    seed_everything(SEED)
    dict_args = vars(args)

    # Model
    model = FastStyleTransfer(**dict_args)

    # Trainer
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)
    trainer.save_checkpoint(f"{ARTIFACTS_PATH}/{args.style}.ckpt")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = FastStyleTransfer.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    args.logger = WandbLogger(
        name="Fast Style Transfer",
        save_dir=ARTIFACTS_PATH,
        project="fast-style-transfer",
    )
    args.callbacks = [ExamplesLoggingCallback(seed=SEED)]

    main(args)
