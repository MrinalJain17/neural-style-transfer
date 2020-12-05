from argparse import ArgumentParser
from multiprocessing import cpu_count

import pytorch_lightning as pl
import torch.optim as optim
from pynache.data import load_style
from pynache.data.datasets import get_coco
from pynache.models import TransformationNetwork, VGGFeatures, losses
from pynache.training.callbacks import ExamplesLoggingCallback
from pynache.training.logger import ARTIFACTS_PATH
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

SEED = 42999


class FastStyleTransfer(pl.LightningModule):
    def __init__(
        self,
        style: str,
        batch_size: int = 4,
        alpha: int = 5.25e-07,
        tv_strength: float = 7e-9,
        **kwargs
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
        super().on_fit_start()

        # Pre-computing style image feature maps
        self.style_features, _ = self.vgg_features(self.style_image.to(self.device))

    def training_step(self, batch, batch_idx):
        _, content_features = self.vgg_features(batch)

        generated_images = self.forward(batch)
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
            default=7e-9,
            help="Strength of the total variation loss",
        )

        return parser


def main(args):
    seed_everything(SEED)
    dict_args = vars(args)

    # Model
    model = FastStyleTransfer(**dict_args)

    # Trainer
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


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
