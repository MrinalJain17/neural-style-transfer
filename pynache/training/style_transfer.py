from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import wandb
from pynache.data import load_content, load_style
from pynache.data.transforms import denormalize
from pynache.models import VGGFeatures, losses
from pynache.models.vgg import FEATURES_CONFIG
from pynache.paths import REPOSITORY_ROOT
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACTS_PATH = (Path(REPOSITORY_ROOT) / "wandb_artifacts").as_posix()
Path(ARTIFACTS_PATH).mkdir(exist_ok=True)

wandb.init(
    name="Original Style Transfer", project="neural-style-transfer", dir=ARTIFACTS_PATH
)


class OriginalStyleTransfer(object):
    def __init__(
        self,
        content,
        style,
        resize=[512, 512],
        normalize=True,
        vgg_config="default",
        use_avg_pool=True,
        use_normalized_vgg=True,
        init_image="noise",
        style_weighing="default",
        alpha=1e-3,
        steps=500,
    ):
        self.device = DEVICE
        self.alpha = alpha
        self.steps = steps

        self.content_image, self.style_image, self.generated_image = self._load_images(
            content, style, init_image, resize, normalize
        )
        self.vgg_features = self._load_model(
            vgg_config, use_avg_pool, use_normalized_vgg
        )

        self.compute_style_loss = losses.StyleLoss(
            num_layers=len(self.vgg_features.style_layers), weights=style_weighing
        )
        self.compute_content_loss = losses.ContentLoss()

    def train(self):
        self._log_image(
            [self.content_image, self.style_image],
            "inputs",
            captions=["Content image", "Style image"],
            step=0,
        )

        _, content_features = self.vgg_features(self.content_image)
        style_features, _ = self.vgg_features(self.style_image)

        optimizer = optim.LBFGS([self.generated_image])

        for step in tqdm(range(self.steps), unit="step"):

            def closure():
                optimizer.zero_grad()
                loss = self._train_step(
                    style_features=style_features,
                    content_features=content_features,
                    step=step,
                )
                loss.backward()
                return loss

            optimizer.step(closure)

            if (step % 20 == 0) or (step + 1 == self.steps):
                self._log_image(
                    [self.generated_image],
                    "outputs",
                    captions=[f"Iteration {step}"],
                    step=step,
                )

    def _load_images(self, content, style, init_image, resize, normalize):
        assert init_image in ["noise", "content"], "Invalid initial image passed"

        content_image = (
            load_content(content, resize, normalize).unsqueeze(dim=0).to(self.device)
        )
        style_image = (
            load_style(style, resize, normalize).unsqueeze(dim=0).to(self.device)
        )

        generated_image = (
            (torch.rand_like(content_image, device=self.device) - 0.5)
            if init_image == "noise"
            else content_image.clone()
        ).requires_grad_(True)

        return content_image, style_image, generated_image

    def _load_model(self, vgg_config, use_avg_pool, use_normalized_vgg):
        return VGGFeatures(
            config=vgg_config,
            use_avg_pool=use_avg_pool,
            use_normalized_vgg=use_normalized_vgg,
        ).to(self.device)

    def _train_step(self, style_features, content_features, step):
        generated_style, generated_content = self.vgg_features(self.generated_image)

        style_loss = self.compute_style_loss(generated_style, style_features)
        content_loss = self.compute_content_loss(generated_content, content_features)
        total_loss = (self.alpha * content_loss) + style_loss
        self._log(style_loss, content_loss, total_loss, step)

        return total_loss

    def _log(self, style_loss, content_loss, total_loss, step):
        _logs = {
            "style_loss": style_loss.detach().cpu().item(),
            "content_loss": content_loss.detach().cpu().item(),
            "total_loss": total_loss.detach().cpu().item(),
        }
        wandb.log(_logs, step=step)

    def _log_image(self, images, section="outputs", captions=None, step=None):
        assert isinstance(images, list) and isinstance(captions, list)
        assert len(images) == len(captions)

        with torch.no_grad():
            images = [
                np.transpose(denormalize(image[0]).detach().cpu().numpy(), (1, 2, 0))
                for image in images
            ]
            wandb.log(
                {
                    section: [
                        wandb.Image(image, caption=caption)
                        for (image, caption) in zip(images, captions)
                    ]
                },
                step=step,
            )


def main(args):
    wandb.config.update(args)

    args_dict = vars(args)
    model = OriginalStyleTransfer(**args_dict)
    model.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--content", type=str, default="tubingen", help="Content image")
    parser.add_argument("--style", type=str, default="starry_night", help="Style image")
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[512, 512],
        help="To resize the content and style image before proceeding",
    )
    parser.add_argument(
        "--normalize",
        type=bool,
        default=True,
        help="To normalize the content and style image before proceeding",
    )
    parser.add_argument(
        "--vgg_config",
        type=str,
        default="default",
        choices=FEATURES_CONFIG,
        help="Describes the set of layers to be used for extracting style/content features",
    )
    parser.add_argument(
        "--use_avg_pool",
        type=bool,
        default=True,
        help="Use average-pooling instead of max-pooling in VGG",
    )
    parser.add_argument(
        "--use_normalized_vgg",
        type=bool,
        default=True,
        help="Use the weight normalized VGG as described in papers",
    )
    parser.add_argument(
        "--init_image",
        type=str,
        default="noise",
        choices=["noise", "content"],
        help="Initial image to be stylized",
    )
    parser.add_argument(
        "--style_weighing",
        type=str,
        default="default",
        choices=losses.STYLE_WEIGHING_SCHEMES,
        help="Weight to be applied to different style layers",
    )
    parser.add_argument(
        "--alpha", type=float, default=1e-3, help="The content-style trade-off"
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="Number of L-BFGS iterations"
    )

    args = parser.parse_args()
    main(args)
