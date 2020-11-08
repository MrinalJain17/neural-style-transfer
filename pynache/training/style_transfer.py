from argparse import ArgumentParser
from typing import List

import torch
import torch.optim as optim
from pynache.data import load_content, load_style
from pynache.data.transforms import _MUL
from pynache.models import VGGFeatures, losses
from pynache.models.vgg import FEATURES_CONFIG
from pynache.training.logger import WandbLogger
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OriginalStyleTransfer(object):
    def __init__(
        self,
        content: str,
        style: str,
        resize: List[int] = [512, 512],
        vgg_config: str = "default",
        use_max_pool: bool = False,
        layer_weighing: str = "default",
        activation_shift: bool = False,
        chained_gram: bool = False,
        tv_strength: float = 0,
        alpha: float = 1e-3,
        init_image: str = "noise",
        steps: int = 500,
        logger: WandbLogger = None,
    ):
        self.device = DEVICE
        self.tv_strength = tv_strength
        self.alpha = alpha
        self.steps = steps
        self.logger = logger

        self.content_image, self.style_image, self.generated_image = self._load_images(
            content, style, init_image, resize
        )
        self.vgg_features = self._load_model(vgg_config, use_max_pool)

        _loss_fx = losses.StyleLossChained if chained_gram else losses.StyleLoss
        self.compute_style_loss = _loss_fx(
            num_layers=len(self.vgg_features.style_layers),
            weights=layer_weighing,
            activation_shift=activation_shift,
        )
        self.compute_content_loss = losses.ContentLoss()
        self.compute_total_variation = losses.TotalVariation()

    def _load_images(self, content, style, init_image, resize):
        assert init_image in ["noise", "content"], "Invalid initial image passed"

        content_image = (
            load_content(content, resize, normalize=True)
            .unsqueeze(dim=0)
            .to(self.device)
        )
        style_image = (
            load_style(style, resize, normalize=True).unsqueeze(dim=0).to(self.device)
        )

        generated_image = (
            (torch.rand_like(content_image, device=self.device) * _MUL) - 128
            if init_image == "noise"
            else content_image.clone()
        ).requires_grad_(True)

        return content_image, style_image, generated_image

    def _load_model(self, vgg_config, use_max_pool):
        return VGGFeatures(
            config=vgg_config, use_avg_pool=(not use_max_pool), use_normalized_vgg=True,
        ).to(self.device)

    def train(self):
        self.logger._log_image(
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

            if (step == 0) or ((step + 1) % 20 == 0):
                self.logger._log_image(
                    [self.generated_image],
                    "outputs",
                    captions=[f"Iteration {step + 1}"],
                    step=step,
                )

    def _train_step(self, style_features, content_features, step):
        generated_style, generated_content = self.vgg_features(self.generated_image)

        style_loss = self.compute_style_loss(generated_style, style_features)
        content_loss = self.compute_content_loss(generated_content, content_features)
        total_variation = self.compute_total_variation(self.generated_image)
        total_loss = (
            (self.alpha * content_loss)
            + style_loss
            + (self.tv_strength * total_variation)
        )

        self.logger._log(style_loss, content_loss, total_loss, step)
        return total_loss


def main(args):
    args_dict = vars(args)
    is_improved = (
        (args.vgg_config != "default")
        or (args.layer_weighing != "default")
        or (args.tv_strength != 0)
        or args.activation_shift
        or args.chained_gram
    )

    name = "Improved Style Transfer" if is_improved else "Original Style Transfer"
    logger = WandbLogger(name=name, args=args)
    args_dict["logger"] = logger

    model = OriginalStyleTransfer(**args_dict)
    model.train()


if __name__ == "__main__":
    parser = ArgumentParser()

    # Input images arguments
    parser.add_argument("--content", type=str, default="tubingen", help="Content image")
    parser.add_argument("--style", type=str, default="starry_night", help="Style image")
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[512, 512],
        help="To resize the content and style image before proceeding",
    )

    # VGG (feature maps related) arguments
    parser.add_argument(
        "--vgg_config",
        type=str,
        default="default",
        choices=list(FEATURES_CONFIG.keys()),
        help="Describes the set of layers to be used for extracting style/content features",
    )
    parser.add_argument(
        "--use_max_pool",
        action="store_true",
        default=False,
        help="Use max-pooling instead of average-pooling in the VGG network",
    )

    # Suggested improvements for loss functions and gram matrix computation
    parser.add_argument(
        "--layer_weighing",
        type=str,
        default="default",
        choices=losses.STYLE_WEIGHING_SCHEMES,
        help="Weight applied to the gram matrices obtained from different style layers",
    )
    parser.add_argument(
        "--activation_shift",
        action="store_true",
        default=False,
        help="Shift the activations when computing the gram matrix",
    )
    parser.add_argument(
        "--chained_gram",
        action="store_true",
        default=False,
        help="Compute gram matrices between adjacent layers (chaining)",
    )
    parser.add_argument(
        "--tv_strength",
        type=float,
        default=0,
        help="Strength of the total variation loss",
    )

    # Training/optimization arguments
    parser.add_argument(
        "--alpha", type=float, default=1e-3, help="The content-style ratio"
    )
    parser.add_argument(
        "--init_image",
        type=str,
        default="noise",
        choices=["noise", "content"],
        help="Initial image to be stylized",
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="Number of L-BFGS iterations"
    )

    args = parser.parse_args()
    main(args)
