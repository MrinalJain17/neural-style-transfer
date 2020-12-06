from argparse import ArgumentParser
from typing import List

from pynache.data import load_content, load_style
from pynache.models import losses, VGGFeatures
from pynache.models.vgg import FEATURES_CONFIG
from pynache.training.logger import WandbLogger
from pynache.utils import to_numpy, to_tensor
from skimage.exposure import match_histograms
import torch
import torch.optim as optim
from tqdm import tqdm

SEED = 42999
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralStyleTransfer(object):
    def __init__(
        self,
        content: str,
        style: str,
        resize: List[int] = [512, 512],
        preserve_color: bool = False,
        activation_shift: bool = False,
        vgg_config: str = "default",
        chained_gram: bool = False,
        layer_weighing: str = "default",
        tv_strength: float = 0,
        alpha: float = 1e-6,
        init_image: str = "content",
        use_max_pool: bool = False,
        steps: int = None,
        use_lbfgs: bool = False,
        logger: WandbLogger = None,
    ):
        self.device = DEVICE
        self.preserve_color = preserve_color
        self.tv_strength = tv_strength
        self.alpha = alpha
        self.steps = steps
        self.use_lbfgs = use_lbfgs
        self.logger = logger

        if self.steps is None:
            self.steps = 500 if self.use_lbfgs else 10000
            self.logger.config.update({"steps": self.steps}, allow_val_change=True)

        self._log_every_n_steps = self.steps // 20
        self._current_step = None

        self.content_image, self.style_image, self.generated_image = self._load_images(
            content, style, init_image, resize
        )
        self.vgg_features = VGGFeatures(
            config=vgg_config, use_avg_pool=(not use_max_pool)
        ).to(self.device)

        _loss_fx = losses.StyleLossChained if chained_gram else losses.StyleLoss
        self.compute_style_loss = torch.jit.script(
            _loss_fx(
                num_layers=len(self.vgg_features.style_layers),
                weights=layer_weighing,
                activation_shift=activation_shift,
            )
        )
        self.compute_content_loss = losses.ContentLoss()
        self.compute_total_variation = losses.TotalVariation()

    def _load_images(self, content, style, init_image, resize):
        assert init_image in ["noise", "content"], "Invalid initial image passed"

        gray = True if self.preserve_color else False
        content_image, style_image = (
            to_numpy(load_content(content, resize, gray=gray)),
            to_numpy(load_style(style, resize, gray=gray)),
        )

        if self.preserve_color:
            # Histogram matching as described in https://arxiv.org/abs/1606.05897
            # for better color preservation.
            style_image = match_histograms(
                style_image, content_image, multichannel=True
            )

        content_image, style_image = (
            to_tensor(content_image).unsqueeze(dim=0).to(self.device),
            to_tensor(style_image).unsqueeze(dim=0).to(self.device),
        )

        generated_image = (
            content_image.clone()
            if init_image == "content"
            else (torch.rand_like(content_image, device=self.device) - 0.5) / 0.5
        ).requires_grad_(True)

        # Used for logging, in case any modifications are made to style/content image
        self.original_images = {
            "content": load_content(content, resize).unsqueeze(dim=0).to(self.device),
            "style": load_style(style, resize).unsqueeze(dim=0).to(self.device),
        }

        return content_image, style_image, generated_image

    def train(self):
        _, content_features = self.vgg_features(self.content_image)
        style_features, _ = self.vgg_features(self.style_image)

        optimizer = (
            optim.LBFGS([self.generated_image])
            if self.use_lbfgs
            else optim.Adam([self.generated_image])
        )

        def closure():
            optimizer.zero_grad()
            loss = self._train_step(style_features, content_features)
            loss.backward()
            return loss

        for step in tqdm(range(self.steps), unit="step"):
            self._current_step = step
            optimizer.step(closure)

            if (step == 0) or ((step + 1) % self._log_every_n_steps == 0):
                self.logger.log_samples(self.generated_image, step=step)

        # Getting back the original content/style images for logging.
        # This is required when preserving color, since only the luminance channel
        # is used in the entire style transfer process.
        self.content_image, self.style_image = (
            self.original_images["content"],
            self.original_images["style"],
        )

        self.logger.log_results(
            self.content_image,
            self.style_image,
            self.generated_image,
            self._current_step,
            self.preserve_color,
        )

    def _train_step(self, style_features, content_features):
        generated_style, generated_content = self.vgg_features(self.generated_image)

        style_loss = self.compute_style_loss(generated_style, style_features)
        content_loss = self.compute_content_loss(generated_content, content_features)
        total_variation = (
            0
            if self.tv_strength == 0
            else self.compute_total_variation(self.generated_image)
        )

        total_loss = (
            (self.alpha * content_loss)
            + style_loss
            + (self.tv_strength * total_variation)
        )

        self.logger.log_loss(style_loss, content_loss, total_loss, self._current_step)
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

    torch.manual_seed(SEED)

    model = NeuralStyleTransfer(**args_dict)
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
    parser.add_argument(
        "--preserve_color",
        action="store_true",
        default=False,
        help="To preserve the color of content image",
    )

    # Suggested improvements for loss functions and gram matrix computation
    parser.add_argument(
        "--activation_shift",
        action="store_true",
        default=False,
        help="Shift the activations when computing the gram matrix",
    )
    parser.add_argument(
        "--vgg_config",
        type=str,
        default="default",
        choices=list(FEATURES_CONFIG.keys()),
        help="Describes the set of layers to be used for extracting style/content features",
    )
    parser.add_argument(
        "--chained_gram",
        action="store_true",
        default=False,
        help="Compute gram matrices between adjacent layers (chaining)",
    )
    parser.add_argument(
        "--layer_weighing",
        type=str,
        default="default",
        choices=losses.STYLE_WEIGHING_SCHEMES,
        help="Weight applied to the gram matrices obtained from different style layers",
    )
    parser.add_argument(
        "--tv_strength",
        type=float,
        default=0,
        help="Strength of the total variation loss",
    )

    # Training/optimization arguments
    parser.add_argument(
        "--alpha", type=float, default=1e-6, help="The content-style ratio"
    )
    parser.add_argument(
        "--init_image",
        type=str,
        default="content",
        choices=["noise", "content"],
        help="Initial image to be stylized",
    )
    parser.add_argument(
        "--use_max_pool",
        action="store_true",
        default=False,
        help="Use max-pooling instead of average-pooling in the VGG network",
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of iterations to run"
    )
    parser.add_argument(
        "--use_lbfgs",
        action="store_true",
        default=False,
        help="Use the L-BFGS optimizer instead of Adam",
    )

    args = parser.parse_args()
    main(args)
