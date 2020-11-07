from pathlib import Path

import torch
import torch.optim as optim
from pynache.data import load_content, load_style
from pynache.models import VGGFeatures
from pynache.models.losses import ContentLoss, StyleLoss
from pynache.models.utils import AverageMeter, denormalize
from pynache.paths import DEFAULT_IMAGES
from torchvision.utils import save_image
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PATH = (Path(DEFAULT_IMAGES) / "sample_outputs").as_posix()


class OriginalStyleTransfer(object):
    def __init__(
        self,
        content,
        style,
        resize=(512, 512),
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
        self.normalize = normalize
        self.resize = resize
        assert init_image in ["noise", "content"], "Invalid initial image passed"

        self._load_images(content, style, init_image)
        self._load_model(vgg_config, use_avg_pool, use_normalized_vgg)

        self.compute_style_loss = StyleLoss(
            num_layers=len(self.vgg_features.style_layers), weights=style_weighing
        )
        self.compute_content_loss = ContentLoss()

        self.style_loss_meter = AverageMeter()
        self.content_loss_meter = AverageMeter()

    def _load_images(self, content, style, init_image):
        self.content_image = (
            load_content(content, self.resize, self.normalize)
            .unsqueeze(dim=0)
            .to(self.device)
        )
        self.style_image = (
            load_style(style, self.resize, self.normalize)
            .unsqueeze(dim=0)
            .to(self.device)
        )
        if init_image == "noise":
            self.generated_image = torch.rand_like(
                self.content_image, requires_grad=True, device=self.device
            )
            with torch.no_grad():
                self.generated_image -= 0.5
        else:
            self.generated_image = self.content_image.clone().requires_grad_(True)

    def _load_model(self, vgg_config, use_avg_pool, use_normalized_vgg):
        self.vgg_features = VGGFeatures(
            config=vgg_config,
            use_avg_pool=use_avg_pool,
            use_normalized_vgg=use_normalized_vgg,
        ).to(self.device)

    def train(self):
        _, content_features = self.vgg_features(self.content_image)
        style_features, _ = self.vgg_features(self.style_image)

        optimizer = optim.LBFGS([self.generated_image])

        iterator = tqdm(range(self.steps), unit="step")
        for step in iterator:

            def closure():
                optimizer.zero_grad()
                loss = self.train_step(
                    style_features=style_features, content_features=content_features
                )
                loss.backward()
                return loss

            optimizer.step(closure)

            iterator.set_postfix(
                {
                    "Style loss": self.style_loss_meter.last,
                    "Content loss": self.content_loss_meter.last,
                }
            )

            if step % 20 == 0:
                with torch.no_grad():
                    sample = self.generated_image[0]
                    if self.normalize:
                        sample = denormalize(sample)
                    save_image(sample, f"{OUTPUT_PATH}/generated_{step}.jpg")

    def train_step(self, style_features, content_features):
        generated_style, generated_content = self.vgg_features(self.generated_image)

        style_loss = self.compute_style_loss(generated_style, style_features)
        content_loss = self.compute_content_loss(generated_content, content_features)
        total_loss = (self.alpha * content_loss) + style_loss

        self.style_loss_meter.update(style_loss.detach().cpu().item())
        self.content_loss_meter.update(content_loss.detach().cpu().item())

        return total_loss


if __name__ == "__main__":
    Path(OUTPUT_PATH).mkdir(exist_ok=True)

    model = OriginalStyleTransfer(content="tubingen", style="starry_night")
    model.train()
