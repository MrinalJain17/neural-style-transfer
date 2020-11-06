from pathlib import Path

import torch
import torch.optim as optim
from pynache.data import load_content, load_style
from pynache.models import VGGFeatures
from pynache.models.losses import MSEWrapper, gram_matrix
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
        resize=(256, 256),
        normalize=True,
        vgg_config="default",
        use_avg_pool=True,
        init_image="noise",
        beta=1e3,
        steps=500,
        lr=1,
    ):
        self.device = DEVICE
        self.beta = beta
        self.steps = steps
        self.lr = lr
        self.normalize = normalize
        self.resize = resize
        assert init_image in ["noise", "content"], "Invalid initial image passed"

        self._load_images(content=content, style=style)
        self._load_model(
            vgg_config=vgg_config, use_avg_pool=use_avg_pool, init_image=init_image
        )

        self.compute_style_loss = MSEWrapper()
        self.compute_content_loss = MSEWrapper()

        self.style_loss_meter = AverageMeter()
        self.content_loss_meter = AverageMeter()

    def _load_images(self, content, style):
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

    def _load_model(self, vgg_config, use_avg_pool, init_image):
        self.vgg_features = VGGFeatures(
            config=vgg_config, use_avg_pool=use_avg_pool
        ).to(self.device)

        if init_image == "noise":
            self.generated_image = torch.randn_like(
                self.content_image, requires_grad=True, device=self.device
            )
        else:
            self.generated_image = self.content_image.clone().requires_grad_(True)

    def train(self):
        _, content_features = self.vgg_features(self.content_image)
        style_features, _ = self.vgg_features(self.style_image)
        style_gram = [gram_matrix(f) for f in style_features]

        optimizer = optim.LBFGS([self.generated_image], lr=self.lr)

        iterator = tqdm(range(self.steps), unit="step")
        for step in iterator:

            def closure():
                optimizer.zero_grad()
                loss = self.train_step(
                    style_gram=style_gram, content_features=content_features
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

    def train_step(self, style_gram, content_features):
        generated_style, generated_content = self.vgg_features(self.generated_image)
        generated_gram = [gram_matrix(f) for f in generated_style]

        style_loss = self.compute_style_loss(generated_gram, style_gram)
        content_loss = self.compute_content_loss(generated_content, content_features)
        total_loss = content_loss + (self.beta * style_loss)

        self.style_loss_meter.update(style_loss.detach().cpu().item())
        self.content_loss_meter.update(content_loss.detach().cpu().item())

        return total_loss


if __name__ == "__main__":
    Path(OUTPUT_PATH).mkdir(exist_ok=True)

    model = OriginalStyleTransfer("ancient_city", "starry_night")
    model.train()
