from pathlib import Path

import torch
import wandb
from pynache.data.transforms import denormalize
from pynache.paths import REPOSITORY_ROOT
from pynache.training.utils import _add_color
from pynache.utils import to_numpy

ARTIFACTS_PATH = (Path(REPOSITORY_ROOT) / "wandb_artifacts").as_posix()
Path(ARTIFACTS_PATH).mkdir(exist_ok=True)


class WandbLogger(object):
    def __init__(self, name, args, project="neural-style-transfer"):
        wandb.init(name=name, project=project, dir=ARTIFACTS_PATH)
        wandb.config.update(args)

    @property
    def config(self):
        return wandb.config

    def log_loss(self, style_loss, content_loss, total_loss, step):
        _logs = {
            "style_loss": style_loss.detach().cpu().item(),
            "content_loss": content_loss.detach().cpu().item(),
            "total_loss": total_loss.detach().cpu().item(),
        }
        wandb.log(_logs, step=step)

    def log_results(
        self, content_image, style_image, generated_image, step, add_color=False,
    ):
        if add_color:
            _generated, _content = generated_image.clone(), content_image.clone()
            generated_image = _add_color(_content, _generated)

        content_image = self._prepare_image(content_image)
        style_image = self._prepare_image(style_image)
        generated_image = self._prepare_image(generated_image)

        self._log_image(
            images=[content_image, style_image, generated_image],
            section="results",
            captions=["Content image", "Style image", "Generated Image"],
            step=step,
        )

    def log_samples(self, generated_image, step):
        generated_image = self._prepare_image(generated_image)
        self._log_image(
            images=[generated_image],
            section="samples",
            captions=[f"Iteration {step + 1}"],
            step=step,
        )

    def _prepare_image(self, image: torch.Tensor):
        assert image.ndim == 4, "Expected input of shape (1, C, H, W)"
        with torch.no_grad():
            image = to_numpy(denormalize(image[0]).detach().cpu())
        return image

    def _log_image(self, images, section="samples", captions=None, step=None):
        wandb.log(
            {
                section: [
                    wandb.Image(image, caption=caption)
                    for (image, caption) in zip(images, captions)
                ]
            },
            step=step,
        )
