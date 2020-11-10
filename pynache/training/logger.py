from pathlib import Path

import numpy as np
import torch
import wandb
from pynache.data.transforms import denormalize
from pynache.paths import REPOSITORY_ROOT

ARTIFACTS_PATH = (Path(REPOSITORY_ROOT) / "wandb_artifacts").as_posix()
Path(ARTIFACTS_PATH).mkdir(exist_ok=True)


class WandbLogger(object):
    def __init__(self, name, args, project="neural-style-transfer"):
        wandb.init(
            name=name, project=project, dir=ARTIFACTS_PATH, allow_val_change=True
        )
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

    def log_inputs(self, content_image, style_image, step=0):
        content_image = self._prepare_image(content_image)
        style_image = self._prepare_image(style_image)

        self._log_image(
            images=[content_image, style_image],
            section="inputs",
            captions=["Content image", "Style image"],
            step=step,
        )

    def log_outputs(self, generated_image, step):
        with torch.no_grad():
            generated_image = self._prepare_image(generated_image)
            self._log_image(
                images=[generated_image],
                section="outputs",
                captions=[f"Iteration {step + 1}"],
                step=step,
            )

    def _prepare_image(self, image: torch.Tensor):
        assert image.ndim == 4, "Expected input of shape (B, C, H, W)"
        return np.transpose(denormalize(image[0]).detach().cpu().numpy(), (1, 2, 0))

    def _log_image(self, images, section="outputs", captions=None, step=None):
        wandb.log(
            {
                section: [
                    wandb.Image(image, caption=caption)
                    for (image, caption) in zip(images, captions)
                ]
            },
            step=step,
        )
