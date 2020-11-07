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
        wandb.init(name=name, project=project, dir=ARTIFACTS_PATH)
        wandb.config.update(args)

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
