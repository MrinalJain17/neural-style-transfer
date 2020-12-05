import torch
from pynache.data.load_inputs import load_content
from pynache.data.transforms import denormalize
from pynache.utils import to_numpy
from pytorch_lightning.callbacks import Callback
from wandb import Image


class ExamplesLoggingCallback(Callback):
    """Callback to upload sample predictions to W&B."""

    def __init__(self, content="kinkaku_ji", seed=None) -> None:
        super().__init__()
        self.content_image = load_content(content, resize=[256, 256]).unsqueeze(0)

    def on_fit_start(self, trainer, pl_module):
        self.style_image = pl_module.style_image[:1, :, :, :]  # Shape: (1, C, H, W)
        self.content_image = self.content_image.to(pl_module.device)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        pl_module.eval()
        with torch.no_grad():
            generated_image = pl_module.forward(self.content_image)
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
