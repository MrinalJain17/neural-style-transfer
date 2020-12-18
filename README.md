# Neural Style Transfer

The following papers are implemented in this repository:

1. [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
2. [Improving the Neural Algorithm of Artistic Style](https://arxiv.org/abs/1605.04603)
3. [Preserving Color in Neural Artistic Style Transfer](https://arxiv.org/abs/1606.05897)
4. [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)

## Examples

![example_1](https://raw.githubusercontent.com/MrinalJain17/neural-style-transfer/main/report/figures/kinkaku_ji_starry_night.png?token=AEV35NP2M7D45FTJ7F6OMMC74UX5O)

---

![example_2](https://raw.githubusercontent.com/MrinalJain17/neural-style-transfer/main/report/figures/kinkaku_ji_the_scream.png?token=AEV35NNXTHKQZ4TOSIQ6RIK74UX5M)

---

![example_3](https://raw.githubusercontent.com/MrinalJain17/neural-style-transfer/main/report/figures/night_skyline_seated_nude.png?token=AEV35NORVL33X3C2JJVS4PS74UX5S)

---

**We have setup an interactive report with a bunch of other visualizations - [Link](https://wandb.ai/mrinaljain17/neural-style-transfer/reports/Neural-Style-Transfer--VmlldzozNzExODQ).**

## Installation

Execute the following command from the root of the repository to install the project:

```bash
    pip install -e .
```

**Note that this step is required to run the project.**

## Requirements

### Base Requirements

1. Python (3.7)
2. PyTorch (1.7)
3. Torchvision (0.8)
4. Tqdm - For displaying progress bars
5. [Weights and Biases](https://github.com/wandb/client) - For visualization

### Additional Requirements

[PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) (1.0) is required
for executing Fast Style Transfer.

## References

1. [Neural Style Transfer by Somshubra Majumdar](https://github.com/titu1994/Neural-Style-Transfer)
