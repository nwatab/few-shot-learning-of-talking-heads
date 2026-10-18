# few-shot-learning-of-talking-heads
Unofficial implementation of "Egor Zakharov et al. Few-Shot Adversarial Learning of Realistic Neural Talking Head Models, 2019"

## Work In Progress
I created generator, embedder, and discriminator on Keras. After a few days of training, it just infers black images:( I am investigating the cause. I opened this repository to get your contribution. Feel free to open an issue and make a pull-request.

![results/0_300/7.jpg](https://raw.githubusercontent.com/asterisk37n/few-shot-learning-of-talking-heads/master/7.jpg)

### Fix: VGG preprocessing
The perceptual losses fed generator outputs in [-1, 1] RGB straight into VGG19 and VGGFace, which were trained on Caffe-style input (BGR, 0-255, ImageNet / VGGFace mean subtracted). `models.py` now converts frames to that format before both networks. In a small-scale run (64px, 24 identities, 1500 steps, CPU) this removed most of the 16px block pattern seen above; it has not been verified at 256px on VoxCeleb2.

## Docker
TensorFlow 1.14 / Keras 2.2.4 on Python 3.6 (versions in `requirements.txt`). TensorFlow 1.x has no arm64 wheels, so on Apple Silicon build and run with `--platform linux/amd64` (CPU emulation, slow).

```sh
docker build --platform linux/amd64 -t fsth .
# The scripts read ../few-shot-learning-of-talking-heads/datasets/..., so mount the repository under that name.
mkdir -p logs trained_models
docker run --rm --platform linux/amd64 \
  -v "$PWD":/few-shot-learning-of-talking-heads -w /few-shot-learning-of-talking-heads \
  -v "$HOME/.keras":/root/.keras \
  fsth python meta_learning.py
```

`fewshot_learning.py` and `prediction.py` run the same way.
