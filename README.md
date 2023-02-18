## BigVGAN: A Universal Neural Vocoder with Large-Scale Training
#### Sang-gil Lee, Wei Ping, Boris Ginsburg, Bryan Catanzaro, Sungroh Yoon

<center><img src="https://user-images.githubusercontent.com/15963413/218609148-881e39df-33af-4af9-ab95-1427c4ebf062.png" width="800"></center>


### [Paper](https://arxiv.org/abs/2206.04658)
### [Audio demo](https://bigvgan-demo.github.io/)

## Installation
Clone the repository and install dependencies.
```shell
# the codebase has been tested on Python 3.8 / 3.10 with PyTorch 1.12.1 / 1.13 conda binaries
git clone https://github.com/NVIDIA/BigVGAN
pip install -r requirements.txt
```

Create symbolic link to the root of the dataset. The codebase uses filelist with the relative path from the dataset. Below are the example commands for LibriTTS dataset.
``` shell
cd LibriTTS && \
ln -s /path/to/your/LibriTTS/train-clean-100 train-clean-100 && \
ln -s /path/to/your/LibriTTS/train-clean-360 train-clean-360 && \
ln -s /path/to/your/LibriTTS/train-other-500 train-other-500 && \
ln -s /path/to/your/LibriTTS/dev-clean dev-clean && \
ln -s /path/to/your/LibriTTS/dev-other dev-other && \
ln -s /path/to/your/LibriTTS/test-clean test-clean && \
ln -s /path/to/your/LibriTTS/test-other test-other && \
cd ..
```

## Training
Train BigVGAN model. Below is an example command for training BigVGAN using LibriTTS dataset at 24kHz with a full 100-band mel spectrogram as input.
```shell
python train.py \
--config configs/bigvgan_24khz_100band.json \
--input_wavs_dir LibriTTS \
--input_training_file LibriTTS/train-full.txt \
--input_validation_file LibriTTS/val-full.txt \
--list_input_unseen_wavs_dir LibriTTS LibriTTS \
--list_input_unseen_validation_file LibriTTS/dev-clean.txt LibriTTS/dev-other.txt \
--checkpoint_path exp/bigvgan
```

## Synthesis
Synthesize from BigVGAN model. Below is an example command for generating audio from the model.
It computes mel spectrograms using wav files from `--input_wavs_dir` and saves the generated audio to `--output_dir`.
```shell
python inference.py \
--checkpoint_file exp/bigvgan/g_05000000 \
--input_wavs_dir /path/to/your/input_wav \
--output_dir /path/to/your/output_wav
```

`inference_e2e.py` supports synthesis directly from the mel spectrogram saved in `.npy` format, with shapes `[1, channel, frame]` or `[channel, frame]`.
It loads mel spectrograms from `--input_mels_dir` and saves the generated audio to `--output_dir`.

Make sure that the STFT hyperparameters for mel spectrogram are the same as the model, which are defined in `config.json` of the corresponding model.
```shell
python inference_e2e.py \
--checkpoint_file exp/bigvgan/g_05000000 \
--input_mels_dir /path/to/your/input_mel \
--output_dir /path/to/your/output_wav
```


## TODO

Pretrained weights are expected to be released soon.


Current codebase only provides a plain PyTorch implementation for the filtered nonlinearity. We are working on a fast CUDA kernel implementation, which will be released in the future. 


## References
HiFi-GAN (for generator and multi-period discriminator): [https://github.com/jik876/hifi-gan](https://github.com/jik876/hifi-gan)

Snake (for periodic activation): [https://github.com/EdwardDixon/snake](https://github.com/EdwardDixon/snake)

alias-free-torch (for anti-aliasing): [https://github.com/junjun3518/alias-free-torch](https://github.com/junjun3518/alias-free-torch)

Julius (for low-pass filter): [https://github.com/adefossez/julius](https://github.com/adefossez/julius)

UnivNet (for multi-resolution discriminator): [https://github.com/mindslab-ai/univnet](https://github.com/mindslab-ai/univnet)
