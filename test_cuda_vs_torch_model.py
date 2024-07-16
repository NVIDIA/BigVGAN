# Copyright (c) 2024 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import math
import torch
import json
from env import AttrDict
from bigvgan import BigVGAN
from time import time
from tqdm import tqdm
import os
from meldataset import mel_spectrogram, MAX_WAV_VALUE
import librosa
from scipy.io.wavfile import write
import numpy as np

import argparse

# for easier debugging
torch.set_printoptions(
    linewidth=200,
    threshold=10_000
)

def generate_soundwave(duration=5.0, sr=24000):
    t = np.linspace(0, duration, int(sr * duration), False, dtype=np.float32)
    
    modulation = np.sin(2 * np.pi * t / duration)

    min_freq = 220
    max_freq = 1760
    frequencies = min_freq + (max_freq - min_freq) * (modulation + 1) / 2
    soundwave = np.sin(2 * np.pi * frequencies * t)
    
    soundwave = soundwave / np.max(np.abs(soundwave)) * 0.95

    return soundwave, sr

def get_mel(x, h):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script to check CUDA kernel correctness.")
    parser.add_argument('--checkpoint_file', type=str, required=True, help="Path to the checkpoint file. Assumes config.json exists in the directory.")
    
    args = parser.parse_args()
    
    config_file = os.path.join(os.path.split(args.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        config = f.read()
    json_config = json.loads(config)
    h = AttrDict({**json_config})
    
    print("loading plain Pytorch BigVGAN")
    generator_original = BigVGAN(h).to("cuda")
    print("loading CUDA kernel BigVGAN with auto-build")
    generator_cuda_kernel = BigVGAN(h, use_cuda_kernel=True).to("cuda")
    
    state_dict_g = load_checkpoint(args.checkpoint_file, "cuda")
    generator_original.load_state_dict(state_dict_g['generator'])
    generator_cuda_kernel.load_state_dict(state_dict_g['generator'])
    
    generator_original.eval()
    generator_original.remove_weight_norm()
    generator_cuda_kernel.eval()
    generator_cuda_kernel.remove_weight_norm()
        
    toc_total_original = 0.
    toc_total_cuda_kernel = 0.
    audio_length_total = 0.
    diff = 0.
    
    num_sample = 10
    num_mel_frame = 128
    for i in tqdm(range(num_sample)):
        # random mel: use large num_mel_frame to test peak gpu util performance
        data = torch.rand((1, h.num_mels, num_mel_frame), device='cuda')
        # original inference
        torch.cuda.synchronize()
        tic = time()
        with torch.inference_mode():
            audio_original = generator_original(data)
            torch.cuda.synchronize()
            toc = time() - tic
        toc_total_original += toc
        # cuda kernel inference
        torch.cuda.synchronize()
        tic = time()
        with torch.inference_mode():
            audio_cuda_kernel = generator_cuda_kernel(data)
            torch.cuda.synchronize()
            toc = time() - tic
        toc_total_cuda_kernel += toc
        audio_length_total += audio_cuda_kernel.shape[-1]

        # both outputs should be (almost) the same 
        test_result = (audio_original - audio_cuda_kernel).abs()
        diff += test_result.mean(dim=-1).item()

    diff /= num_sample
    if diff <= 2e-3: # we can expect a small difference (~1e-3) which does not affect perceptual quality
        print(
            f"\n[Success] test CUDA fused vs. plain torch BigVGAN inference"
            f"\n > mean_difference={diff}"
            f"\n > fused_values={audio_cuda_kernel[-1][-1][-30:].tolist()}"
            f"\n > torch_values={audio_original[-1][-1][-30:].tolist()}"
        )
    else:
        print(
            f"\n[Fail] test CUDA fused vs. plain torch BigVGAN inference"
            f"\n > mean_difference={diff}"
            f"\n > fused_values={audio_cuda_kernel[-1][-1][-30:].tolist()}, "
            f"\n > torch_values={audio_original[-1][-1][-30:].tolist()}"
        )
        
    audio_second = audio_length_total / h.sampling_rate      
    khz_original = audio_length_total / toc_total_original / 1000
    khz_cuda_kernel = audio_length_total / toc_total_cuda_kernel / 1000

    print('Original BigVGAN: took {:.2f} seconds to generate {:.2f} seconds of audio, {:.1f}kHz, {:.1f} faster than realtime'.format(toc_total_original, audio_second, khz_original, audio_second / toc_total_original))
    print('CUDA kernel BigVGAN: took {:.2f} seconds to generate {:.2f} seconds of audio, {:.1f}kHz, {:.1f} faster than realtime'.format(toc_total_cuda_kernel, audio_second, khz_cuda_kernel, audio_second / toc_total_cuda_kernel))
    print('speedup of CUDA kernel: {}'.format(khz_cuda_kernel/khz_original))
    
    # use artificial sine waves for inference test
    audio_real, sr = generate_soundwave(duration=5., sr=h.sampling_rate)
    audio_real = torch.tensor(audio_real).to("cuda")
    # compute mel spectrogram from the ground truth audio
    x = get_mel(audio_real.unsqueeze(0), h)

    with torch.inference_mode():
        y_g_hat_original = generator_original(x)
        y_g_hat_cuda_kernel = generator_cuda_kernel(x)
        
    audio_real = audio_real.squeeze()
    audio_real = audio_real * MAX_WAV_VALUE
    audio_real = audio_real.cpu().numpy().astype('int16')
    
    audio_original = y_g_hat_original.squeeze()
    audio_original = audio_original * MAX_WAV_VALUE
    audio_original = audio_original.cpu().numpy().astype('int16')
    
    audio_cuda_kernel = y_g_hat_cuda_kernel.squeeze()
    audio_cuda_kernel = audio_cuda_kernel * MAX_WAV_VALUE
    audio_cuda_kernel = audio_cuda_kernel.cpu().numpy().astype('int16')

    os.makedirs('tmp', exist_ok=True)
    output_file_real = os.path.join('tmp', 'audio_real.wav')
    output_file_original = os.path.join('tmp', 'audio_generated_original.wav')
    output_file_cuda_kernel = os.path.join('tmp', 'audio_generated_cuda_kernel.wav')
    write(output_file_real, h.sampling_rate, audio_real)
    write(output_file_original, h.sampling_rate, audio_original)
    write(output_file_cuda_kernel, h.sampling_rate, audio_cuda_kernel)
    print("Example generated audios of original vs. fused CUDA kernel written to tmp!")
    print("Done")