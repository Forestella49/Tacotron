import matplotlib.pylab as plt
import sys
import numpy as np
import torch
import librosa
from model import Tacotron2
from layers import TacotronSTFT
from stft import STFT
from train import load_model
from text import text_to_sequence
from hparams import add_hparams, get_hparams
import argparse
import os
from scipy.io.wavfile import write
from VocGAN.model.generator import ModifiedGenerator
from VocGAN.utils.hparams import HParam, load_hparam_str
from VocGAN.denoiser import Denoiser


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')
    plt.savefig(r"C:\Users\Admin\PycharmProjects\tacotron2_son\sample\mel_dir\mel_345000_4.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_hparams(parser)
    args = parser.parse_args()

    hparams = get_hparams(args, parser)

    checkpoint_path = r"C:\Users\Admin\PycharmProjects\tacotron2_son\checkpoint\checkpoint_345000"
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval().half()

    text = "이것은 육천에폭짜리 음성이다."
    sequence = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    plot_data((mel_outputs.float().data.cpu().numpy()[0],
               mel_outputs_postnet.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T))
    mel = mel_outputs_postnet.float().data.cpu()
    filter_length = 1024
    hop_length = 256
    win_length = 1024
    sampling_rate = 22050
    mel_fmin = 0.0
    mel_fmax = None
    taco_stft = TacotronSTFT(
        filter_length, hop_length, win_length,
        sampling_rate=sampling_rate, mel_fmin=mel_fmin,
        mel_fmax=mel_fmax)

    # Project from Mel-Spectrogram to Spectrogram
    mel_decompress = taco_stft.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    filter_length = 1024
    hop_length = 256
    win_length = 1024
    sampling_rate = 22050
    mel_fmin = 0.0
    mel_fmax = 8000.0

    taco_stft_other = TacotronSTFT(
        filter_length, hop_length, win_length,
        sampling_rate=sampling_rate, mel_fmin=mel_fmin, mel_fmax=mel_fmax)

    # Project from Spectrogram to r9y9's WaveNet Mel-Spectrogram
    mel_minmax = taco_stft_other.spectral_normalize(
        torch.matmul(taco_stft_other.mel_basis, spec_from_mel))

    checkpoint = torch.load(r"C:\Users\Admin\PycharmProjects\tacotron2_son\son_voc_c6ef83b_6000.pt")
    hp = HParam(r"C:\Users\Admin\PycharmProjects\tacotron2_son\VocGAN\config\config.yaml")

    model = ModifiedGenerator(hp.audio.n_mel_channels, hp.model.n_residual_layers,
                              ratios=hp.model.generator_ratio, mult=hp.model.mult,
                              out_band=hp.model.out_channels).cuda()
    MAX_WAV_VALUE = 32768.0
    model.load_state_dict(checkpoint['model_g'])
    model.eval(inference=True)

    with torch.no_grad():
        mel = mel_minmax
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.cuda()
        audio = model.inference(mel)

        audio = audio.squeeze(0)  # collapse all dimension except time axis
        d = True  # 이상하면 False하셈
        if d:
            denoiser = Denoiser(model).cuda()
            audio = denoiser(audio, 0.01)
        audio = audio.squeeze()
        audio = audio[:-(hp.audio.hop_length * 10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE - 1)
        audio = audio.short()
        audio2 = audio.float().cpu().detach().numpy()
        out_path = r"C:\Users\Admin\PycharmProjects\tacotron2_son\sample\audio_dir\audio_345000_6.wav"
        yt, index = librosa.effects.trim(audio2, frame_length=5120, hop_length=256, top_db=50)
        audio = audio.cpu().detach().numpy()
        audio = audio[:index[-1]]
        write(out_path, hp.audio.sampling_rate, audio)

