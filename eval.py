import matplotlib.pylab as plt
import sys
import numpy as np
import torch
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from hparams import add_hparams, get_hparams
import argparse


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')
    plt.savefig(r"C:\Users\Admin\PycharmProjects\tacotron2_son\checkpoint\mel_98000.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_hparams(parser)
    args = parser.parse_args()

    hparams = get_hparams(args, parser)

    checkpoint_path = r"C:\Users\Admin\PycharmProjects\tacotron2_son\checkpoint\checkpoint_98000"
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval().half()



    text = "우리는 인공지능을 공부하고 있습니다"
    sequence = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
                                      torch.from_numpy(sequence)).cuda().long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    plot_data((mel_outputs.float().data.cpu().numpy()[0],
            mel_outputs_postnet.float().data.cpu().numpy()[0],
            alignments.float().data.cpu().numpy()[0].T))