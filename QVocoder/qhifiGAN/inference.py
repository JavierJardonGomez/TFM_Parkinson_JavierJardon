from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldatasetQ import mel_spectrogram, MAX_WAV_VALUE, load_wav
from modelsQ5B import QGenerator
import torchaudio 

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = QGenerator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    # generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            data= get_mel(wav.unsqueeze(0))

            data_first = torchaudio.functional.compute_deltas(data)
            data_second = torchaudio.functional.compute_deltas(data_first)
            data_third = torchaudio.functional.compute_deltas(data_second)

            x = torch.cat([data,data_first, data_second, data_third], dim=1)
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    #parser.add_argument('--input_wavs_dir', default='../LJSpeech-1.1/test_wavs')
    #parser.add_argument('--input_wavs_dir', default='../wavs_e_22050Hz')
    parser.add_argument('--input_wavs_dir', default='../wavs_e_22050Hz_normalize')
    #parser.add_argument('--input_wavs_dir', default='../pre_processed_input_wavs_e')
    parser.add_argument('--output_dir', default='../experiment_cp_final_v2_mels_batch16_lr_g39')
    #parser.add_argument('--input_wavs_dir', default='../input_wavs')
    #parser.add_argument('--output_dir', default='../experiment_crazy')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':

    #   Cambia el directorio de trabajo al directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()

