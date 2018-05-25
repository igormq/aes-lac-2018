import io
import os
import re
import subprocess
import sys
from tempfile import NamedTemporaryFile

import numpy as np
import torch

import librosa
import soundfile as sf
import torchaudio
from codes.preprocessing import LabelBinarizer, OrderedLabelEncoder
from num2words import num2words
from torchaudio.transforms import *
from unidecode import unidecode

from .utils.io_utils import read_labels


class ToSpectrogram(object):
    """Create a spectrogram from a raw audio signal

    Args:
        frame_length (int): window size, often called the fft size as well
        hop (int, optional): length of hop between STFT windows. default: ws // 2
        fft_size (int, optional): number of fft bins. default: ws // 2 + 1
        pad_end (int): indicates the amount of zero padding at the end of
            `signal` before STFT
        normalize (bool): Apply standard mean and deviation normalization to spectrogram
        window (torch windowing function or str): default: torch.hann_window
        window_params (dict, optional): arguments for window function
        librosa_compat (bool): if `True`the stft will be librosa compatible
    """

    def __init__(self,
                 frame_length=320,
                 hop=160,
                 fft_size=None,
                 pad_end=0,
                 normalize=True,
                 window=torch.hann_window,
                 window_params={},
                 librosa_compat=False,
                 eps=1e-9):

        if librosa_compat:
            window_params.setdefault('periodic', False)

        if isinstance(window, torch.Tensor):
            self.window = window
        elif isinstance(window, str):
            self.window = getattr(torch.functional, '{}_window'.format(window))(frame_length,
                                                      **window_params)
        else:
            self.window = window(frame_length, **window_params)
            self.window = torch.tensor(self.window, dtype=torch.float)

        self.frame_length = frame_length
        self.hop = hop if hop is not None else frame_length // 2
        self.fft_size = fft_size or self.frame_length
        self.normalize = normalize
        self.pad_end = pad_end
        self.window_params = window_params
        self.eps = eps
        self.librosa_compat = librosa_compat

    def __call__(self, x):
        """
        Args:
            x (Tensor or Variable): Tensor of audio of size (C, N)
        Returns:
            spectogram (Tensor or Variable): channels x hops x fft_size (c, l, f), where channels
                is unchanged, hops is the number of hops, and fft_size is the
                number of fourier bins, which should be the window size divided
                by 2 plus 1.
        """

        assert x.dim() == 1 and isinstance(x, torch.Tensor)

        if not self.librosa_compat:
            S = torch.stft(x, self.frame_length, self.hop, self.fft_size, True,
                        True, self.window, self.pad_end)  # (c, l, fft_size, 2)

            # Get magnitude of "complex" tensor (C, L, N_FFT//2 + 1)
            S = S.pow(2).sum(-1).sqrt() / self.window.pow(2).sum().sqrt()
        else:
            # The original code is as follows:
            S = librosa.stft(x.numpy(), n_fft=self.fft_size, hop_length=self.hop,
                            win_length=self.frame_length, window=self.window.numpy()).transpose((1, 0))
            S, _ = librosa.magphase(S)
            S = torch.tensor(S, dtype=torch.float)

            # The following implementation does not have the same output as the above implementation!

            # centered mode
            # x = torch.from_numpy(np.pad(x, int(self.fft_size//2), mode='reflect'))
            # S = torch.stft(x, self.frame_length, self.hop, self.fft_size, False,
            #             True, self.window, self.pad_end)

            # S = S.pow(2).sum(-1).sqrt()

        S = torch.log1p(S)  # to log scale

        if self.normalize:
            S = (S - S.mean()) / (S.std() + self.eps)

        return S


class ToTensor(object):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.

    Args:
        sample_rate (int): the desired sample rate
        augment (bool): if `True`, will add random jitter and gain
        tempo_prob (float): probability of tempo jitter being applied to sample
        tempo_range (list, int): list with (tempo_min, tempo_max)
        gain_prob (float): probability of gain being added to sample
        gain_range (float): gain level in dB
    Returns:
        the augmented utterance.
    """

    def __init__(self,
                 sample_rate=16000,
                 augment=True,
                 tempo_range=(0.85, 1.15),
                 gain_range=(-6, 8)):
        self.sample_rate = sample_rate
        self.augment = augment

        self.tempo_range = tempo_range
        self.gain_range = gain_range

    def _load(self, path):
        if isinstance(path, bytes):
            data, sample_rate = sf.read(io.ByteIO(path))
            return torch.from_numpy(data).float(), sample_rate

        return torchaudio.load(path)

    def __call__(self, x):

        if not self.augment:
            y, sample_rate = self._load(x)
            assert sample_rate == self.sample_rate

            return y.squeeze()

        low_tempo, high_tempo = self.tempo_range
        tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)

        low_gain, high_gain = self.gain_range
        gain_value = np.random.uniform(low=low_gain, high=high_gain)

        audio = self._augment_audio_with_sox(
            x,
            sample_rate=self.sample_rate,
            tempo=tempo_value,
            gain=gain_value)

        return audio

    def _augment_audio_with_sox(self, x, sample_rate, tempo, gain):
        """
        Changes tempo and gain of the recording with sox and loads it.
        """
        if isinstance(x, bytes):
            with NamedTemporaryFile(suffix='.wav', delete=False) as original_file:
                path = original_file.name
                data, sample_rate = self._load(x)
                sf.write(original_file.name, data.numpy(), sample_rate)
        else:
            path = x

        with NamedTemporaryFile(suffix=".wav") as augmented_file:
            augmented_filename = augmented_file.name
            sox_augment_params = [
                "tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)
            ]
            sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(
                path, sample_rate, augmented_filename,
                " ".join(sox_augment_params))

            ret_code = subprocess.call(sox_params, shell=True)
            if ret_code < 0:
                raise RuntimeError(
                    'sox was terminated by signal {}'.format(ret_code))

            y, sample_rate = torchaudio.load(augmented_filename)
            assert sample_rate == self.sample_rate

            if isinstance(x, bytes):
                os.unlink(path)

            return y.squeeze()


class NoiseInjection(object):
    def __init__(self,
                 path,
                 sample_rate=16000,
                 noise_levels=(0, 0.5),
                 prob=0.4):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level,
        the more noise added.

        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py

        Args:
            path (str): where the noise audio files is located
            sample_rate (int): target sample rate
            noise_level(list, float): list or set with (noise_min, noise_max),
                where 1.0 means all noise, not original signal
            prob (float): Probability of noise being added per sample
        """

        if not os.path.exists(path):
            raise IOError("Directory does not exist: {}".format(path))

        self.paths = path is not None and librosa.util.find_files(path)
        self.sample_rate = sample_rate
        self.noise_levels = noise_levels

    def __call__(self, x):
        assert x.dim() == 1, 'Only mono audio is accepted'

        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)

        noise_len = self._get_audio_length(noise_path)
        signal_len = len(x) / self.sample_rate
        noise_start = torch.rand(()) * (noise_len - signal_len)
        noise_end = noise_start + signal_len
        noise = self._load_crop_and_resample(noise_path, self.sample_rate,
                                             noise_start, noise_end)
        assert len(x) == len(noise)

        noise_energy = (noise.dot(noise) / noise.size).sqrt()
        signal_energy = (x.dot(x) / x.size).sqrt()

        return x + noise_level * noise * signal_energy / noise_energy

    def _get_audio_length(self, path):
        output = subprocess.check_output(
            ['soxi -D \"%s\"' % path.strip()], shell=True)
        return float(output)

    def _load_crop_and_resample(self, path, sample_rate, start_time, end_time):
        """Crop and resample the recording with sox and loads it."""
        with NamedTemporaryFile(suffix=".wav") as tar_file:
            tar_filename = tar_file.name
            sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(
                path, sample_rate, tar_filename, start_time, end_time)
            os.system(sox_params)
            y, _ = torchaudio.load(tar_filename)

            return torchaudio.transforms.DownmixMono()(y)


class ToLabel(object):
    """ Parse transcript file or list of utterances into labels given a dictionary

    Args:
        labels(str or list): list of labels or string with the labels or path to a json containing
            the labels
        to_upper (bool): if `True`, characters are parsed to uppercase before transforming
        one_hot (bool): if `True`, the __call__ will return labels one-hot encoded
        convert_number_to_word (bool): if `True`, converts number to words
        lang (str): locale used to convert number to strings
        remove_accents (bool): if `True`, all accents are removed prior transform
    """

    GET_NUMBERS_PATTERN = r'-?\d+(?:\.|,)?\d*'

    def __init__(self,
                 labels='labels.en.json',
                 to_upper=True,
                 one_hot=False,
                 convert_number_to_words=True,
                 lang=None,
                 remove_accents=True,
                 dtype=np.int):


        if isinstance(labels, str):
            if os.path.isfile(labels):
                labels_list = read_labels(labels)
                lang = lang or labels.split('.')[-2]
            else:
                labels_list = list(labels)
        elif isinstance(labels, (list, set)):
            labels_list = labels
        else:
            raise ValueError('labels type was not recognized.')

        self._labels = labels
        self._lang = lang
        self._to_upper = to_upper
        self._dtype = dtype
        self._one_hot = one_hot
        self._convert_number_to_words = convert_number_to_words
        self._remove_accents = remove_accents



        if self._one_hot:
            self.label_encoder = LabelBinarizer(pos_label=0).fit(labels_list)
        else:
            self.label_encoder = OrderedLabelEncoder().fit(labels_list)

        self._output_size = len(
            self.label_encoder.classes_) if self._one_hot else 1

    def __call__(self, x):
        """
        Args:
            x (bytes, str, or file path): a file path containing the desired
                transcript to be converted or the string to be converted
        Returns:
            ndarray of size (size, num_labels) if `one_hot` is True else (size,)
        """
        if isinstance(x, str):
            if os.path.isfile(x):
                with open(x, 'r', encoding='utf8') as f:
                    transcript = f.readline().strip()
            else:
                transcript = x
        elif isinstance(x, bytes):
            transcript = x.decode('utf8')
        else:
            raise ValueError('input type was not recognized')

        if self._to_upper:
            transcript = transcript.upper()

        if self._convert_number_to_words:
            transcript = self._parse_numbers(transcript)

        if self._remove_accents:
            transcript = unidecode(transcript)

        # split into characters and filter not found
        transcript = self._split_and_filter(transcript)
        labels = np.asarray(
            self.label_encoder.transform(transcript), dtype=self._dtype)

        if self._one_hot:
            return labels

        return labels[:, np.newaxis]

    def _split_and_filter(self, transcript):
        return list(
            filter(lambda x: x in self.label_encoder.classes_, transcript))

    def _parse_numbers(self, transcript):
        return re.sub(
            ToLabel.GET_NUMBERS_PATTERN,
            lambda x: num2words(x.group().replace(',', '.'), lang=self._lang),
            transcript)

    def __repr__(self):
        repr_ = '{}(path={}, to_upper={}, one_hot={}, '.format(
            self.__class__.__name__, self._labels, self._to_upper,
            self._one_hot)
        repr_ += 'convert_number_to_words={}, lang={}, '.format(
            self._convert_number_to_words, self._lang)
        repr_ += 'remove_accents={}, dtype={})'.format(self._remove_accents,
                                                       self._dtype)

        return repr_


def parse(objs):
    """ Parse dict from json file in transform objects
    """
    transforms = objs.copy()
    if isinstance(objs, dict):
        transforms = [objs]

    for i, obj in enumerate(transforms):
        transforms[i] = getattr(sys.modules[__name__],
                                obj.transform)(**obj.params)

    if len(transforms) == 1:
        return transforms[0]

    return Compose(transforms)
