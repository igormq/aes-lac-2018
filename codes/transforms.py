import os
import librosa
import torch
import subprocess
import torchaudio
import numpy as np

from tempfile import NamedTemporaryFile


class ToSpectrogram(object):
    """Create a spectrogram from a raw audio signal

    Args:
        frame_length (int): window size, often called the fft size as well
        hop (int, optional): length of hop between STFT windows. default: ws // 2
        fft_size (int, optional): number of fft bins. default: ws // 2 + 1
        pad_end (int): indicates the amount of zero padding at the end of
            `signal` before STFT
        normalize (bool): Apply standard mean and deviation normalization to spectrogram
        window (torch windowing function): default: torch.hann_window
        window_kwargs (dict, optional): arguments for window function
    """

    def __init__(self,
                 frame_length=320,
                 hop=160,
                 fft_size=512,
                 pad_end=0,
                 normalize=False,
                 window=torch.hann_window,
                 window_kwargs=None):
        if isinstance(window, torch.Tensor):
            self.window = window
        else:
            self.window = window(
                frame_length) if window_kwargs is None else window(
                    frame_length, **window_kwargs)
            self.window = torch.Tensor(self.window)

        self.frame_length = frame_length
        self.hop = hop if hop is not None else frame_length // 2
        self.fft_size = fft_size  # number of fft bins
        self.normalize = normalize
        self.pad_end = pad_end
        self.window_kwargs = window_kwargs

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

        assert x.dim() == 2 and isinstance(x, torch.Tensor)

        S = torch.stft(x, self.frame_length, self.hop, self.fft_size, True,
                       True, self.window, self.pad_end)  # (c, l, fft_size, 2)

        S /= self.window.pow(2).sum().sqrt()
        S = S.pow(2).sum(-1)  # get power of "complex" tensor (c, l, fft_size)

        S = torch.log1p(S)  # to log scale

        if self.normalize:
            S = (S - S.mean()) / (S.std() + eps)

        return S


class ToTensor(object):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.

    Args:
        sample_rate (int): the desired sample rate
        tempo_prob (float): probability of tempo jitter being applied to sample
        tempo_range (list, int): list with (tempo_min, tempo_max)
        gain_prob (float): probability of gain being added to sample
        gain_range (float): gain level in dB
    Returns:
        the augmented utterance.
    """

    def __init__(self,
                 sample_rate=16000,
                 tempo_prob=1,
                 tempo_range=(0.85, 1.15),
                 gain_prob=1,
                 gain_range=(-6, 8)):
        self.sample_rate = sample_rate

        self.tempo_prob = tempo_prob
        self.tempo_range = tempo_range

        self.gain_prob = gain_prob
        self.gain_range = gain_range

    def __call__(self, x):

        tempo_value = 1
        if np.random.binomial(1, self.tempo_prob):
            low_tempo, high_tempo = self.tempo_range
            tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)

        gain_value = 0
        if np.random.binomial(1, self.gain_prob):
            low_gain, high_gain = self.gain_range
            gain_value = np.random.uniform(low=low_gain, high=high_gain)

        audio = self._augment_audio_with_sox(
            path=x,
            sample_rate=self.sample_rate,
            tempo=tempo_value,
            gain=gain_value)

        return audio

    def _augment_audio_with_sox(self, path, sample_rate, tempo, gain):
        """
        Changes tempo and gain of the recording with sox and loads it.
        """
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

            return y


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

            return torchaudio.DownmixMono()(y)