""" Corpus handler
"""
import glob as glob
import logging
import os
import shutil
import tarfile
import time
from collections import OrderedDict
from multiprocessing import Pool

import librosa
import pandas as pd
import sox
import wget
from tqdm import tqdm

logging.getLogger('sox').setLevel(logging.ERROR)


class Corpus(object):
    """ Corpus Class
        If dataset does not exist, it will be downloaded
    """

    def __init__(self, download_urls, data_dir, min_duration=1, max_duration=15, fs=16e3, name='', num_jobs=4):

        self._num_jobs = num_jobs
        self.fs = fs
        self.min_duration = min_duration
        self.max_duration = max_duration

        # Initialize sox
        self.sox = sox.Transformer()
        self.sox.set_globals(verbosity=0)
        self.sox = self.sox.convert(samplerate=self.fs, n_channels=1)

        self.download_urls = download_urls
        self.data_dir = data_dir

        self.name = name

    def download(self, files_to_download=None, remove_extracted=False):
        start = time.time()

        dataset_dir = os.path.join(self.data_dir, '{}_dataset'.format(self.name))

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # dataset urls is a dict with keys (train, valid, test)
        # with values a list of files to be downloaded
        manifest_paths = []
        for set_type, urls in self.download_urls.items():
            set_dir = os.path.join(dataset_dir, set_type)

            if not os.path.exists(set_dir):
                os.makedirs(set_dir)

            set_wav_dir = os.path.join(set_dir, 'wav')

            if not os.path.exists(set_wav_dir):
                os.makedirs(set_wav_dir)

            set_txt_dir = os.path.join(set_dir, 'txt')

            if not os.path.exists(set_txt_dir):
                os.makedirs(set_txt_dir)

            downloads_dir = os.path.join(dataset_dir, "downloads")

            if not os.path.exists(downloads_dir):
                os.makedirs(downloads_dir)

            extracted_dir = os.path.join(downloads_dir, "extracted")

            for url in urls:

                if url is not None:
                    # check if we want to download this file
                    if files_to_download:
                        for f in files_to_download:
                            if url.find(f) != -1:
                                break
                        else:
                            print("Skipping url: {}".format(url))
                            continue

                    fname = wget.detect_filename(url)
                    name = os.path.splitext(fname)[0]
                    target_fname = os.path.join(downloads_dir, fname)

                    curr_extracted_dir = os.path.join(extracted_dir, name)

                    print('Downloading {}...'.format(fname))

                    if not os.path.exists(target_fname):
                        wget.download(url, target_fname)

                    print("Unpacking {}...".format(fname))

                    if not os.path.exists(curr_extracted_dir):
                        tar = tarfile.open(target_fname)
                        tar.extractall(curr_extracted_dir)
                        tar.close()

                    assert os.path.exists(extracted_dir), "Archive {} was not properly uncompressed.".format(fname)

                else:
                    print('No URL found. Skipping download.')
                    curr_extracted_dir = extracted_dir

                    assert os.path.exists(extracted_dir), 'No folder found in {}'.format(extracted_dir)

                print("Converting and extracting transcripts...")

                self._wav_txt_split(curr_extracted_dir, set_wav_dir, set_txt_dir, set_type)

                if remove_extracted:
                    shutil.rmtree(curr_extracted_dir)

            manifest_paths.append(
                self._create_manifest(
                    set_wav_dir,
                    os.path.join(self.data_dir, '{}.{}.csv'.format(self.name, set_type)),
                    prune=set_type.startswith('train')))

        print("Done. Time elapsed {:.2f}s".format(time.time() - start))

        return manifest_paths

    def _create_manifest(self, data_path, name, ordered=True, prune=False):
        manifest_path = '{}'.format(name)

        print('Looking for files...')
        file_paths = glob.glob(os.path.join(data_path, '*.wav'))

        print('Saving {}...'.format(manifest_path))
        wav_files, transcripts_files, durations = [], [], []
        for wav_path in tqdm(file_paths, total=len(file_paths), desc='Get durations'):
            transcript_path = wav_path.replace('{0}wav{0}'.format(os.path.sep),
                                               '{0}txt{0}'.format(os.path.sep)).replace('.wav', '.txt')
            wav_files.append(os.path.relpath(wav_path, self.data_dir))
            transcripts_files.append(os.path.relpath(transcript_path, self.data_dir))
            durations.append(librosa.core.get_duration(filename=wav_path, sr=self.fs))

        df = pd.DataFrame(OrderedDict(audio_paths=wav_files, transcriptions=transcripts_files, durations=durations))

        if ordered:
            print('Sorting files by length...')
            df = df.sort_values(by=['durations'])

        if prune and (self.min_duration and self.max_duration):
            print("Pruning manifests between {} and {} seconds. ".format(self.min_duration, self.max_duration), end='')
            df = df[(df['durations'] >= self.min_duration) & (df['durations'] <= self.max_duration)]
            print('Total pruned: {} files'.format(len(file_paths) - len(df)))

        df.to_csv(manifest_path, index=False, header=False, columns=['audio_paths', 'transcriptions', 'durations'])

        return os.path.abspath(manifest_path)

    def _wav_txt_split(self, files_dir, wav_dir, txt_dir, set_type):

        file_paths = self.get_data(files_dir, set_type)

        pool = Pool(self._num_jobs)
        with tqdm(total=len(file_paths)) as pbar:
            for (audio_path, transcript_path) in file_paths:
                pool.apply_async(
                    self.process_data,
                    args=(files_dir, audio_path, transcript_path, wav_dir, txt_dir),
                    callback=lambda args: pbar.update())
            pool.close()
            pool.join()

    def find_audios(self, root_dir):
        return set(librosa.util.find_files(root_dir, recurse=True))

    def get_data(self, root_dir, set_type):
        """ Returns a list of tuple containing (audio_path, transcription)

        Note: all paths are relative to root_dir
        """
        audio_paths = self.find_audios(root_dir)
        return list(zip(audio_paths, [None] * len(audio_paths)))

    def process_data(self, root_dir, audio_path, transcript_path, wav_dir, txt_dir):

        assert os.path.exists(root_dir) and os.path.exists(audio_path)

        unique_name = os.path.splitext(os.path.relpath(audio_path, start=root_dir))[0].replace(os.sep, '-')

        # process transcript
        try:
            txt_transcript_path = os.path.join(txt_dir, '{}.txt'.format(unique_name))
            transcription = self._preprocess_transcript(self.process_transcript(root_dir, transcript_path, audio_path))

        except ValueError:
            return

        # Saving wave file
        wav_path = os.path.join(wav_dir, '{}.wav'.format(unique_name))
        self.process_audio(audio_path, wav_path)

        # Saving transcription
        with open(txt_transcript_path, "w", encoding='utf8') as f:
            f.write(transcription)
            f.flush()

    def process_audio(self, audio_path, wav_path):
        """ Converts audio to wav
        """
        self.sox.build(audio_path, wav_path)

    def _preprocess_transcript(self, phrase):
        return phrase.strip().upper()

    def process_transcript(self, root_dir, transcript_path, audio_path):
        raise NotImplementedError('process_transcript must be overwritten.')
