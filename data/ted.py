""" TEDLium dataset handler
"""
import os
import re
import shutil
import subprocess
import unicodedata

from tqdm import tqdm

import utils
from corpus import Corpus


class TEDLIUM(Corpus):

    __version__ = 'v2'

    DATASET_URLS = {
        "train":
        ["http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz"],
        "val": ["http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz"],
        "test":
        ["http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz"]
    }

    def __init__(self,
                 target_dir='ted_dataset',
                 min_duration=1,
                 max_duration=15,
                 fs=16000,
                 suffix='ted'):
        super().__init__(
            TEDLIUM.DATASET_URLS,
            target_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            fs=fs,
            suffix=suffix)

    def process_audio(self, audio_path, wav_path):
        shutil.move(audio_path, wav_path)

    def get_data(self, root_dir, set_type):

        if set_type == 'val':
            set_type = 'dev'

        root_dir = os.path.join(root_dir, 'TEDLIUM_release2', set_type)

        data = []
        entries = os.listdir(os.path.join(root_dir, 'sph'))
        for sph_file in tqdm(entries, total=len(entries)):
            speaker_name = sph_file.split('.sph')[0]

            sph_path = os.path.join(root_dir, 'sph', sph_file)
            stm_path = os.path.join(root_dir, 'stm',
                                    '{}.stm'.format(speaker_name))

            assert os.path.exists(sph_path) and os.path.exists(stm_path)

            all_utterances = self._get_utterances_from_stm(stm_path)
            all_utterances = filter(self._filter_short_utterances,
                                    all_utterances)

            for utterance_id, utterance in enumerate(all_utterances):
                cut_audio_path = os.path.join(root_dir, 'sph',
                                              '{}_{}.wav'.format(
                                                  utterance['filename'],
                                                  str(utterance_id)))
                self._cut_utterance(sph_path, cut_audio_path,
                                    utterance['start_time'],
                                    utterance['end_time'])

                data.append((cut_audio_path, utterance['transcript']))

        return data

    def _get_utterances_from_stm(self, stm_file):
        """ Return list of entries containing phrase and its start/end timings
        """
        results = []
        with open(stm_file, 'r') as f:
            for stm_line in f:
                tokens = stm_line.split()
                start_time = float(tokens[3])
                end_time = float(tokens[4])
                filename = tokens[0]
                transcript = unicodedata.normalize('NFKD',
                                                ' '.join(t for t in tokens[6:]).strip()). \
                    encode('utf-8', 'ignore').decode('utf-8', 'ignore')
                if transcript != 'ignore_time_segment_in_scoring':
                    results.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'filename': filename,
                        'transcript': transcript
                    })
        return results

    def _filter_short_utterances(self, utterance_info, min_len_sec=1.0):
        return utterance_info["end_time"] - utterance_info["start_time"] > min_len_sec

    def process_transcript(self, root_dir, transcript_path, audio_path):
        return transcript_path

    def _cut_utterance(self, audio_path, target_audio_path, start_time, end_time):
        cmd = "sox {} -r {} -b 16 -c 1 {} trim {} = {}".format(
            audio_path, self.fs, target_audio_path, start_time, end_time)

        ret_code = subprocess.call(cmd, shell=True)

        if ret_code < 0:
            raise RuntimeError(
                'sox was terminated by signal {}'.format(ret_code))


if __name__ == "__main__":
    parser = utils.get_argparse('ted_dataset')
    args = parser.parse_args()

    ted = TEDLIUM(
        target_dir=args.target_dir,
        fs=args.fs,
        max_duration=args.max_duration,
        min_duration=args.min_duration)
    manifest_paths = ted.download(args.files_to_download)

    for manifest_path in manifest_paths:
        print('Manifest created at {}'.format(manifest_path))
