""" AN4 dataset handler
"""
import os
import subprocess

from . import utils
from .corpus import Corpus


class AN4(Corpus):

    DATASET_URLS = {
        "train": [
            "http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz"
        ],
        "test": [
            "http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz"
        ]
    }

    def __init__(self,
                 target_dir='an4_dataset',
                 min_duration=1,
                 max_duration=15,
                 fs=16000,
                 name='an4'):
        super().__init__(
            AN4.DATASET_URLS,
            target_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            fs=fs,
            name=name)

    def process_audio(self, audio_path, wav_path):
        cmd = 'sox -t raw -r {} -b 16 -e signed-integer -B -c 1 "{}" "{}"'.format(
            self.fs, audio_path, wav_path)

        ret_code = subprocess.call(cmd, shell=True)

        if ret_code < 0:
            raise RuntimeError(
                'sox was terminated by signal {}'.format(ret_code))

    def get_data(self, root_dir, set_type):
        root_dir = os.path.join(root_dir, 'an4')
        wav_path = os.path.join(root_dir, 'wav')
        file_ids_path = os.path.join(root_dir,
                                     'etc/an4_{}.fileids'.format(set_type))
        transcripts_path = os.path.join(
            root_dir, 'etc/an4_{}.transcription'.format(set_type))

        data = []
        with open(file_ids_path, 'r') as f, open(transcripts_path, 'r') as t:

            for audio_path, transcript in zip(f, t):
                audio_path = os.path.join(wav_path, '{}.raw'.format(
                    audio_path.strip()))
                transcript = transcript.strip()

                data.append((audio_path, transcript))

        return data

    def process_transcript(self, root_dir, transcript_path, audio_path):
        return transcript_path.split('(')[0].strip("<s>").split('<')[
            0].strip().upper()


if __name__ == "__main__":
    parser = utils.get_argparse(
        os.path.join(os.path.split(os.path.abspath(__file__))[0]))
    args = parser.parse_args()

    an4 = AN4(
        target_dir=args.target_dir,
        fs=args.fs,
        max_duration=args.max_duration,
        min_duration=args.min_duration)
    manifest_paths = an4.download(args.files_to_download)

    for manifest_path in manifest_paths:
        print('Manifest created at {}'.format(manifest_path))
