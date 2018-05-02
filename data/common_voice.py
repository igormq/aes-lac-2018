""" Common voice dataset handler
"""
import os

import pandas as pd

from .corpus import Corpus

from . import utils


class CommonVoice(Corpus):

    DATASET_URLS = {
        "train": [
            "https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz"
        ],
        "val": [
            "https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz"
        ],
        "test": [
            "https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz"
        ]
    }

    FILES_TO_PROCESS = {
        "train": "cv-valid-train.csv",
        "val": "cv-valid-dev.csv",
        "test": "cv-valid-test.csv",
    }

    def __init__(self,
                 target_dir='common_voice_dataset',
                 min_duration=1,
                 max_duration=15,
                 fs=16000,
                 name='common_voice'):
        super().__init__(
            CommonVoice.DATASET_URLS,
            target_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            fs=fs,
            name=name)

    def get_data(self, root_dir, set_type):

        root_dir = os.path.join(root_dir, 'cv_corpus_v1')
        csv_path = os.path.join(root_dir,
                                CommonVoice.FILES_TO_PROCESS[set_type])

        data = []
        with pd.read_csv(csv_path) as df:
            for _, row in df.iterrows():
                file_path = os.path.join(root_dir, row['filename'])
                data.append(file_path, row['text'])
        return data

    def process_transcript(self, root_dir, transcript_path, audio_path):
        return transcript_path.strip().upper()


if __name__ == "__main__":
    parser = utils.get_argparse(
        os.path.join(os.path.split(os.path.abspath(__file__))[0]))
    args = parser.parse_args()

    common_voice = CommonVoice(
        target_dir=args.target_dir,
        fs=args.fs,
        max_duration=args.max_duration,
        min_duration=args.min_duration)
    manifest_paths = common_voice.download(args.files_to_download)

    for manifest_path in manifest_paths:
        print('Manifest created at {}'.format(manifest_path))
