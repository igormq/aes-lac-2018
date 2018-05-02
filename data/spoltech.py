""" Spoltech Brazilian Portuguese dataset handler
"""
import os

from . import utils
from .corpus import Corpus


class CSLUSpoltech(Corpus):
    """ CSLU Spoltech PT-BR class
    """

    DATASET_URLS = {
        "train": [None],
    }

    def __init__(self,
                 target_dir='spoltech_dataset',
                 min_duration=1,
                 max_duration=15,
                 fs=16e3,
                 suffix='spoltech'):
        super().__init__(
            CSLUSpoltech.DATASET_URLS,
            target_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            fs=fs,
            suffix=suffix)

    def process_transcript(self, root_dir, transcript_path, audio_path):
        path, _ = os.path.splitext(audio_path)

        transcription_file = path.replace('speech', 'trans') + '.txt'

        with open(transcription_file, 'r', encoding='iso-8859-1') as f:
            text = f.readlines()[0].strip()
            return text

        raise ValueError('No transcription found for {}'.format(audio_path))


if __name__ == "__main__":
    parser = utils.get_argparse('spoltech_dataset')
    args = parser.parse_args()

    spoltech = CSLUSpoltech(
        target_dir=args.target_dir,
        fs=args.fs,
        max_duration=args.max_duration,
        min_duration=args.min_duration)
    manifest_paths = spoltech.download()

    for manifest_path in manifest_paths:
        print('Manifest created at {}'.format(manifest_path))
