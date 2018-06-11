""" LapsBM dataset handler
"""
import os

from . import utils
from .corpus import Corpus


class LapsBM(Corpus):
    """ LapsBM class
    """

    DATASET_URLS = {
        "val": ["https://www.dropbox.com/s/c1cieo0u0earx69/lapsbm-val.tar.gz?dl=1"],
        "test": ["https://www.dropbox.com/s/ssbjagkynylldvd/lapsbm-test.tar.gz?dl=1"],
    }

    def __init__(self, target_dir='lapsbm_dataset', min_duration=1, max_duration=15, fs=16e3, name='lapsbm'):
        super().__init__(
            LapsBM.DATASET_URLS, target_dir, min_duration=min_duration, max_duration=max_duration, fs=fs, name=name)

    def process_transcript(self, root_dir, transcript_path, audio_path):
        path, _ = os.path.splitext(audio_path)

        transcription_file = path + '.txt'

        with open(transcription_file, 'r', encoding='utf8') as f:
            return f.readlines()[0].strip()

        raise ValueError('No transcription found for {}'.format(transcript_path))


if __name__ == "__main__":
    parser = utils.get_argparse(os.path.join(os.path.split(os.path.abspath(__file__))[0]))
    args = parser.parse_args()

    lapsbm = LapsBM(
        target_dir=args.target_dir, fs=args.fs, max_duration=args.max_duration, min_duration=args.min_duration)
    manifest_paths = lapsbm.download(args.files_to_download)

    for manifest_path in manifest_paths:
        print('Manifest created at {}'.format(manifest_path))
