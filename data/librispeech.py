""" Librispeech dataset handler
"""
import glob
import os

from . import utils
from .corpus import Corpus


class LibriSpeech(Corpus):

    DATASET_URLS = {
        "train": [
            "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
            "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
            "http://www.openslr.org/resources/12/train-other-500.tar.gz"
        ],
        "val": [
            "http://www.openslr.org/resources/12/dev-clean.tar.gz",
            "http://www.openslr.org/resources/12/dev-other.tar.gz"
        ],
        "test_clean":
        ["http://www.openslr.org/resources/12/test-clean.tar.gz"],
        "test_other":
        ["http://www.openslr.org/resources/12/test-other.tar.gz"]
    }

    def __init__(self,
                 target_dir='librispeech_dataset',
                 min_duration=1,
                 max_duration=15,
                 fs=16000,
                 name='librispeech'):
        super().__init__(
            LibriSpeech.DATASET_URLS,
            target_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            fs=fs,
            name=name)

    def get_data(self, root_dir, set_type):

        data = []
        for audio_path in glob.iglob(
                os.path.join(root_dir, '**', '*.flac'), recursive=True):
            transcript_path = os.path.join(
                "-".join(audio_path.split('-')[:-1]) + ".trans.txt")

            key = os.path.basename(audio_path).replace(".flac",
                                                       "").split("-")[-1]

            with open(transcript_path, 'r') as f:
                transcriptions = f.read().strip().split("\n")
                transcriptions = {
                    t.split()[0].split("-")[-1]: " ".join(t.split()[1:])
                    for t in transcriptions
                }

                assert key in transcriptions, "{} is not in the transcriptions".format(
                    key)

                data.append((audio_path, transcriptions[key]))

        return data

    def process_transcript(self, root_dir, transcript_path, audio_path):
        return transcript_path.strip().upper()


if __name__ == "__main__":
    parser = utils.get_argparse(
        os.path.join(os.path.split(os.path.abspath(__file__))[0]))
    args = parser.parse_args()

    libri_speech = LibriSpeech(
        target_dir=args.target_dir,
        fs=args.fs,
        max_duration=args.max_duration,
        min_duration=args.min_duration)
    manifest_paths = libri_speech.download(args.files_to_download)

    for manifest_path in manifest_paths:
        print('Manifest created at {}'.format(manifest_path))
