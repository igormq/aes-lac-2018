""" Sid dataset handler
"""
import os
import re

import utils
from corpus import Corpus


class Sid(Corpus):

    DATASET_URLS = {
        "train": ["https://www.dropbox.com/s/0wxlweatglrr7wl/sid.tar.gz?dl=1"]
        # "train": ["https://www.dropbox.com/s/wjq6csm8a5nn7ux/tiny_sid.tar.gz?dl=1"]
    }

    def __init__(self,
                 target_dir='sid_dataset',
                 min_duration=1,
                 max_duration=15,
                 fs=16000,
                 suffix='sid'):
        super().__init__(
            Sid.DATASET_URLS,
            target_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            fs=fs,
            suffix=suffix)

    def get_data(self, root_dir, set_type):
        audio_paths = list(self.find_audios(root_dir))

        search_dirs = set([os.path.split(p)[0] for p in audio_paths])

        data = []
        for curr_dir in search_dirs:
            transcriptions_file = [
                filename for filename in os.listdir(curr_dir)
                if re.search(r'prompts\.txt', filename, re.IGNORECASE)
            ][0]

            assert os.path.isfile(
                (transcriptions_file
                 )), "prompts.txt not found in {}".format(root_dir)

            with open(transcriptions_file, 'r', encoding='utf8') as f:
                for line in f.readlines():
                    curr_id, transcript = line.strip().split('=')

                    pattern = re.compile(r'{}{}[FM][0-9]{{4}}{}'.format(
                        curr_dir, os.sep, curr_id))

                    filtered_audio_paths = list(
                        filter(pattern.match, audio_paths))

                    assert ~len(
                        filtered_audio_paths
                    ), 'Found more than one audio file for the transcription id {} in {}'.format(
                        curr_id, transcriptions_file)

                    audio_path = audio_paths.pop(
                        audio_paths.index(filtered_audio_paths[0]))

                    data.append((audio_path, transcript))

        assert ~len(audio_paths), "Some transcriptions were not found"

        return data

    def process_transcript(self, root_dir, transcript_path, audio_path):
        return transcript_path


if __name__ == "__main__":
    parser = utils.get_argparse('sid_dataset')
    args = parser.parse_args()

    sid = Sid(
        target_dir=args.target_dir,
        fs=args.fs,
        max_duration=args.max_duration,
        min_duration=args.min_duration)
    manifest_paths = sid.download(args.files_to_download)

    for manifest_path in manifest_paths:
        print('Manifest created at {}'.format(manifest_path))
