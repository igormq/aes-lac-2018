""" VoxForge PT-BR dataset handler
"""
import glob
import os
import re

from . import utils
from .corpus import Corpus


class VoxForge(Corpus):
    """ VoxForge class
    """

    DATASET_URLS = {
        "train": [
            "https://www.dropbox.com/s/wrguetal6xmrgta/voxforge-ptbr.tar.gz?dl=1"
        ]
    }

    def __init__(self,
                 target_dir='voxforge_dataset',
                 min_duration=1,
                 max_duration=15,
                 fs=16e3,
                 name='voxforge'):
        super().__init__(
            VoxForge.DATASET_URLS,
            target_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            fs=fs,
            name=name)

    def get_data(self, root_dir, set_type):
        audio_paths = list(self.find_audios(root_dir))

        search_dirs = set([
            os.path.split(p)[0].replace('{}wav'.format(os.path.sep), '')
            for p in audio_paths
        ])

        data = []
        for curr_dir in search_dirs:
            transcriptions_file = glob.glob(
                os.path.join(curr_dir, '**', 'PROMPTS'), recursive=True)[0]

            assert os.path.isfile(
                (transcriptions_file
                 )), "prompts.txt not found in {}".format(root_dir)

            with open(transcriptions_file, 'r', encoding='utf8') as f:
                for line in f.readlines():
                    path, transcript = line.strip().split(' ', maxsplit=1)

                    curr_id = path.split('/')[-1]
                    name = transcriptions_file.replace('{}etc'.format(os.sep), '').split(os.sep)[-2]
                    
                    pattern = re.compile(r'{}(?:[\\\/a-z]*){}(?:.[a-z]+)$'.format(
                        name, curr_id))

                    filtered_audio_paths = list(
                        filter(pattern.findall, audio_paths))
                    
                    if len(filtered_audio_paths) != 1:
                        print('WARNING: Found zero or more than one audio file for the transcription id {} in {}. Skipping...'.format(
                        curr_id, transcriptions_file))
                        continue

                    audio_path = audio_paths.pop(
                        audio_paths.index(filtered_audio_paths[0]))

                    data.append((audio_path, transcript))

        assert ~len(audio_paths), "Some transcriptions were not found"

        return data

    def process_transcript(self, root_dir, transcript_path, audio_path):
        return transcript_path


if __name__ == "__main__":
    parser = utils.get_argparse(
        os.path.join(os.path.split(os.path.abspath(__file__))[0]))
    args = parser.parse_args()

    voxforge = VoxForge(
        target_dir=args.target_dir,
        fs=args.fs,
        max_duration=args.max_duration,
        min_duration=args.min_duration)
    manifest_paths = voxforge.download(args.files_to_download)

    for manifest_path in manifest_paths:
        print('Manifest created at {}'.format(manifest_path))
