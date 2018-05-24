import io
import os
from zipfile import ZipFile

import torch
from torch.utils.data import DataLoader, Dataset


class AudioDataset(Dataset):
    def __init__(self,
                 data_dir,
                 manifest_filepath,
                 transforms=None,
                 target_transforms=None):

        with open(manifest_filepath) as f:
            data = f.readlines()

        self.data_dir = data_dir
        if os.path.isdir(data_dir):
            self.is_zipped = False
        elif data_dir.endswith('.zip'):
            self.is_zipped = True
        else:
            raise ValueError('data_dir is not a directory nor a zip file')

        self.data = [[
            path for path in x.strip().split(',')[:2]
        ] for x in data]

        if not self.is_zipped:
            self.data = [[os.path.join(data_dir, path) for path in x] for x in self.data]

        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):
        audio_path, transcript_path = self.data[index]

        if self.is_zipped:
            with ZipFile(self.data_dir) as zfile:
                with zfile.open(audio_path) as afile:
                    input = afile.read()

                with zfile.open(transcript_path) as tfile:
                    target = tfile.read()
        else:
            input = audio_path
            target = transcript_path

        if self.transforms is not None:
            input = self.transforms(input)

        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return input, target

    def __len__(self):
        return len(self.data)


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn()

    def _collate_fn(self, batch_first=False, sort=False, pack=False):
        def apply(batch):

            minibatch_size = len(batch)

            longest_sample = max(batch, key=lambda x: x[0].shape[0])[0]
            max_seq_length, freq_size = longest_sample.shape

            inputs = torch.zeros(minibatch_size, max_seq_length, freq_size)
            input_percentages = torch.zeros(minibatch_size, dtype=torch.float)
            target_sizes = torch.zeros(minibatch_size, dtype=torch.int)

            targets = []
            for i in range(minibatch_size):
                input, target = batch[i]
                curr_seq_length = input.shape[0]

                inputs[i, :curr_seq_length, :].copy_(input)
                input_percentages[i] = curr_seq_length / float(max_seq_length)

                target_sizes[i] = len(target)
                targets.extend(target)

            targets = torch.tensor(targets, dtype=torch.int).squeeze()

            return inputs, targets, input_percentages, target_sizes

        return apply
