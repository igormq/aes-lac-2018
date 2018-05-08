import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self,
                 data_dir,
                 manifest_filepath,
                 transforms=None,
                 target_transforms=None):

        with open(manifest_filepath) as f:
            data = f.readlines()
        self.data = [[
            os.path.join(data_dir, path) for path in x.strip().split(',')[:2]
        ] for x in data]

        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):
        audio_path, transcript_path = self.data[index]

        input = audio_path
        if self.transforms is not None:
            input = self.transforms(audio_path)

        target = transcript_path
        if self.target_transforms is not None:
            target = self.target_transforms(transcript_path)

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
            input_percentages = torch.FloatTensor(minibatch_size)
            target_sizes = torch.IntTensor(minibatch_size)

            targets = []
            for i in range(minibatch_size):
                input, target = batch[i]
                curr_seq_length = input.shape[0]

                inputs[i, :curr_seq_length, :].copy_(input)
                input_percentages[i] = curr_seq_length / float(max_seq_length)

                target_sizes[i] = len(target)
                targets.extend(target)

            targets = torch.IntTensor(targets).squeeze()

            return inputs, targets, input_percentages, target_sizes

        return apply