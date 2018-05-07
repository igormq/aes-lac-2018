import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self,
                 manifest_filepath,
                 transforms=None,
                 target_transforms=None):

        with open(manifest_filepath) as f:
            data = f.readlines()
        self.data = [x.strip().split(',') for x in data]

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
        def apply(self, batch):

            longest_sample = max(batch, key=lambda p: p[0].size(1))[0]
            freq_size = longest_sample.size(0)
            minibatch_size = len(batch)
            max_seqlength = longest_sample.size(1)
            inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
            input_percentages = torch.FloatTensor(minibatch_size)
            target_sizes = torch.IntTensor(minibatch_size)

            targets = []
            for x in range(minibatch_size):
                input, target = batch[x]

                seq_length = input.size(1)
                inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
                input_percentages[x] = seq_length / float(max_seqlength)
                target_sizes[x] = len(target)
                targets.extend(target)
            targets = torch.IntTensor(targets)

            return inputs, targets, input_percentages, target_sizes

        return apply