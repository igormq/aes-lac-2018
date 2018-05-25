import bisect
import io
import os
from zipfile import ZipFile

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


class AudioDataset(Dataset):
    def __init__(self,
                 data_dir,
                 manifest_filepath,
                 transforms=None,
                 target_transforms=None):
        self.manifest_filepath = manifest_filepath

        with open(self.manifest_filepath) as f:
            data = f.readlines()

        self.durations = [float(x.strip().split(',')[-1]) for x in data]

        self.data = [[
            path for path in x.strip().split(',')[:2]
        ] for x in data]

        self.is_zipped = False
        # Check if file exists in the system, otherwise look for a zipped file
        if not os.path.isfile(os.path.join(self.data[0][0])):
            if not os.path.isfile(self.manifest_filepath.replace('.csv', '.zip')):
                raise IOError('Data not found.')

            self.is_zipped = True
            self.zfile = ZipFile(self.manifest_filepath.replace('.csv', '.zip'))

        if not self.is_zipped:
            self.data = [[os.path.join(data_dir, path) for path in x] for x in self.data]

        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):
        audio_path, transcript_path = self.data[index]

        if self.is_zipped:
            with self.zfile.open(audio_path) as afile:
                input = afile.read()

            with self.zfile.open(transcript_path) as tfile:
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

class ConcatAudioDataset(ConcatDataset):
    """
    Dataset to concatenate multiple audio datasets

    Arguments:
        datasets (iterable): List of datasets to be concatenated
    """

    def __init__(self, datasets):
        super(ConcatAudioDataset, self).__init__(datasets)

        self._durations = [dataset.durations for dataset in self.datasets]
        self._durations = [d for durations in self.durations for d in durations]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        return super().__getitem__(idx), dataset_idx

    @property
    def durations(self):
        return self._durations



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

            if len(batch[0]) != 3:
                is_multi_task = False
                task = torch.tensor([0] * minibatch_size, dtype=torch.int)
            else:
                is_multi_task = True
                task = torch.tensor([b[-1] for b in batch], dtype=torch.int)

            data = []
            for t in set(task):
                task_size = (task == t).sum().item()
                task_batch = batch[task == t]

                longest_sample = max(task_batch, key=lambda x: x[0].shape[0])[0]
                max_seq_length, freq_size = longest_sample.shape

                inputs = torch.zeros(task_size, max_seq_length, freq_size)
                input_percentages = torch.zeros(task_size, dtype=torch.float)
                target_sizes = torch.zeros(task_size, dtype=torch.int)

                targets = []
                for i in range(task_size):
                    if len(task_batch[i]) == 2:
                        input, target = task_batch[i]
                        task = 0
                    else:
                        input, target, task = task_batch[i]

                    curr_seq_length = input.shape[0]

                    inputs[i, :curr_seq_length, :].copy_(input)
                    input_percentages[i] = curr_seq_length / float(max_seq_length)

                    target_sizes[i] = len(target)
                    targets.extend(target)

                targets = torch.tensor(targets, dtype=torch.int).squeeze()

                data.append(inputs, targets, input_percentages, target_sizes)

            if not is_multi_task:
                return data[0]

            return zip(*data)

        return apply
