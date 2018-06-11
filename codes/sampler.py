import math

import numpy as np
import torch
from torch.distributed import get_rank, get_world_size
from torch.utils.data.sampler import Sampler


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        # deterministically shuffle based on epoch
        np.random.seed(epoch)
        np.random.shuffle(self.bins)


class WeightedBucketingRandomSampler(Sampler):
    def __init__(self, data_source, batch_size=1, sampling='equal', num_epochs=None):
        self.data_source = data_source
        self.durations = self.data_source.durations
        self.batch_size = batch_size
        self.sampling = sampling
        self.num_epochs = num_epochs
        self.tasks_count = [
            j - i for i, j in zip(([0] + self.data_source.cumulative_sizes[:-1]), self.data_source.cumulative_sizes)
        ]

        self.bins = self.draw_bins()

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def draw_bins(self, epoch=0):
        replacement = True
        if self.sampling == 'equal':
            nested_weights = [[count] * count for count in self.tasks_count]
            weights = torch.tensor(
                [sum(self.tasks_count) / w for weights in nested_weights for w in weights], dtype=torch.double)
        elif self.sampling == 'unbalanced':
            replacement = False
            weights = torch.tensor([1] * len(self.data_source), dtype=torch.double)
        elif self.sampling == 'schedule':
            if len(self.tasks_count) != 2:
                raise ValueError('number of dataset should be 2')

            if self.num_epochs is None:
                raise ValueError('num_epochs must be set')

            prob = (self.num_epochs - epoch) / self.num_epochs
            probs = [prob, 1 - prob]

            orig_nested_weights = [[count] * count for count in self.tasks_count]
            nested_weights = [[probs[i]] * count for i, count in enumerate(self.tasks_count)]
            weights = torch.tensor(
                [
                    1 / o * w for orig, weights in zip(orig_nested_weights, nested_weights)
                    for o, w in zip(orig, weights)
                ],
                dtype=torch.double)
        else:
            raise ValueError('sampling option not recognized.')

        torch.manual_seed(epoch)
        ids = torch.multinomial(weights, len(self.data_source), replacement)
        ids_durations = [self.durations[id] for id in ids]

        sorted_idxs = np.argsort(ids_durations)
        ids = ids[sorted_idxs]

        bins = [ids[i:i + self.batch_size] for i in range(0, len(ids), self.batch_size)]

        return bins

    def shuffle(self, epoch):
        # deterministically shuffle based on epoch
        self.bins = self.draw_bins(epoch)


class DistributedBucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1, num_replicas=None, rank=None):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(DistributedBucketingSampler, self).__init__(data_source)
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.data_source = data_source
        self.ids = list(range(0, len(data_source)))
        self.batch_size = batch_size
        self.bins = [self.ids[i:i + batch_size] for i in range(0, len(self.ids), batch_size)]
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.bins) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        offset = self.rank
        # add extra samples to make it evenly divisible
        bins = self.bins + self.bins[:(self.total_size - len(self.bins))]
        assert len(bins) == self.total_size
        samples = bins[offset::self.num_replicas]  # Get every Nth bin, starting from rank
        return iter(samples)

    def __len__(self):
        return self.num_samples

    def shuffle(self, epoch):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(epoch)
        bin_ids = list(torch.randperm(len(self.bins), generator=g))
        self.bins = [self.bins[i] for i in bin_ids]
