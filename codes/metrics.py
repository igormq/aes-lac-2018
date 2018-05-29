from ignite.exceptions import NotComputableError
from ignite.metrics import Metric, Loss
from warpctc_pytorch import CTCLoss as warp_CTCLoss

class ConcatMetrics(Metric):

    def __init__(self, metrics, output_transform=lambda x: x):
        self.metrics = metrics

    def update(self, outputs):
        if isinstance(outputs, dict):
            for i, output in outputs.items():
                self.metrics[i].update(output)
        else:
            for i, output in enumerate(outputs):
                self.metrics[i].update(output)

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def compute(self):
        num, den = 0, 0

        val_metrics = []
        for metric in self.metrics:
            num += metric.val
            den += metric.den
            val_metrics.append(metric.compute())

        val_metrics.append(num/den)

        if len(self.metrics) == 1:
            return val_metrics[0]

        return val_metrics

class CTCLoss(Loss):
    """
    Calculates the average CTC loss.
    """

    def __init__(self, output_transform=lambda x: x):
        super().__init__(warp_CTCLoss(), output_transform)

    def update(self, output):
        out, targets, out_sizes, target_sizes = output

        # CTC loss is batch_first = False, i.e., T x B x D
        out = out.transpose(0, 1)

        loss = self._loss_fn(out, targets, out_sizes, target_sizes).sum()

        assert len(loss.shape
                   ) == 0, '`CTCLoss` did not return the average loss'

        self._sum += loss.item()
        self._num_examples += out.shape[1]

    @property
    def val(self):
        return self._sum

    @property
    def den(self):
        return self._num_examples


class EditDistance(Metric):
    """
    Calculates the Word Error Rate (WER)

    `update` must receive output of the form (out, targets, out_sizes, target_sizes), where:
        `out` must be tensor of shape (batch_size, max_sequence_length, features)
        `targets` must be 1D integer tensor
        `out_sizes` must be 1D tensor with the actual size of each batch in `out`
        `target_sizes` must be 1D tensor with the size o each target sequence

    Args:
        decoder (Decoder): decoder object.
        distance_fn (callable): function with signature (x, y) that calculates the distance beetween x and y.
        normalization_fn (callable): function with signature (edit_distance, reference) that calculates the
            normalization between the distance value and the reference argument.
        output_transform (callable): a callable that is used to transform the
            model's output into the form expected by the metric. This can be
            useful if, for example, you have a multi-output model and you want to
            compute the metric with respect to one of the outputs.
        stateful (bool): if `True` the edit distance will be globally calculated (i.e., total sum/total den).
    """

    def __init__(self,
                 decoder,
                 distance_fn,
                 normalization_fn,
                 output_transform=lambda x: x,
                 stateful=False):
        self._decoder = decoder
        self._distance_fn = distance_fn
        self._normalization_fn = normalization_fn
        self._stateful = stateful

        super().__init__(output_transform)

    def reset(self):
        self._total_edit_distance = 0
        self._num_examples = 0

    def update(self, output):
        out, targets, out_sizes, target_sizes = output

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        decoded_output, _ = self._decoder.decode(out, out_sizes)
        target_strings = self._decoder.convert_to_strings(split_targets)

        for idx in range(len(target_strings)):
            transcript = decoded_output[idx][0]
            reference = target_strings[idx][0]

            edit_distance = self._distance_fn(transcript, reference)

            if not self._stateful:
                self._total_edit_distance += self._normalization_fn(
                    edit_distance, reference)
            else:
                self._total_edit_distance += edit_distance
                self._num_examples += self._normalization_fn(
                    edit_distance, reference)

        if not self._stateful:
            self._num_examples += out.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'WER must have at least one example before it can be computed')
        return (self._total_edit_distance / self._num_examples) * 100


    @property
    def val(self):
        return self._total_edit_distance

    @property
    def den(self):
        return self._num_examples


class WER(EditDistance):
    def __init__(self, decoder, output_transform=lambda x: x, stateful=False):
        def normalize(x, y):
            den = len(y.split())
            if not den:
                return x
            return x / den

        super().__init__(decoder, decoder.wer, normalize,
                         output_transform, stateful)


class CER(EditDistance):
    def __init__(self, decoder, output_transform=lambda x: x, stateful=False):
        def normalize(x, y):
            den = len(y)
            if not den:
                return x
            return x / den

        super().__init__(decoder, decoder.cer, normalize,
                         output_transform, stateful)
