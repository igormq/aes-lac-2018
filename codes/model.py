import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SequenceWise(nn.Module):
    """
        Collapses input of dim (sequences x batch_size x num_features) to
        (sequences * batch_size) x num_features, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.

        Args:
            module: Module to apply input to.
    """

    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        T, B = x.shape[:2]

        x = x.view(T * B, -1)
        x = self.module(x)
        x = x.view(T, B, -1)

        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class BatchRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 rnn_type=nn.LSTM,
                 bidirectional=False,
                 batch_norm=True):
        super(BatchRNN, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._bidirectional = bidirectional

        self.batch_norm = SequenceWise(nn.BatchNorm1d(
            self._input_size)) if batch_norm else None
        self.rnn = rnn_type(
            input_size=self._input_size,
            hidden_size=self._hidden_size,
            bidirectional=self._bidirectional,
            bias=False)

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)

        if self._bidirectional:
            # T x B x (H*2) -> T x B x H
            T, B = x.shape[:2]
            x = x.view(T, B, 2, -1).sum(2).view(T, B, -1)

        return x


class Lookahead(nn.Module):
    """ Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks

    Input shape is a ndarray with shape (sequence, batch, feature)
    """

    def __init__(self, num_features, context):
        super(Lookahead, self).__init__()

        assert context > 0, "Context should be greater than 0"

        self.num_features = num_features
        self.context = context

        self.weight = Parameter(torch.Tensor(num_features, context + 1))
        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        self.weight.uniform_(-stdv, stdv)

    def forward(self, input):
        seq_len = input.shape[0]

        x = F.pad(input, (0, 0, 0, 0, 0, self.context))

        # add lookahead windows (with context+1 width) as a fourth dimension
        # for each seq-batch-feature combination
        # TxLxNxH - sequence, context, batch, feature
        x = [x[i:i + self.context + 1] for i in range(seq_len)]
        x = torch.stack(x)

        # TxNxHxL - sequence, batch, feature, context
        x = x.permute(0, 2, 3, 1)

        x = torch.mul(x, self.weight).sum(dim=3)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', context=' + str(self.context) + ')'


class DeepSpeech(nn.Module):
    __version__ = '0.0.1'

    def __init__(self,
                 rnn_type=nn.LSTM,
                 num_classes=29,
                 rnn_hidden_size=768,
                 num_rnn_layers=5,
                 window_size=320,
                 bidirectional=True,
                 context=20):
        super(DeepSpeech, self).__init__()

        if isinstance(rnn_type, str):
            rnn_type = getattr(torch.nn, rnn_type.upper())

        self._rnn_type = rnn_type
        self._num_classes = num_classes
        self._rnn_hidden_size = rnn_hidden_size
        self._num_rnn_layers = num_rnn_layers
        self._window_size = window_size
        self._bidirectional = bidirectional
        self._context = context

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(0, 10)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True))

        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((self._window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size - 21) / 2 + 1)
        rnn_input_size *= 32

        rnns = []
        rnn = BatchRNN(
            input_size=rnn_input_size,
            hidden_size=self._rnn_hidden_size,
            rnn_type=self._rnn_type,
            bidirectional=self._bidirectional,
            batch_norm=False)
        rnns.append(('0', rnn))

        for x in range(self._num_rnn_layers - 1):
            rnn = BatchRNN(
                input_size=self._rnn_hidden_size,
                hidden_size=self._rnn_hidden_size,
                rnn_type=self._rnn_type,
                bidirectional=self._bidirectional)
            rnns.append(('%d' % (x + 1), rnn))

        self.rnns = nn.Sequential(OrderedDict(rnns))

        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(self._rnn_hidden_size, context=self._context),
            nn.Hardtanh(0, 20,
                        inplace=True)) if not self._bidirectional else None

        fc = nn.Sequential(
            nn.BatchNorm1d(self._rnn_hidden_size),
            nn.Linear(self._rnn_hidden_size, self._num_classes, bias=False))
        self.fc = nn.Sequential(SequenceWise(fc))

    def forward(self, x):
        # B x T x D -> B x 1 x D x T
        x = x.unsqueeze(1).transpose(2, 3).contiguous()

        x = self.conv(x)

        # Collapse feature dimension
        B, C, D, T = x.shape

        # B x C x D x T -> T x B x C * D
        x = x.view(B, C * D, T).transpose(1, 2).transpose(0, 1).contiguous()
        x = self.rnns(x)

        # no need for lookahead layer in bidirectional
        if not self._bidirectional:
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)

        # identity in training mode, softmax in eval mode
        if not self.training:
            return F.softmax(x, dim=-1)

        return x

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(
            rnn_type=package['rnn_type'],
            num_classes=package['num_classes'],
            rnn_hidden_size=package['rnn_hidden_size'],
            num_rnn_layers=package['num_rnn_layers'],
            window_size=package['window_size'],
            bidirectional=package['bidirectional'],
            context=package['context'])

        # the blacklist parameters are params that were previous erroneously saved by the model
        # care should be taken in future versions that if batch_norm on the first rnn is required
        # that it be named something else
        blacklist = [
            'rnns.0.batch_norm.module.weight', 'rnns.0.batch_norm.module.bias',
            'rnns.0.batch_norm.module.running_mean',
            'rnns.0.batch_norm.module.running_var'
        ]
        for x in blacklist:
            if x in package['state_dict']:
                del package['state_dict'][x]

        model.load_state_dict(package['state_dict'])

        for x in model.rnns:
            x.flatten_parameters()

        return model

    @classmethod
    def load_model_package(cls, package):
        model = cls(
            rnn_type=package['rnn_type'],
            num_classes=package['num_classes'],
            rnn_hidden_size=package['rnn_hidden_size'],
            num_rnn_layers=package['num_rnn_layers'],
            window_size=package['window_size'],
            bidirectional=package['bidirectional'],
            context=package['context'])
        return model


if __name__ == '__main__':
    import os.path
    import argparse

    parser = argparse.ArgumentParser(
        description='DeepSpeech model information')
    parser.add_argument(
        '--model-path',
        default='models/deepspeech_final.pth',
        help='Path to model file created by training')
    args = parser.parse_args()

    package = torch.load(
        args.model_path, map_location=lambda storage, loc: storage)

    model = DeepSpeech.load_model(args.model_path)

    print("Model name:         ", os.path.basename(args.model_path))
    print("DeepSpeech version: ", model.__version__)
    print("")
    print("Recurrent Neural Network Properties")
    print("  RNN Type:         ", model._rnn_type.__name__.lower())
    print("  RNN Layers:       ", model._hidden_layers)
    print("  RNN Size:         ", model._hidden_size)
    print("  Classes:          ", len(model._labels))
    print("")
    print("Model Features")
    print("  Num classes:      ", model._labels)
    print("  Sample Rate:      ", model._audio_conf.get("sample_rate", "n/a"))
    print("  Window Type:      ", model._audio_conf.get("window", "n/a"))
    print("  Window Size:      ", model._audio_conf.get("window_size", "n/a"))
    print("  Window Stride:    ", model._audio_conf.get(
        "window_stride", "n/a"))

    if package.get('loss', None) is not None:
        print("")
        print("Training Information")
        epochs = package['epoch']
        print("  Epochs:           ", epochs)
        print("  Current Loss:      {0:.3f}".format(
            package['loss'][epochs - 1]))
        print("  Current CER:       {0:.3f}".format(
            package['cer'][epochs - 1]))
        print("  Current WER:       {0:.3f}".format(
            package['wer'][epochs - 1]))

    if package.get('meta', None) is not None:
        print("")
        print("Additional Metadata")
        for k, v in model._meta:
            print("  ", k, ": ", v)
