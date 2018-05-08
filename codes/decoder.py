#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Modified to support pytorch Tensors

import warnings

import Levenshtein as Lev
import torch

from codes.preprocessing import OrderedLabelEncoder
from six.moves import xrange

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)


class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Arguments:
        label_encoder (sklearn.preprocessing.label_encoder): LabelEncoder objects.
        blank_index (int, optional): index for the blank '_' character. Default: 0.
    """

    def __init__(self, label_encoder, blank_index=0):
        if isinstance(label_encoder, str):
            label_encoder = list(label_encoder)

        if isinstance(label_encoder, (set, list)):
            label_encoder = OrderedLabelEncoder().fit(label_encoder)

        self.label_encoder = label_encoder
        self.blank_index = blank_index

    def wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))

    def cer(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return Lev.distance(s1, s2)

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError


class BeamCTCDecoder(Decoder):
    def __init__(self,
                 label_encoder,
                 lm_path=None,
                 alpha=0,
                 beta=0,
                 cutoff_top_n=40,
                 cutoff_prob=1.0,
                 beam_width=100,
                 num_processes=4,
                 blank_index=0):
        super(BeamCTCDecoder, self).__init__(label_encoder)
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")
        self._decoder = CTCBeamDecoder(label_encoder, lm_path, alpha, beta,
                                       cutoff_top_n, cutoff_prob, beam_width,
                                       num_processes, blank_index)

    def convert_to_strings(self, out, seq_len):
        results = []
        for b, batch in enumerate(out):
            utterances = []
            for p, utt in enumerate(batch):
                size = seq_len[b][p]
                if size > 0:
                    transcript = ''.join(
                        self.label_encoder.inverse_transform(utt[0:size]))
                else:
                    transcript = ''
                utterances.append(transcript)
            results.append(utterances)
        return results

    def convert_tensor(self, offsets, sizes):
        results = []
        for b, batch in enumerate(offsets):
            utterances = []
            for p, utt in enumerate(batch):
                size = sizes[b][p]
                if sizes[b][p] > 0:
                    utterances.append(utt[0:size])
                else:
                    utterances.append(torch.IntTensor())
            results.append(utterances)
        return results

    def decode(self, probs, sizes=None):
        """
        Decodes probability output using ctcdecode package.
        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes: Size of each sequence in the mini-batch
        Returns:
            string: sequences of the model's best guess for the transcription
        """
        probs = probs.cpu()
        out, _, offsets, seq_lens = self._decoder.decode(probs, sizes)

        strings = self.convert_to_strings(out, seq_lens)
        offsets = self.convert_tensor(offsets, seq_lens)
        return strings, offsets


class GreedyDecoder(Decoder):
    def __init__(self, label_encoder, blank_index=0):
        super(GreedyDecoder, self).__init__(label_encoder, blank_index)

    def convert_to_strings(self,
                           sequences,
                           sizes=None,
                           remove_repetitions=False,
                           return_offsets=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None

        for i in xrange(len(sequences)):
            seq_len = sizes[i] if sizes is not None else len(sequences[i])

            string, string_offsets = self.process_string(
                sequences[i], seq_len, remove_repetitions)
            strings.append([string])  # We only return one path

            if return_offsets:
                offsets.append([string_offsets])

        if return_offsets:
            return strings, offsets

        return strings

    def process_string(self, sequence, size, remove_repetitions=False):
        processed_seq = []
        offsets = []
        for i in range(size):
            curr_seq = sequence[i].item()
            if curr_seq != self.blank_index:
                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and curr_seq == sequence[i - 1].item():
                    continue

                processed_seq.append(curr_seq)
                offsets.append(i)

        if not len(processed_seq):
            return '', torch.IntTensor(offsets)

        return ''.join(self.label_encoder.inverse_transform(
            processed_seq)), torch.IntTensor(offsets)

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of batch x seq_length x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        _, max_probs = torch.max(probs, 2)
        strings, offsets = self.convert_to_strings(
            max_probs.view(max_probs.shape[0], max_probs.shape[1]),
            sizes,
            remove_repetitions=True,
            return_offsets=True)
        return strings, offsets
