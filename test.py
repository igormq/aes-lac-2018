import argparse
import tempfile

import numpy as np
import torch
import json
from tqdm import tqdm

from codes.data import AudioDataLoader, AudioDataset
from codes.decoder import GreedyDecoder
from codes.model import DeepSpeech
from codes.transforms import Compose, ToLabel, ToSpectrogram, ToTensor


def _load_legacy(config):
    assert config['version'] == '0.0.1'

    sample_rate = config['audio_conf']['sample_rate']
    frame_length = int(sample_rate * config['audio_conf']['window_size'])
    hop = int(sample_rate * config['audio_conf']['window_stride'])
    rnn_hidden_size = config['hidden_size']
    num_rnn_layers = config['hidden_layers']
    rnn_type = config['rnn_type']
    labels = config['labels']
    num_classes = len(labels)
    bidirectional = config['bidirectional']
    context = config.get('context', 20)
    state_dict = config['state_dict']

    model = DeepSpeech(rnn_type=rnn_type,
                num_classes=num_classes,
                rnn_hidden_size=rnn_hidden_size,
                num_rnn_layers=num_rnn_layers,
                window_size=frame_length,
                bidirectional=bidirectional,
                context=context)


    # the blacklist parameters are params that were previous erroneously saved by the model
    # care should be taken in future versions that if batch_norm on the first rnn is required
    # that it be named something else
    blacklist = [
        'rnns.0.batch_norm.module.weight', 'rnns.0.batch_norm.module.bias',
        'rnns.0.batch_norm.module.running_mean',
        'rnns.0.batch_norm.module.running_var'
    ]
    for b in blacklist:
        if b in state_dict:
            del state_dict[b]

    model.load_state_dict(state_dict)

    transforms = Compose([ToTensor(augment=False, sample_rate=sample_rate),
                          ToSpectrogram(frame_length=frame_length, hop=hop, librosa_compat=True)])

    target_transforms = ToLabel(labels)

    return model, transforms, target_transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSpeech transcription')
    parser.add_argument('--data-dir')
    parser.add_argument('--model-path', default='models/deepspeech_final.pth',
                        help='Path to model file created by training')
    parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
    parser.add_argument('--manifest', metavar='DIR',
                        help='path to validation manifest csv', default='data/test_manifest.csv')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam", "none"], type=str, help="Decoder to use")
    parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
    no_decoder_args = parser.add_argument_group("No Decoder Options", "Configuration options for when no decoder is "
                                                                    "specified")
    no_decoder_args.add_argument('--output-path', default=None, type=str, help="Where to save raw acoustic output")
    beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
    beam_args.add_argument('--top-paths', default=1, type=int, help='number of beams to return')
    beam_args.add_argument('--beam-width', default=10, type=int, help='Beam width to use')
    beam_args.add_argument('--lm-path', default=None, type=str,
                        help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
    beam_args.add_argument('--alpha', default=0.8, type=float, help='Language model weight')
    beam_args.add_argument('--beta', default=1, type=float, help='Language model word bonus (all words)')
    beam_args.add_argument('--cutoff-top-n', default=40, type=int,
                        help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                                'vocabulary will be used in beam search, default 40.')
    beam_args.add_argument('--cutoff-prob', default=1.0, type=float,
                        help='Cutoff probability in pruning,default 1.0, no pruning.')
    beam_args.add_argument('--lm-workers', default=1, type=int, help='Number of LM processes to use')
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    model, transforms, target_transforms = _load_legacy(torch.load(args.model_path))
    label_encoder = target_transforms.label_encoder

    model.eval()

    device = 'cpu'
    if args.cuda:
        device = 'cuda'
        model.to(device)

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(label_encoder, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    elif args.decoder == "greedy":
        decoder = GreedyDecoder(label_encoder)
    else:
        decoder = None

    target_decoder = GreedyDecoder(label_encoder)

    dataset = AudioDataset(args.data_dir, args.manifest, transforms=transforms, target_transforms=target_transforms)

    loader = AudioDataLoader(dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []
    for i, (data) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets, input_percentages, target_sizes = data

        inputs = inputs.to(device)

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        if args.cuda:
            inputs = inputs.cuda()

        out = model(inputs)  # NxTxH

        seq_length = out.shape[1]
        sizes = input_percentages.mul_(int(seq_length)).int()

        if decoder is None:
            # add output to data array, and continue
            output_data.append((out.numpy(), sizes.numpy()))
            continue

        decoded_output, _, = decoder.decode(out, sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)
        for i in range(len(target_strings)):
            transcript, reference = decoded_output[i][0], target_strings[i][0]
            wer_inst = decoder.wer(transcript, reference)
            cer_inst = decoder.cer(transcript, reference)
            total_wer += wer_inst
            total_cer += cer_inst
            num_tokens += len(reference.split())
            num_chars += len(reference)
            if args.verbose:
                print("Ref:", reference.lower())
                print("Hyp:", transcript.lower())
                print("WER:", float(wer_inst) / len(reference.split()), "CER:", float(cer_inst) / len(reference), "\n")

    if decoder is not None:
        wer = float(total_wer) / num_tokens
        cer = float(total_cer) / num_chars

        print('Test Summary \t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(wer=wer * 100, cer=cer * 100))
    else:
        np.save(args.output_path, output_data)
