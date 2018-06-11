import argparse

import numpy as np
import torch
from tqdm import tqdm

from codes.data import AudioDataLoader, AudioDataset
from codes.decoder import GreedyDecoder
from codes.utils.model_utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSpeech transcription')
    parser.add_argument('--data-dir')
    parser.add_argument(
        '--model-path', default='models/deepspeech_final.pth', help='Path to model file created by training')
    parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
    parser.add_argument(
        '--manifest', metavar='DIR', help='path to validation manifest csv', default='data/test_manifest.csv')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--decoder', default="greedy", choices=["greedy", "none"], type=str, help="Decoder to use")
    parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
    no_decoder_args = parser.add_argument_group("No Decoder Options", "Configuration options for when no decoder is "
                                                "specified")
    no_decoder_args.add_argument('--output-path', default=None, type=str, help="Where to save raw acoustic output")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    model, _, transforms, target_transforms = load_model(
        args.model_path, return_transforms=True, data_dir=args.data_dir)
    model.eval()

    target_transforms = target_transforms[0]
    label_encoder = target_transforms.label_encoder

    device = 'cpu'
    if args.cuda:
        device = 'cuda'
        model.to(device)

    if args.decoder == "greedy":
        decoder = GreedyDecoder(label_encoder)
    else:
        decoder = None

    target_decoder = GreedyDecoder(label_encoder)

    dataset = AudioDataset(args.data_dir, args.manifest, transforms=transforms, target_transforms=target_transforms)

    loader = AudioDataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []
    with tqdm(total=len(loader)) as pbar:
        for i, (data) in enumerate(loader):
            inputs, targets, input_percentages, target_sizes = data

            inputs = inputs.to(device)

            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size

            out = model(inputs)  # NxTxH

            seq_length = out.shape[1]
            sizes = input_percentages.mul_(int(seq_length)).int()

            if decoder is None:
                # add output to data array, and continue
                output_data.append((out.cpu().numpy(), sizes.numpy()))
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
                    pbar.write("Ref: {}".format(reference.lower()))
                    pbar.write("Hyp: {}".format(transcript.lower()))
                    pbar.write("WER: {}\t CER: {}\n".format(
                        float(wer_inst) / len(reference.split()),
                        float(cer_inst) / len(reference)))

            pbar.update(1)

    if decoder is not None:
        wer = float(total_wer) / num_tokens
        cer = float(total_cer) / num_chars

        print('Test Summary \t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(wer=wer * 100, cer=cer * 100))
    else:
        np.save(args.output_path, output_data)
