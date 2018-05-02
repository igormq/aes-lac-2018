import argparse
import errno
import json
import os
import time

import torch.distributed as dist
import torch.utils.data.distributed
from tqdm import tqdm
from warpctc_pytorch import CTCLoss

from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
from data.distributed import DistributedDataParallel
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns


def get_data_loaders(train_manifest,
                     val_manifest,
                     labels,
                     audio_conf,
                     batch_size=32,
                     num_workers=4,
                     augment=False,
                     distributed=False,
                     local_rank=None):

    train_dataset = SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath=train_manifest,
        labels=labels,
        normalize=True,
        augment=augment)

    test_dataset = SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath=val_manifest,
        labels=labels,
        normalize=True,
        augment=False)

    if not distributed:
        train_sampler = BucketingSampler(train_dataset, batch_size=batch_size)
    else:
        train_sampler = DistributedBucketingSampler(
            train_dataset, batch_size=batch_size, rank=local_rank)

    train_loader = AudioDataLoader(
        train_dataset, num_workers=num_workers, batch_sampler=train_sampler)

    test_loader = AudioDataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers)

def continue_from():
    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(
            args.continue_from, map_location=lambda storage, loc: storage)
        model = DeepSpeech.load_model_package(package)
        labels = DeepSpeech.get_labels(model)
        audio_conf = DeepSpeech.get_audio_conf(model)
        parameters = model.parameters()
        optimizer = torch.optim.SGD(
            parameters, lr=args.lr, momentum=args.momentum, nesterov=True)

        if not args.finetune:  # Don't want to restart training
            optimizer.load_state_dict(package['optim_dict'])

            start_epoch = int(package.get(
                'epoch', 1)) - 1  # Index start at 0 for training
            start_iter = package.get('iteration', None)
            if start_iter is None:
                start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
                start_iter = 0
            else:
                start_iter += 1

            avg_loss = int(package.get('avg_loss', 0))
            loss_results, cer_results, wer_results = package[
                'loss_results'], package['cer_results'], package['wer_results']

            if main_proc and args.visdom and \
                            package[
                                'loss_results'] is not None and start_epoch > 0:  # Add previous scores to visdom graph
                x_axis = epochs[0:start_epoch]
                y_axis = torch.stack(
                    (loss_results[0:start_epoch], wer_results[0:start_epoch],
                     cer_results[0:start_epoch]),
                    dim=1)
                viz_window = viz.line(
                    X=x_axis,
                    Y=y_axis,
                    opts=opts,
                )
            if main_proc and args.tensorboard and \
                            package[
                                'loss_results'] is not None and start_epoch > 0:  # Previous scores to tensorboard logs
                for i in range(start_epoch):
                    values = {
                        'Avg Train Loss': loss_results[i],
                        'Avg WER': wer_results[i],
                        'Avg CER': cer_results[i]
                    }
                    tensorboard_writer.add_scalars(args.id, values, i + 1)
    else:
        with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

        audio_conf = dict(
            sample_rate=args.sample_rate,
            window_size=args.window_size,
            window_stride=args.window_stride,
            window=args.window,
            noise_dir=args.noise_dir,
            noise_prob=args.noise_prob,
            noise_levels=(args.noise_min, args.noise_max))

        model = DeepSpeech(
            rnn_hidden_size=args.hidden_size,
            nb_layers=args.hidden_layers,
            labels=labels,
            rnn_type=supported_rnns[args.rnn_type],
            audio_conf=audio_conf,
            bidirectional=args.bidirectional)
        parameters = model.parameters()
        optimizer = torch.optim.SGD(
            parameters, lr=args.lr, momentum=args.momentum, nesterov=True)

if __name__ == '__main__':

    if not torch.cuda.is_available():
        raise RuntimeError('Training script requires GPU. :(')

    # For reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser(description='DeepSpeech-ish model training')
    parser.add_argument(
        '--train-manifest',
        metavar='DIR',
        help='path to train manifest csv',
        default='data/train_manifest.csv')
    parser.add_argument(
        '--val-manifest',
        metavar='DIR',
        help='path to validation manifest csv',
        default='data/val_manifest.csv')
    parser.add_argument(
        '--labels-path',
        default='labels.json',
        help='Contains all characters for transcription')

    # Audio pre-processing
    parser.add_argument(
        '--window-size',
        default=.02,
        type=float,
        help='Window size for spectrogram in seconds')
    parser.add_argument(
        '--window-stride',
        default=.01,
        type=float,
        help='Window stride for spectrogram in seconds')
    parser.add_argument(
        '--window',
        default='hamming',
        help='Window type for spectrogram generation')
    parser.add_argument(
        '--sample-rate', default=16000, type=int, help='Sample rate')

    parser.add_argument(
        '--epochs', default=70, type=int, help='Number of training epochs')
    parser.add_argument(
        '--batch-size', default=32, type=int, help='Batch size for training')
    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=3e-4,
        type=float,
        help='initial learning rate')
    parser.add_argument(
        '--num-workers',
        default=4,
        type=int,
        help='Number of workers used in data-loading')
    parser.add_argument(
        '--hidden-size', default=800, type=int, help='Hidden size of RNNs')
    parser.add_argument(
        '--hidden-layers', default=5, type=int, help='Number of RNN layers')
    parser.add_argument(
        '--rnn-type',
        default='gru',
        help='Type of the RNN. rnn|gru|lstm are supported',
        choices=['rnn', 'gru', 'lstm'])
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument(
        '--max-norm',
        default=400,
        type=int,
        help='Norm cutoff to prevent explosion of gradients')
    parser.add_argument(
        '--learning-anneal',
        default=1.1,
        type=float,
        help='Annealing applied to learning rate every epoch')
    parser.add_argument(
        '--silent',
        dest='silent',
        action='store_true',
        help='Turn off progress tracking per iteration')
    parser.add_argument(
        '--checkpoint',
        dest='checkpoint',
        action='store_true',
        help='Enables checkpoint saving of model')
    parser.add_argument(
        '--checkpoint-per-batch',
        default=0,
        type=int,
        help='Save checkpoint per batch. 0 means never save')
    parser.add_argument(
        '--visdom',
        dest='visdom',
        action='store_true',
        help='Turn on visdom graphing')
    parser.add_argument(
        '--tensorboard',
        dest='tensorboard',
        action='store_true',
        help='Turn on tensorboard graphing')
    parser.add_argument(
        '--log-dir',
        default='visualize/model_final',
        help='Location of tensorboard log')
    parser.add_argument(
        '--log-params',
        dest='log_params',
        action='store_true',
        help='Log parameter values and gradients')
    parser.add_argument(
        '--id',
        default='AES LAC 2018 training',
        help='Identifier for visdom/tensorboard run')
    parser.add_argument(
        '--save-folder',
        default='models/',
        help='Location to save epoch models')
    parser.add_argument(
        '--model-path',
        default='models/model_final.pth',
        help='Location to save best validation model')
    parser.add_argument(
        '--continue-from', default='', help='Continue from checkpoint model')
    parser.add_argument(
        '--finetune',
        dest='finetune',
        action='store_true',
        help='Finetune the model from checkpoint "continue_from"')
    parser.add_argument(
        '--augment',
        dest='augment',
        action='store_true',
        help='Use random tempo and gain perturbations.')
    parser.add_argument(
        '--noise-dir',
        default=None,
        help=
        'Directory to inject noise into audio. If default, noise Inject not added'
    )
    parser.add_argument(
        '--noise-prob',
        default=0.4,
        help='Probability of noise being added per sample')
    parser.add_argument(
        '--noise-min',
        default=0.0,
        help=
        'Minimum noise level to sample from. (1.0 means all noise, not original signal)',
        type=float)
    parser.add_argument(
        '--noise-max',
        default=0.5,
        help='Maximum noise levels to sample from. Maximum 1.0',
        type=float)
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help=
        'Turn off shuffling and sample from dataset based on sequence length (smallest to largest)'
    )
    parser.add_argument(
        '--no-sorta-grad',
        action='store_true',
        help=
        'Turn off ordering of dataset on sequence length for the first epoch.')
    parser.add_argument(
        '--no-bidirectional',
        dest='bidirectional',
        action='store_false',
        default=True,
        help='Turn off bi-directional RNNs, introduces lookahead convolution')

    # Distributed params
    parser.add_argument('--local', action='store_true')
    parser.add_argument(
        '--init-method',
        default='env://',
        type=str,
        help='url used to set up distributed training')
    parser.add_argument(
        '--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument(
        '--local-rank', type=int, help='The rank of this process')

    args = parser.parse_args()
    args.distributed = not args.local

    device = torch.device('cuda' if args.local else 'cuda:{}'.format(
        args.local_rank))

    main_proc = False
    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.init_method)
        # Only the first proc should save models
        main_proc = args.local_rank == 0


    if main_proc and args.visdom:
        pass
    if main_proc and args.tensorboard:
        pass

    criterion = CTCLoss()
    avg_loss, start_epoch, start_iter = 0, 0, 0

    # continues_from

    #load data_loaders

    decoder = GreedyDecoder(labels)

    if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    if not args.distributed:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)

    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    batch_time = Timer(average=True)
    batch_time.attach(trainer,
                  start=Events.EPOCH_STARTED,
                  resume=Events.ITERATION_STARTED,
                  pause=Events.ITERATION_COMPLETED,
                  step=Events.ITERATION_COMPLETED)

    def trainer(model, optimizer, criterion):

        data_timer = Timer(average=False)

        def _update(engine, batch):

            engine.state.data_timer.resume()
            inputs, targets, input_percentages, target_sizes = batch
            engine.state.data_timer.pause()
            engine.state.data_timer.step()

            inputs = torch.Tensor(inputs)
            target_sizes = torch.Tensor(target_sizes)
            targets = torch.Tensor(targets)

            out = model(inputs)
            out = out.transpose(0, 1)  # TxNxH

            seq_length = out.size(0)
            sizes = torch.Tensor(input_percentages.mul_(int(seq_length)).int())

            loss = criterion(out, targets, sizes, target_sizes)
            loss = loss / inputs.size(0)  # average the loss by minibatch

            loss_sum = loss.data.sum()
            inf = float("inf")
            if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss.item()

            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)

            # optimizer step
            optimizer.step()

            if args.cuda:
                torch.cuda.synchronize()

            return loss_value

        return Engine(_update)


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_iteration(engine):
        if not args.silent:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time:.3f}\t'
                    'Data {data_time:.3f}\t'
                    'Loss {loss:.4f}\t'.format(
                        (epoch + 1), (i + 1),
                        len(train_sampler),
                        batch_time=batch_time,
                        data_time=data_time,
                    loss=losses))

    # checkpoint
    if args.checkpoint_per_batch > 0:
        ckpt_handler = ModelCheckpoint(os.path.join(save_folder, 'model'), 'myprefix', save_interval=1, n_saved=5)
        ckpt_handler.add_event_handler(Events.ITERATION_COMPLETED, handler, {'mymodel': DeepSpeech.serialize(
                model,
                optimizer=optimizer,
                epoch=epoch,
                iteration=i,
                loss_results=loss_results,
                wer_results=wer_results,
                cer_results=cer_results,
                avg_loss=avg_loss), file_path)})


    def evaluator(model, metrics):

        def _inference(engine, batch):
            inputs, targets, input_percentages, target_sizes = batch

            inputs = torch.Tensor(inputs).to(device)


            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size

            out = model(inputs)  # NxTxH
            seq_length = out.size(1)
            sizes = input_percentages.mul_(int(seq_length)).int()

            decoded_output, _ = decoder.decode(out.data, sizes)
            target_strings = decoder.convert_to_strings(split_targets)

            wer, cer = 0, 0
            for x in range(len(target_strings)):
                transcript, reference = decoded_output[x][0], target_strings[
                    x][0]
                wer += decoder.wer(transcript, reference) / float(
                    len(reference.split()))
                cer += decoder.cer(transcript, reference) / float(
                    len(reference))

            total_cer += cer
            total_wer += wer

            if args.cuda:
                torch.cuda.synchronize()
            del out

        wer = (total_wer / len(test_loader.dataset)) * 100
        cer = (total_cer / len(test_loader.dataset)) * 100

        loss_results[epoch] = avg_loss
        wer_results[epoch] = wer
        cer_results[epoch] = cer

        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(epoch + 1, wer=wer, cer=cer))

        if args.visdom and main_proc:
            x_axis = epochs[0:epoch + 1]
            y_axis = torch.stack(
                (loss_results[0:epoch + 1], wer_results[0:epoch + 1],
                 cer_results[0:epoch + 1]),
                dim=1)
            if viz_window is None:
                viz_window = viz.line(
                    X=x_axis,
                    Y=y_axis,
                    opts=opts,
                )
            else:
                viz.line(
                    X=x_axis.unsqueeze(0).expand(
                        y_axis.size(1),
                        x_axis.size(0)).transpose(0, 1),  # Visdom fix
                    Y=y_axis,
                    win=viz_window,
                    update='replace',
                )

        if args.tensorboard and main_proc:
            values = {
                'Avg Train Loss': avg_loss,
                'Avg WER': wer,
                'Avg CER': cer
            }
            tensorboard_writer.add_scalars(args.id, values, epoch + 1)
            if args.log_params:
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    tensorboard_writer.add_histogram(tag, to_np(value),
                                                     epoch + 1)
                    tensorboard_writer.add_histogram(tag + '/grad',
                                                     to_np(value.grad),
                                                     epoch + 1)

        if args.checkpoint and main_proc:
            file_path = '%s/deepspeech_%d.pth.tar' % (args.save_folder,
                                                      epoch + 1)
            torch.save(
                DeepSpeech.serialize(
                    model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss_results=loss_results,
                    wer_results=wer_results,
                    cer_results=cer_results), file_path)

        # anneal lr
        optim_state = optimizer.state_dict()
        optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0][
            'lr'] / args.learning_anneal
        optimizer.load_state_dict(optim_state)
        print('Learning rate annealed to: {lr:.6f}'.format(
            lr=optim_state['param_groups'][0]['lr']))

        if (best_wer is None or best_wer > wer) and main_proc:
            print(
                "Found better validated model, saving to %s" % args.model_path)
            torch.save(
                DeepSpeech.serialize(
                    model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss_results=loss_results,
                    wer_results=wer_results,
                    cer_results=cer_results), args.model_path)
            best_wer = wer

        avg_loss = 0
        if not args.no_shuffle:
            print("Shuffling batches...")
            train_sampler.shuffle(epoch)