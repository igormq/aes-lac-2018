import argparse
import json
import os

import torch

from codes import metrics, transforms
from codes.data import AudioDataLoader, AudioDataset
from codes.decoder import GreedyDecoder
from codes.model import DeepSpeech
from codes.sampler import BucketingSampler, DistributedBucketingSampler
from codes.trainer import Trainer
from codes.utils import model_utils as mu
from easydict import EasyDict as edict
from ignite import handlers
from ignite.engine import Engine, Events
from warpctc_pytorch import CTCLoss as warp_CTCLoss


def get_data_loaders(data_dir,
                     train_manifest,
                     val_manifest,
                     train_transforms,
                     val_transforms,
                     target_transforms,
                     batch_size=32,
                     num_workers=4,
                     distributed=False,
                     local_rank=None):

    train_dataset = AudioDataset(data_dir, train_manifest, train_transforms,
                                 target_transforms)

    val_loader = AudioDataset(data_dir, val_manifest, val_transforms, target_transforms)

    if not distributed:
        train_sampler = BucketingSampler(train_dataset, batch_size=batch_size)
    else:
        train_sampler = DistributedBucketingSampler(
            train_dataset, batch_size=batch_size, rank=local_rank)

    train_loader = AudioDataLoader(
        train_dataset, num_workers=num_workers, batch_sampler=train_sampler)

    val_loader = AudioDataLoader(
        val_loader, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader


# def continue_from():
#     if args.continue_from:  # Starting from previous model
#         print("Loading checkpoint model %s" % args.continue_from)
#         package = torch.load(
#             args.continue_from, map_location=lambda storage, loc: storage)
#         model = DeepSpeech.load_model_package(package)
#         labels = DeepSpeech.get_labels(model)
#         audio_conf = DeepSpeech.get_audio_conf(model)
#         parameters = model.parameters()
#         optimizer = torch.optim.SGD(
#             parameters, lr=args.lr, momentum=args.momentum, nesterov=True)

#         if not args.finetune:  # Don't want to restart training
#             optimizer.load_state_dict(package['optim_dict'])

#             start_epoch = int(package.get(
#                 'epoch', 1)) - 1  # Index start at 0 for training
#             start_iter = package.get('iteration', None)
#             if start_iter is None:
#                 start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
#                 start_iter = 0
#             else:
#                 start_iter += 1

#             avg_loss = int(package.get('avg_loss', 0))
#             loss_results, cer_results, wer_results = package[
#                 'loss_results'], package['cer_results'], package['wer_results']

#             if main_proc and args.visdom and \
#                             package[
#                                 'loss_results'] is not None and start_epoch > 0:  # Add previous scores to visdom graph
#                 x_axis = epochs[0:start_epoch]
#                 y_axis = torch.stack(
#                     (loss_results[0:start_epoch], wer_results[0:start_epoch],
#                      cer_results[0:start_epoch]),
#                     dim=1)
#                 viz_window = viz.line(
#                     X=x_axis,
#                     Y=y_axis,
#                     opts=opts,
#                 )
#             if main_proc and args.tensorboard and \
#                             package[
#                                 'loss_results'] is not None and start_epoch > 0:  # Previous scores to tensorboard logs
#                 for i in range(start_epoch):
#                     values = {
#                         'Avg Train Loss': loss_results[i],
#                         'Avg WER': wer_results[i],
#                         'Avg CER': cer_results[i]
#                     }
#                     tensorboard_writer.add_scalars(args.id, values, i + 1)
#     else:
#         with open(args.labels_path) as label_file:
#             labels = str(''.join(json.load(label_file)))

#         audio_conf = dict(
#             sample_rate=args.sample_rate,
#             window_size=args.window_size,
#             window_stride=args.window_stride,
#             window=args.window,
#             noise_dir=args.noise_dir,
#             noise_prob=args.noise_prob,
#             noise_levels=(args.noise_min, args.noise_max))

#         model = DeepSpeech(
#             rnn_hidden_size=args.hidden_size,
#             num_layers=args.hidden_layers,
#             labels=labels,
#             rnn_type=args.rnn_type,
#             audio_conf=audio_conf,
#             bidirectional=args.bidirectional)
#         parameters = model.parameters()
#         optimizer = torch.optim.SGD(
#             parameters, lr=args.lr, momentum=args.momentum, nesterov=True)

if __name__ == '__main__':

    if not torch.cuda.is_available():
        raise RuntimeError('Training script requires GPU. :(')

    # For reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser(
        description='DeepSpeech-ish model training')
    parser.add_argument(
        '--data-dir',
        metavar='DIR',
        help='path to data directory',
        default='data/')
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
        '--config-file',
        '--config',
        required=True,
        help='Path to config JSON file')

    parser.add_argument(
        '--batch-size', default=32, type=int, help='Batch size for training')
    parser.add_argument(
        '--num-workers',
        default=4,
        type=int,
        help='Number of workers used in data-loading')

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
        '--tensorboad-logdir',
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

    with open(args.config_file, 'r', encoding='utf8') as f:
        args.config = json.load(f, object_hook=edict)
        del args.config_file

    device = torch.device('cuda' if args.local else 'cuda:{}'.format(
        args.local_rank))

    main_proc = False
    if args.distributed:
        torch.distributed.init_process_group(
            backend=args.dist_backend, init_method=args.init_method)
        # Only the first proc should save models
        main_proc = args.local_rank == 0

    if main_proc and args.visdom:
        pass

    if main_proc and args.tensorboard:
        pass

    train_transforms = transforms.parse(
        args.config.transforms.train, data_dir=args.data_dir)
    val_transforms = transforms.parse(
        args.config.transforms.val, data_dir=args.data_dir)
    target_transforms = transforms.parse(
        args.config.transforms.label, data_dir=args.data_dir)

    criterion = warp_CTCLoss()
    decoder = GreedyDecoder(target_transforms.label_encoder)


    if args.finetune and args.continue_from:
        model = mu.load_model(args.continue_from)
    else:
        model = DeepSpeech(**args.config.network.params)

    if not args.distributed:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)

    # TODO: continues_from
    start_epoch = 0

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.config.training.learning_rate,
        momentum=args.config.training.momentum,
        nesterov=True)

    metrics = {
        'loss': metrics.CTCLoss(),
        'wer': metrics.WER(decoder),
        'cer': metrics.CER(decoder)
    }

    trainer = Trainer(
        model, optimizer, criterion, metrics, device=device, **args.config.training)

    print(model)
    print("Number of parameters: {}".format(mu.num_of_parameters(model)))

    #load data_loaders
    train_loader, val_loader = get_data_loaders(args.data_dir,
        args.train_manifest, args.val_manifest, train_transforms,
        val_transforms, target_transforms, args.batch_size, args.num_workers,
        args.distributed, args.local_rank)

    trainer.attach(train_loader, val_loader, args)

    trainer.train(args.config.training.num_epochs)
