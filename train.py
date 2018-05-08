import argparse
import json
import os

import torch
from easydict import EasyDict as edict

from codes import metrics, transforms
from codes.data import AudioDataLoader, AudioDataset
from codes.decoder import GreedyDecoder
from codes.model import DeepSpeech
from codes.utils import model_utils as mu
from codes.sampler import BucketingSampler, DistributedBucketingSampler
from ignite import handlers
from ignite.engine import Engine, Events
from warpctc_pytorch import CTCLoss as warp_CTCLoss


def Trainer(model, optimizer, criterion, device=torch.device('cuda'),
            **kwargs):

    data_timer = handlers.Timer(average=True)

    def _update(engine, batch):
        model.train()

        engine.data_timer.resume()
        inputs, targets, input_percentages, target_sizes = batch
        inputs.to(device)
        engine.data_timer.pause()
        engine.data_timer.step()

        out = model(inputs)
        # CTC loss is batch_first = False, i.e., T x B x D
        out = out.transpose(0, 1)

        seq_length = out.shape[0]
        out_sizes = (input_percentages * seq_length).int()

        loss = criterion(out, targets, out_sizes.to('cpu'), target_sizes.to('cpu'))
        loss = loss / inputs.shape[0]  # average the loss by minibatch


        loss_sum = loss.sum()
        inf = float("inf")
        if loss_sum == inf or loss_sum == -inf:
            print("WARNING: received an inf loss, setting loss value to 0")
            loss_value = 0
        else:
            loss_value = loss.item()

        # compute gradient
        optimizer.zero_grad()
        loss.backward()

        # Clipping the norm, avoiding gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                      kwargs.get('max_norm', 400))

        # optimizer step
        optimizer.step()

        torch.cuda.synchronize()

        return loss_value

    engine = Engine(_update)
    engine.data_timer = data_timer

    return engine


def Evaluator(model, metrics, device=torch.device('cuda')):
    def _inference(engine, batch):
        model.eval()

        with torch.no_grad():
            inputs, targets, input_percentages, target_sizes = batch
            inputs.to(device)

            inputs = torch.Tensor(inputs).to(device)

            out = model(inputs)  # NxTxH

            seq_length = out.shape[1]
            out_sizes = (input_percentages * seq_length).int()

            return out, targets, out_sizes, target_sizes

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


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
        model, optimizer, criterion, device=device, **args.config.training)
    train_evaluator = Evaluator(model, metrics, device=device)
    val_evaluator = Evaluator(model, metrics.copy(), device=device)

    print(model)
    print("Number of parameters: {}".format(mu.num_of_parameters(model)))

    #load data_loaders
    train_loader, val_loader = get_data_loaders(args.data_dir,
        args.train_manifest, args.val_manifest, train_transforms,
        val_transforms, target_transforms, args.batch_size, args.num_workers,
        args.distributed, args.local_rank)

    # Sorta grad and shuffle
    if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_loader.batch_sampler.shuffle(start_epoch)

        @trainer.on(Events.STARTED)
        def sampler_on_started(engine):
            print("Shuffling batches for the following epochs")
            train_loader.batch_sampler.shuffle(start_epoch)

    # Learning rate schedule
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, args.config.training.learning_anneal)

    # Iteration logger

    batch_timer = handlers.Timer(average=True)
    batch_timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED)

    if not args.silent:

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_iteration(engine):

            if not args.silent:
                iter = (engine.state.iteration - 1) % len(train_loader) + 1
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_timer:.3f}\t'
                      'Data {data_timer:.3f}\t'
                      'Loss {loss:.4f}\t'.format(
                          (engine.state.epoch),
                          iter,
                          len(train_loader),
                          batch_timer=batch_timer.value(),
                          data_timer=engine.data_timer.value(),
                          loss=engine.state.output))

    # Epoch checkpoint
    ckpt_handler = handlers.ModelCheckpoint(
        os.path.join(args.save_folder, args.config.network.name),
        args.config.network.name,
        save_interval=1,
        n_saved=args.config.training.num_epochs)

    @trainer.on(Events.EPOCH_COMPLETED)
    def trainer_epoch_completed(engine):
        train_evaluator.run(train_loader)
        print(''.join(
            ['Training Summary Epoch: [{0}]\t'.format(engine.state.epoch)
             ] + [
                 'Average {} {:.3f}\t'.format(name, metric)
                 for name, metric in train_evaluator.state.metrics.items()
             ]))

        val_evaluator.run(val_loader)
        print(''.join([
            'Validation Summary Epoch: [{0}]\t'.format(engine.state.epoch)
        ] + [
            'Average {} {:.3f}\t'.format(name, metric)
            for name, metric in val_evaluator.state.metrics.items()
        ]))

        ckpt_handler(
            engine, {
                'state_dict': mu.get_state_dict(model),
                'args': args,
                'optimizer': optimizer,
                'epoch': engine.state.epoch,
                'iteration': engine.state.iteration,
                'metrics': train_evaluator.state.metrics,
                'val_metrics': val_evaluator.state.metrics,
            })

        if not args.no_shuffle:
            print("\nShuffling batches...")
            train_loader.batch_sampler.shuffle(engine.state.epoch)

        old_lr = args.config.training.learning_rate * (
            args.config.training.learning_anneal**engine.state.epoch)
        new_lr = args.config.training.learning_rate * (args.config.training.learning_anneal**
                                       (engine.state.epoch + 1))
        print('\nAnnealing learning rate from {:.5g} to {:5g}.\n'.format(
            old_lr, new_lr))
        scheduler.step()

    # best WER checkpoint
    best_ckpt_handler = handlers.ModelCheckpoint(
        os.path.join(args.save_folder, args.config.network.name),
        args.config.network.name,
        score_function=lambda engine: engine.state.metrics['wer'],
        n_saved=5)

    @val_evaluator.on(Events.EPOCH_COMPLETED)
    def val_evaluator_epoch_completed(engine):
        best_ckpt_handler(
            engine, {
                'state_dict': mu.get_state_dict(model),
                'args': args,
                'optimizer': optimizer,
                'epoch': trainer.state.epoch,
                'iteration': trainer.state.iteration,
                'metrics': train_evaluator.state.metrics,
                'val_metrics': engine.state.metrics,
            })

    trainer.run(train_loader, args.config.training.num_epochs)
