import argparse
import json
import logging
import os
import pprint
import random
from collections import OrderedDict

import numpy as np
import torch

import argcomplete
from codes import metrics, transforms
from codes.decoder import GreedyDecoder
from codes.engine import create_evaluator, create_trainer
from codes.handlers import TensorboardX, Visdom
from codes.utils import common_utils as cu
from codes.utils import io_utils as iu
from codes.utils import model_utils as mu
from codes.utils import training_utils as tu
from easydict import EasyDict as edict
from ignite import handlers
from ignite.engine import Engine, Events
from warpctc_pytorch import CTCLoss as warp_CTCLoss

LOG = logging.getLogger('aes-lac-2018')


def display_metrics(metrics):
    display = ''
    for name, metric in metrics.items():
        if isinstance(metric, float):
            display += 'Average {} {:.3f}\t'.format(name, metric)
        elif isinstance(metric, (list, tuple)):
            display += 'Average {} '.format(name)
            for i, m in enumerate(metric):
                display += '{:.3f}/'.format(m)
            display = list(display)[:-1]
            display = ''.join(display + ['\t'])
        else:
            display += '{} {}'.format(metric, name)

    return display


if __name__ == '__main__':

    if not torch.cuda.is_available():
        raise RuntimeError('Training script requires GPU. :(')

    # For reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description='DeepSpeech-ish model training')
    parser.add_argument('config_file', help='Path to config JSON file')

    parser.add_argument(
        '--data-dir',
        metavar='DIR',
        help='path to data directory',
        default=os.getenv('PT_DATA_DIR', 'data/'))
    parser.add_argument(
        '--zipped',
        action='store_true',
        help='if `True`, loads training files from .zip file')
    parser.add_argument(
        '--train-manifest',
        nargs='+',
        metavar='DIR',
        help='path to train manifest csv',
        required=True)
    parser.add_argument(
        '--val-manifest',
        metavar='DIR',
        nargs='+',
        help='path to validation manifest csv',
        required=True)

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
        default=os.getenv('PT_OUTPUT_DIR', 'results/'),
        help='Location to save epoch models')

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

    # logging params
    parser.add_argument(
        '-v', '--verbose', action='count', help='Increase log file verbosity')

    argcomplete.autocomplete(parser)
    args = edict(vars(parser.parse_args()))
    args.distributed = not args.local

    with open(args.config_file, 'r', encoding='utf8') as f:
        args.config = json.load(f, object_hook=edict)
        args.config = iu.expand_values(args.config, **args)
        del args.config_file

    cu.setup_logging(
        os.path.join(args.save_folder, args.config.model.name + '.log'),
        args.verbose)

    LOG.info(pprint.pformat(args))

    device = torch.device('cuda' if args.local else 'cuda:{}'.format(
        args.local_rank))

    main_proc = True
    if args.distributed:
        torch.distributed.init_process_group(
            backend=args.dist_backend, init_method=args.init_method)
        # Only the first proc should save models
        main_proc = args.local_rank == 0

    # Load model
    if args.continue_from:
        LOG.info('Loading model from {}'.format(args.continue_from))
        model, _, _ = mu.load_model(args.continue_from)
    else:
        model = tu.get_model(args.config.model)

    # Load optimizer
    params = tu.get_per_params_lr(model, args.config.optimizer)
    optimizer = tu.get_optimizer(params, args.config.optimizer)

    # Learning rate schedule
    scheduler = tu.get_scheduler(optimizer, args.config.scheduler)

    start_epoch, start_iteration = 0, 0
    train_history, val_history = {}, {}
    if args.continue_from and not args.finetune:
        LOG.info('Continue from {} and not fine-tuning'.format(
            args.continue_from))
        ckpt = torch.load(args.continue_from)
        start_epoch = ckpt['epoch']
        start_iteration = ckpt['iteration']
        train_history, val_history = ckpt['metrics'], ckpt['val_metrics']
        args.config = ckpt['args']['config']
        optimizer.load_state_dict(ckpt['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        scheduler.load_state_dict(ckpt['optimizer'])

    train_transforms, val_transforms, target_transforms = tu.get_default_transforms(
        args.data_dir, args.config)

    LOG.info('Train transforms: {}'.format(train_transforms))
    LOG.info('Valid transforms: {}'.format(val_transforms))
    LOG.info('Target transforms: {}'.format(target_transforms))

    if args.continue_from and args.finetune:
        LOG.info('Continue from {} and fine-tuning'.format(args.continue_from))
        model = tu.finetune_model(model, args.config.model)

    model = model.to(device)
    if not args.distributed:
        model = torch.nn.DataParallel(model).to(device)
    else:
        LOG.info('Setup distributed training')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)

    num_langs = len(args.config.model.langs)

    criterion = [warp_CTCLoss() for _ in range(num_langs)]
    decoder = [
        GreedyDecoder(target_transforms[i].label_encoder)
        for i in range(num_langs)
    ]

    metrics = OrderedDict(
        ctcloss=
        metrics.ConcatMetrics([metrics.CTCLoss() for i in range(num_langs)]),
        wer=
        metrics.ConcatMetrics(
            [metrics.WER(decoder[i]) for i in range(num_langs)]),
        cer=
        metrics.ConcatMetrics(
            [metrics.CER(decoder[i]) for i in range(num_langs)])
    )

    LOG.info(model)
    total_params = mu.num_of_parameters(model)
    trainable_params = mu.num_of_parameters(model, True)

    LOG.info("Total params: {}".format(total_params))
    LOG.info("Trainable params: {}".format(trainable_params))
    LOG.info(
        "Non-trainable params: {}".format(total_params - trainable_params))

    # Loading data loaders
    train_loader, val_loader = tu.get_data_loaders(
        train_transforms, val_transforms, target_transforms, args)

    # Creating trainer and evaluator
    trainer = create_trainer(
        model,
        optimizer,
        criterion,
        device,
        skip_n=int(start_iteration % len(train_loader)),
        **args.config.training)
    evaluator = create_evaluator(model, metrics, device)

    ## Handlers
    if main_proc and args.visdom:
        LOG.info('Logging into visdom...')
        visdom = Visdom(args.config.model.name)

    if main_proc and args.tensorboard:
        LOG.info('Logging into Tensorboard')
        tensorboard = TensorboardX(
            os.path.join(args.save_folder, args.config.model.name,
                         'tensorboard'))

    # Epoch checkpoint
    ckpt_handler = handlers.ModelCheckpoint(
        os.path.join(args.save_folder, args.config.model.name),
        'model',
        save_interval=1,
        n_saved=args.config.training.num_epochs,
        require_empty=False)

    # best WER checkpoint
    best_ckpt_handler = handlers.ModelCheckpoint(
        os.path.join(args.save_folder, args.config.model.name),
        'model',
        score_function=lambda engine: engine.state.metrics['cer'],
        n_saved=5,
        require_empty=False)

    if args.checkpoint_per_batch:
        # batch checkpoint
        batch_ckpt_handler = handlers.ModelCheckpoint(
            os.path.join(args.save_folder, args.config.model.name),
            'model',
            save_interval=args.checkpoint_per_batch,
            n_saved=1,
            require_empty=False)

    if not args.silent:
        batch_timer = handlers.Timer(average=True)
        batch_timer.attach(
            trainer,
            start=Events.EPOCH_STARTED,
            resume=Events.ITERATION_STARTED,
            pause=Events.ITERATION_COMPLETED,
            step=Events.ITERATION_COMPLETED)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_iteration(engine):
            iter = (engine.state.iteration - 1) % len(train_loader) + 1
            LOG.info('Epoch: [{0}][{1}/{2}]\t'
                     'Time {batch_timer:.3f}\t'
                     'Data {data_timer:.3f}\t'
                     'Loss {loss:{format}}\t'.format(
                         (engine.state.epoch),
                         iter,
                         len(train_loader),
                         batch_timer=batch_timer.value(),
                         data_timer=engine.data_timer.value(),
                         loss=engine.state.output,
                         format='.4f'
                         if isinstance(engine.state.output, float) else ''))

            if main_proc and args.tensorboard:
                tensorboard.update_loss(
                    engine.state.output, iteration=engine.state.iteration)

            if main_proc and args.visdom:
                visdom.update_loss(
                    engine.state.output, iteration=engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch(engine):
        evaluator.run(train_loader)
        train_metrics = evaluator.state.metrics
        LOG.info('Training Summary Epoch: [{0}]\t'.format(engine.state.epoch) +
                 display_metrics(train_metrics))

        # Saving the values
        for name, metric in train_metrics.items():
            train_history.setdefault(name, [])
            train_history[name].append(metric)

        if main_proc and args.tensorboard:
            tensorboard.update_metrics(train_metrics, epoch=engine.state.epoch)

        if main_proc and args.visdom:
            visdom.update_metrics(train_metrics, epoch=engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_val_epoch(engine):
        evaluator.run(val_loader)
        val_metrics = evaluator.state.metrics
        LOG.info('Validation Summary Epoch: [{0}]\t'.format(
            engine.state.epoch) + display_metrics(val_metrics))

        # Saving the values
        for name, metric in val_metrics.items():
            val_history.setdefault(name, [])
            val_history[name].append(metric)

        if main_proc and args.tensorboard:
            tensorboard.update_metrics(
                val_metrics, epoch=engine.state.epoch, mode='Val')

        if main_proc and args.visdom:
            visdom.update_metrics(
                val_metrics, epoch=engine.state.epoch, mode='Val')

    # Annealing LR
    @trainer.on(Events.EPOCH_COMPLETED)
    def anneal_lr(engine):
        old_lr = args.config.optimizer.params.lr * (
            args.config.scheduler.params.gamma**engine.state.epoch)
        new_lr = args.config.optimizer.params.lr * (
            args.config.scheduler.params.gamma**(engine.state.epoch + 1))
        LOG.info('\nAnnealing learning rate from {:.5g} to {:5g}.\n'.format(
            old_lr, new_lr))
        scheduler.step()

    if args.checkpoint_per_batch:

        @trainer.on(Events.ITERATION_COMPLETED)
        def save_batch_checkpoint(engine):
            batch_ckpt_handler(
                evaluator, {
                    'batch-ckpt': {
                        'args': vars(args),
                        'state_dict': mu.get_state_dict(model),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': engine.state.epoch,
                        'iteration': engine.state.iteration,
                        'metrics': train_history,
                        'val_metrics': val_history,
                    }
                })

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine):
        ckpt_handler(
            engine, {
                'ckpt': {
                    'args': vars(args),
                    'state_dict': mu.get_state_dict(model),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': engine.state.epoch,
                    'iteration': engine.state.iteration,
                    'metrics': train_history,
                    'val_metrics': val_history,
                }
            })

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_best_model(engine):
        best_ckpt_handler(
            evaluator, {
                'best-ckpt': {
                    'args': vars(args),
                    'state_dict': mu.get_state_dict(model),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': engine.state.epoch,
                    'iteration': engine.state.iteration,
                    'metrics': train_history,
                    'val_metrics': val_history,
                }
            })

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_log(engine):
        with open(
                os.path.join(args.save_folder, args.config.model.name,
                             'metrics-log'), 'a') as f:
            f.write('Epoch [{}] '.format(engine.state.epoch))

            for name, history in zip(['Train', 'Val'], [train_history, val_history]):
                f.write('| {} '.format(name))
                for k, v in history.items():
                    f.write('{} '.format(k))
                    if isinstance(v[-1], float):
                        f.write('{:.3f}'.format(v[-1]))
                    elif isinstance(v[-1], (tuple, list)):
                        for i, t_k in enumerate(v[-1]):
                            f.write('{:.3f}{}'.format(t_k, '/' if i < len(v[-1]) - 1 else ''))
                    else:
                        f.write('{}'.format(v[-1]))

    # Sorta grad and shuffle
    if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
        LOG.info("Shuffling batches for the following epochs")
        train_loader.batch_sampler.shuffle(start_epoch)

    if not args.no_shuffle:

        @trainer.on(Events.EPOCH_COMPLETED)
        def epoch_shuffle(engine):
            LOG.info("\nShuffling batches...")
            train_loader.batch_sampler.shuffle(engine.state.epoch)

    # Training
    if args.continue_from and not args.finetune:

        @trainer.on(Events.STARTED)
        def set_start_epoch(engine):
            engine.state.epoch = start_epoch
            engine.state.iteration = start_epoch * len(train_loader)

        @trainer.on(Events.STARTED)
        def start_lr(engine):
            LOG.info('Adjusting initial learning rate')
            scheduler.step(start_epoch)

    trainer.run(train_loader, args.config.training.num_epochs)
