import argparse
import random
import json
import os

import argcomplete
import numpy as np
import torch
from easydict import EasyDict as edict
from warpctc_pytorch import CTCLoss as warp_CTCLoss

from codes import metrics, transforms
from codes.utils import io_utils as iu
from codes.data import AudioDataLoader, AudioDataset
from codes.decoder import GreedyDecoder
from codes.engine import create_evaluator, create_trainer
from codes.model import DeepSpeech
from codes.sampler import BucketingSampler, DistributedBucketingSampler
from codes.utils import model_utils as mu
from ignite import handlers
from ignite.engine import Engine, Events

from codes.handlers import Visdom, TensorboardX

def batch_norm_eval_mode(m):
    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        m.eval()

def finetune_model(model, num_classes, freeze_layers, map_fc):
    if freeze_layers is not None:

        if isinstance(freeze_layers, str):
            freeze_layers = [freeze_layers]

        num_params = 0
        for layer in freeze_layers:

            if layer == 'all':
                params = model.parameters()
            else:
                params = getattr(model, layer)

            if isinstance(params, torch.Tensor):
                params = [params]

            elif isinstance(params, torch.nn.Module):
                params.apply(batch_norm_eval_mode) # set batchnorm to inference mode
                params = params.parameters()

            for p in params:
                num_params += p.numel()
                p.requires_grad = False


        print('\tFreezed {} parameters'.format(num_params))

    last_fc = model.fc[0].module[1]
    if last_fc.out_features != num_classes  or (freeze_layers and freeze_layers[0] == 'all'):
        print('\tChanging the last FC layer')
        old_weight = model.fc[0].module[1].weight

        new_weight = torch.nn.Linear(
            last_fc.in_features,
            num_classes,
            bias=False)

        model.fc[0].module[1].weight = new_weight

        if map_fc is not None:
            print('\t Mapping FC weights')
            map_idxs = json.load(open(map_fc))
            old_idxs, new_idxs = zip(*map_idxs)

            # Copy the common weights
            new_weight[new_idxs].copy_(old_weight[old_idxs])

            # Random initialize other weights
            other_idxs = list(set(range(num_classes)).difference(set(new_idxs)))
            torch.nn.init.normal_(new_weight[other_idxs], 0, 0.01)

            assert np.alltrue(new_weight == model.fc[0].module[1])
        else:
            torch.nn.init.normal_(new_weight, 0, 0.01)

    return model

def get_learning_rate(model, training_params):
    per_layer_lr = training_params.get('per_layer_lr', None)

    if per_layer_lr is None:
        return model.parameters(), training_params['learning_rate']

    has_base = False
    ignored_params = []
    params = []
    for layer_lr in per_layer_lr:
        if len(layer_lr) == 1:
            name = layer_lr[0]
            lr = training_params['learning_rate']
        else:
            name, lr = layer_lr

        if name == 'base':
            has_base = True
            continue

        params.append({
            'params': getattr(model, name).parameters(),
            'lr': lr
        })
        ignored_params.extend(list(map(id, params[-1]['params'])))

    if has_base:
        params.append({
            'params': filter(lambda p: id(p) not in ignored_params, model.parameters())
        })

    return params, training_params['learning_rate']


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

    val_loader = AudioDataset(data_dir, val_manifest, val_transforms,
                              target_transforms)

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
    parser.add_argument(
        '--data-dir',
        metavar='DIR',
        help='path to data directory',
        default=os.getenv('PT_DATA_DIR', 'data/'))
    parser.add_argument(
        '--train-manifest',
        metavar='DIR',
        help='path to train manifest csv',
        required=True)
    parser.add_argument(
        '--val-manifest',
        metavar='DIR',
        help='path to validation manifest csv',
        required=True)
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

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    args.distributed = not args.local

    with open(args.config_file, 'r', encoding='utf8') as f:
        args.config = json.load(f, object_hook=edict)
        args.config = iu.expand_values(args.config, **args)
        del args.config_file

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
        print('Loading model from {}'.format(args.continue_from))
        model, _, _ = mu.load_model(args.continue_from)
    else:
        model = DeepSpeech(**args.config.network.params)

    # Load optmizer
    params, lr = get_learning_rate(model, args.config.training)
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=args.config.training.momentum,
        nesterov=True)

    # Learning rate schedule
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, args.config.training.learning_anneal)

    start_epoch, start_iteration = 0, 0
    train_history, val_history = {}, {}
    if args.continue_from and not args.finetune:
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

    train_transforms = transforms.parse(
        args.config.transforms.train, data_dir=args.data_dir)
    val_transforms = transforms.parse(
        args.config.transforms.val, data_dir=args.data_dir)
    target_transforms = transforms.parse(
        args.config.transforms.label, data_dir=args.data_dir)

    if args.continue_from and args.finetune:
        model = finetune_model(model, len(
                target_transforms.label_encoder.classes_), args.config.network.get('freeze', None), args.config.network.get('map_fc', None))

    model = model.to(device)
    if not args.distributed:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)

    criterion = warp_CTCLoss()
    decoder = GreedyDecoder(target_transforms.label_encoder)

    metrics = {
        'ctcloss': metrics.CTCLoss(),
        'wer': metrics.WER(decoder),
        'cer': metrics.CER(decoder)
    }

    print(model)
    total_params = mu.num_of_parameters(model)
    trainable_params = mu.num_of_parameters(model, True)

    print("Total params: {}".format(total_params))
    print("Trainable params: {}".format(trainable_params))
    print("Non-trainable params: {}".format(total_params - trainable_params))

    # Loading data loaders
    train_loader, val_loader = get_data_loaders(
        args.data_dir, args.train_manifest, args.val_manifest,
        train_transforms, val_transforms, target_transforms, args.batch_size,
        args.num_workers, args.distributed, args.local_rank)

    # Creating trainer and evaluator
    trainer = create_trainer(model, optimizer, criterion, device,
                             skip_n=int(start_iteration % len(train_loader)), **args.config.training)
    evaluator = create_evaluator(model, metrics, device)

    ## Handlers
    if main_proc and args.visdom:
        print('Logging into visdom...')
        visdom = Visdom(args.config.network.name)

    if main_proc and args.tensorboard:
        print('Logging into Tensorboard')
        tensorboard = TensorboardX(os.path.join(args.save_folder, args.config.network.name, 'tensorboard'))
        # dummy_input = train_loader.dataset[0][0][None, ...]
        # tensorboard.add_graph(model, (dummy_input,))

    # Epoch checkpoint
    ckpt_handler = handlers.ModelCheckpoint(
        os.path.join(args.save_folder, args.config.network.name),
        'model',
        save_interval=1,
        n_saved=args.config.training.num_epochs,
        require_empty=False)

    # best WER checkpoint
    best_ckpt_handler = handlers.ModelCheckpoint(
        os.path.join(args.save_folder, args.config.network.name),
        'model',
        score_function=lambda engine: engine.state.metrics['wer'],
        n_saved=5,
        require_empty=False)

    if args.checkpoint_per_batch:
        # batch checkpoint
        batch_ckpt_handler = handlers.ModelCheckpoint(
            os.path.join(args.save_folder, args.config.network.name),
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
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_timer:.3f}\t'
                'Data {data_timer:.3f}\t'
                'Loss {loss:{format}}\t'.format(
                    (engine.state.epoch),
                    iter,
                    len(train_loader),
                    batch_timer=batch_timer.value(),
                    data_timer=engine.data_timer.value(),
                    loss=engine.state.output,
                    format='.4f' if isinstance(engine.state.output, float) else  ''),
                flush=True)

            if main_proc and args.tensorboard:
                tensorboard.update_loss(engine.state.output, iteration=engine.state.iteration)

            if main_proc and args.visdom:
                visdom.update_loss(engine.state.output, iteration=engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch(engine):
        evaluator.run(train_loader)
        train_metrics = evaluator.state.metrics
        print(
            ''.join(
                ['Training Summary Epoch: [{0}]\t'.format(engine.state.epoch)
                 ] + [
                     'Average {} {:.3f}\t'.format(name, metric)
                     for name, metric in train_metrics.items()
                 ]),
            flush=True)

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
        print(
            ''.join([
                'Validation Summary Epoch: [{0}]\t'.format(engine.state.epoch)
            ] + [
                'Average {} {:.3f}\t'.format(name, metric)
                for name, metric in val_metrics.items()
            ]),
            flush=True)

        # Saving the values
        for name, metric in val_metrics.items():
            val_history.setdefault(name, [])
            val_history[name].append(metric)

        if main_proc and args.tensorboard:
            tensorboard.update_metrics(val_metrics, epoch=engine.state.epoch, mode='Val')

        if main_proc and args.visdom:
            visdom.update_metrics(val_metrics, epoch=engine.state.epoch, mode='Val')

    # Annealing LR
    @trainer.on(Events.EPOCH_COMPLETED)
    def anneal_lr(engine):
        old_lr = args.config.training.learning_rate * (
            args.config.training.learning_anneal**engine.state.epoch)
        new_lr = args.config.training.learning_rate * (
            args.config.training.learning_anneal**(engine.state.epoch + 1))
        print(
            '\nAnnealing learning rate from {:.5g} to {:5g}.\n'.format(
                old_lr, new_lr),
            flush=True)
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
        with open(os.path.join(args.save_folder, args.config.network.name, 'metrics-log'), 'a') as f:
            f.write('Epoch [{}] '.format(engine.state.epoch))
            f.write('| Train {}'.format(' '.join(['{} {:.3f}'.format(k,v[-1]) for k, v in train_history.items()])))
            f.write('| Val {}\n'.format(' '.join(['{} {:.3f}'.format(k,v[-1]) for k, v in val_history.items()])))

    # Sorta grad and shuffle
    if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
        print("Shuffling batches for the following epochs", flush=True)
        train_loader.batch_sampler.shuffle(start_epoch)

    if not args.no_shuffle:

        @trainer.on(Events.EPOCH_COMPLETED)
        def epoch_shuffle(engine):
            print("\nShuffling batches...", flush=True)
            train_loader.batch_sampler.shuffle(engine.state.epoch)

    # Training
    if args.continue_from and not args.finetune:
        @trainer.on(Events.STARTED)
        def set_start_epoch(engine):
            engine.state.epoch = start_epoch
            engine.state.iteration = start_epoch * len(train_loader)

        @trainer.on(Events.STARTED)
        def start_lr(engine):
            print('Adjusting initial learning rate')
            scheduler.step(start_epoch)

    trainer.run(train_loader, args.config.training.num_epochs)
