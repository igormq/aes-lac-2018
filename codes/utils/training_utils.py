import json
import logging
import os

import numpy as np
import torch

from codes import transforms
from codes.data import AudioDataLoader, AudioDataset, ConcatAudioDataset
from codes.model import DeepSpeech, MultiTaskModel, SequenceWiseClassifier
from codes.sampler import (BucketingSampler, DistributedBucketingSampler,
                           WeightedBucketingRandomSampler)

LOG = logging.getLogger('aes-lac-2018')
NUM_CLASSES = {'pt_BR': 43, 'en': 29}


def get_default_transforms(data_dir, config):
    train_transforms = transforms.Compose(
        [transforms.ToTensor(augment=config.training.augment),
         transforms.ToSpectrogram(librosa_compat=True)])

    val_transforms = transforms.Compose([transforms.ToTensor(), transforms.ToSpectrogram(librosa_compat=True)])

    target_transforms = []
    for lang in config.model.langs:
        lang_transform = transforms.ToLabel(
            os.path.join(data_dir, 'labels.{}.json'.format(lang)),
            lang=lang,
            remove_accents=False if lang == 'pt_BR' else True)

        target_transforms.append(lang_transform)

    return train_transforms, val_transforms, target_transforms


def get_model(model_dict):

    if isinstance(model_dict.langs, (tuple, set, list)) and len(model_dict.langs) > 1:
        model_dict.params.include_classifier = False
        common = DeepSpeech(**model_dict.params)
        heads = [SequenceWiseClassifier(common._rnn_hidden_size, NUM_CLASSES[lang]) for lang in model_dict.langs]

        return MultiTaskModel(common, heads)
    else:
        LOG.info(model_dict.langs[0])
        model_dict.params = model_dict.get('params', {})
        model_dict.params.setdefault('num_classes', NUM_CLASSES[model_dict.langs[0]])
        return DeepSpeech(**model_dict.params)


def batch_norm_eval_mode(m):
    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        m.eval()


def _freeze_layers(model, freeze_layers):
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
                params.apply(batch_norm_eval_mode)  # set batchnorm to inference mode
                params = params.parameters()

            for p in params:
                num_params += p.numel()
                p.requires_grad = False

        LOG.info('\tFreezed {} parameters'.format(num_params))

    return model


def finetune_model(model, obj):
    freeze_layers = obj.get('freeze_layers', None)
    num_classes = NUM_CLASSES[obj['lang']]
    map_fc = obj.get('map_fc', None)

    model = _freeze_layers(model, freeze_layers)

    last_fc = model.fc[0].module[1]
    if last_fc.out_features != num_classes or (freeze_layers and freeze_layers[0] == 'all'):
        LOG.info('\tChanging the last FC layer')
        old_linear = model.fc[0].module[1]

        new_linear = torch.nn.Linear(last_fc.in_features, num_classes, bias=False)

        model.fc[0].module[1] = new_linear

        if map_fc is not None:
            LOG.info('\t Mapping FC weights')
            map_idxs = json.load(open(map_fc))
            old_idxs, new_idxs = zip(*map_idxs)

            with torch.no_grad():
                # Copy the common weights
                new_linear.weight.index_copy_(0, torch.tensor(new_idxs),
                                              old_linear.weight.index_select(0, torch.tensor(old_idxs)))

                # Random initialize other weights
                other_idxs = list(set(range(num_classes)).difference(set(new_idxs)))
                torch.nn.init.normal_(new_linear.weight[other_idxs], 0, 0.01)

            assert np.alltrue(new_linear == model.fc[0].module[1])
        else:
            LOG.info('\tRandom initiliazing the FC weights')
            torch.nn.init.normal_(new_linear, 0, 0.01)

    return model


def get_optimizer(params, obj):
    return getattr(torch.optim, obj.get('name', 'SGD'))(params, **obj.params)


def get_scheduler(params, obj):
    return getattr(torch.optim.lr_scheduler, obj.get('name', 'SGD'))(params, **obj.params)


def get_per_params_lr(model, obj):
    per_layer_lr = obj.get('per_layer_lr', None)

    if per_layer_lr is None:
        return model.parameters()

    has_base = False
    params, ignored_params = [], []
    for layer_conf in per_layer_lr:

        name = layer_conf[0]
        if name == 'base':
            has_base = True
            continue

        layer_conf_dict = {'params': getattr(model, name).parameters()}

        if len(layer_conf) > 1:
            layer_conf_dict.update({'lr': layer_conf[1]})

        params.append(layer_conf_dict)
        ignored_params.extend(list(map(id, params[-1]['params'])))

    if has_base:
        params.append({'params': filter(lambda p: id(p) not in ignored_params, model.parameters())})

    return params


def get_data_loaders(train_transforms, val_transforms, target_transforms, args):

    if args.zipped:
        train_zip_filename = os.path.splitext(os.path.split(args.train_manifest)[-1])[0] + '.zip'
        val_zip_filename = os.path.splitext(os.path.split(args.val_manifest)[-1])[0] + '.zip'

        train_data_dir = os.path.join(args.data_dir, train_zip_filename)
        val_data_dir = os.path.join(args.data_dir, val_zip_filename)
    else:
        train_data_dir = val_data_dir = args.data_dir

    if not isinstance(target_transforms, (list, tuple)):
        target_transforms = [target_transforms]

    train_dataset = [
        AudioDataset(train_data_dir, args.train_manifest[i], train_transforms, t)
        for i, t in enumerate(target_transforms)
    ]

    val_dataset = [
        AudioDataset(val_data_dir, args.val_manifest[i], val_transforms, t) for i, t in enumerate(target_transforms)
    ]

    if len(target_transforms) == 1:
        train_dataset = train_dataset[0]
        val_dataset = val_dataset[0]
    else:
        train_dataset = ConcatAudioDataset(train_dataset)
        val_dataset = ConcatAudioDataset(val_dataset)

    if not args.distributed:
        if len(target_transforms) == 1:
            train_sampler = BucketingSampler(train_dataset, batch_size=args.config.training.batch_size)
        else:
            sampling = args.config.training.get('sampling', 'equal')
            train_sampler = WeightedBucketingRandomSampler(
                train_dataset,
                batch_size=args.config.training.batch_size,
                sampling=sampling,
                num_epochs=args.config.training.num_epochs)
    else:
        train_sampler = DistributedBucketingSampler(
            train_dataset, batch_size=args.config.training.batch_size, rank=args.local_rank)

    train_loader = AudioDataLoader(
        train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler, num_tasks=len(target_transforms))

    val_loader = AudioDataLoader(
        val_dataset,
        batch_size=args.config.training.batch_size,
        num_workers=args.num_workers,
        num_tasks=len(target_transforms))

    return train_loader, val_loader
