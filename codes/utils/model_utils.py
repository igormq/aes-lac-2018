import torch

from codes import transforms as T
from codes.model import DeepSpeech
from easydict import EasyDict as edict

from . import training_utils as tu


def num_of_parameters(model, trainable=False):
    params = 0
    for p in model.parameters():
        if trainable and not p.requires_grad:
            continue
        params += p.numel()
    return params


def get_state_dict(model):
    model_is_cuda = next(model.parameters()).is_cuda
    model = model.module if model_is_cuda else model
    return model.state_dict()


def map_old_json(args):
    obj = args.config
    new_obj = edict(model=edict(), training=edict(), optimizer=edict(), scheduler=edict())

    new_obj.model.name = obj.network.name
    new_obj.model.map_fc = obj.network.get('map_fc', None)
    new_obj.model.freeze_layers = obj.network.get('freeze_layers', None)
    new_obj.model.langs = [obj.transforms.label[0].params.labels.split('.')[-2]]
    new_obj.model.params = obj.network.params
    new_obj.model.params.num_classes = tu.NUM_CLASSES[new_obj.model.langs[0]]

    new_obj.training.num_epochs = obj.training.num_epochs
    new_obj.training.batch_size = args.batch_size
    new_obj.training.max_norm = obj.training.max_norm
    new_obj.training.augment = obj.transforms.train[0].params.augment
    new_obj.training.finetune = args.finetune

    new_obj.optimizer.name = 'SGD'
    new_obj.optimizer.params = edict()
    new_obj.optimizer.params.lr = obj.training.learning_rate
    new_obj.optimizer.params.momentum = obj.training.momentum
    new_obj.optimizer.params.nesterov = True
    new_obj.optimizer.per_layer_lr = obj.training.get('per_layer_lr', None)

    new_obj.scheduler.name = 'ExponentialLR'
    new_obj.scheduler.params = edict()
    new_obj.scheduler.params.gamma = obj.training.learning_anneal

    args.config = new_obj
    return args


def load_model(model_path, num_classes=29, return_transforms=False, data_dir=None):
    ckpt = torch.load(model_path, map_location={'cuda:0': 'cpu'})

    if 'version' in ckpt and ckpt['version'] == '0.0.1':
        return load_legacy_model(ckpt)

    args = edict(ckpt['args'])
    if 'network' in args.config:
        args = map_old_json(args)

    data_dir = data_dir or args.data_dir

    model = tu.get_model(args.config.model)
    model.load_state_dict(ckpt['state_dict'])

    if not return_transforms:
        return model

    train_transforms, val_transforms, target_transforms = tu.get_default_transforms(data_dir, args.config)

    return model, train_transforms, val_transforms, target_transforms


def load_legacy_model(config):
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

    model = DeepSpeech(
        rnn_type=rnn_type,
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
        'rnns.0.batch_norm.module.weight', 'rnns.0.batch_norm.module.bias', 'rnns.0.batch_norm.module.running_mean',
        'rnns.0.batch_norm.module.running_var'
    ]
    for b in blacklist:
        if b in state_dict:
            del state_dict[b]

    model.load_state_dict(state_dict)

    transforms = T.Compose([
        T.ToTensor(augment=False, sample_rate=sample_rate),
        T.ToSpectrogram(frame_length=frame_length, hop=hop, librosa_compat=True)
    ])

    target_transforms = T.ToLabel(labels)

    return model, transforms, target_transforms
