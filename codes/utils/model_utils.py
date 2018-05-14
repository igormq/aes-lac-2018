
import torch
from codes.model import DeepSpeech
from codes import transforms as T

def num_of_parameters(model):
    params = 0
    for p in model.parameters():
        params += p.numel()
    return params

def get_state_dict(model):
    model_is_cuda = next(model.parameters()).is_cuda
    model = model.module if model_is_cuda else model
    return model.state_dict()

def load_model(model_path):
    config = torch.load(model_path)

    if 'version' in config and config['version'] == '0.0.1':
        return load_legacy_model(config)

    raise NotImplementedError('Not implemented yet.')

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

    transforms = T.Compose([T.ToTensor(augment=False, sample_rate=sample_rate),
                          T.ToSpectrogram(frame_length=frame_length, hop=hop, librosa_compat=True)])

    target_transforms = T.ToLabel(labels)

    return model, transforms, target_transforms