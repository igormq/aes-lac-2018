
class BaseLogger(object):
    def __init__(self):
        self.metric = {}
        self.epoch = 0
        self.iteration = 0
        self.logger = {'Train': {}, 'Val': {}}

    def _add_logger(self, metric):
        raise NotImplementedError

    def _add_metric(self, metric):
        self.metric[metric] = []
        self._add_logger(metric)

    def update_metrics(self, metrics, epoch=None, mode='Train'):
        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for name, value in metrics.items():
            if name == 'loss':
                raise ValueError('Metric name `loss` is not allowed.')

            if isinstance(value, (tuple, list)):
                name = ['{}_{}'.format(name, i) for i in range(len(value))]
            else:
                name = [name]
                value = [value]

            for n, v in zip(name, value):
                if n not in self.metric:
                    self._add_metric(n)

                self.metric[n].append([epoch, v])
                self.logger[mode][n].log(epoch, v, name=mode)

    def update_loss(self, loss, iteration=None):
        if iteration is None:
            iteration = self.iteration
            self.iteration += 1

        if 'loss' not in self.metric:
            self._add_metric('loss')

        self.metric['loss'].append([iteration, loss])
        self.logger['Train']['loss'].log(iteration, loss, name='Train')

    def state_dict(self):
        return {
            'metric': self.metric,
            'epoch': self.epoch,
            'iteration': self.iteration
        }

    def load_state_dict(self, state):
        for name, val in state.items():
            setattr(self, name, val)