import errno
import logging
import os

from tensorboardX import SummaryWriter

from .logger import BaseLogger

LOG = logging.getLogger('aes-lac-2018')


class TensorboardXLogger(object):
    pass


class TensorboardX(BaseLogger):
    def __init__(self, log_dir):
        super().__init__()

        try:
            os.makedirs(log_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                LOG.info('Tensorboard log directory already exists.')
                for file in os.listdir(log_dir):
                    file_path = os.path.join(log_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception:
                        raise
            else:
                raise

        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def add_graph(self, model, dummy_input):
        self.writer.add_graph(model, dummy_input)


    def _add_logger(self, metric):
        def add_scalar(x, y, name='Train'):
            self.writer.add_scalar('{}/{}'.format(name, metric), y, x)

        metric_logger = TensorboardXLogger()
        metric_logger.log = add_scalar

        self.logger['Train'][metric] = metric_logger
        self.logger['Val'][metric] = metric_logger

    def state_dict(self):
        return super().state_dict().update({
            'log_dir': self.log_dir
        })
