import os
import errno

from ignite.engines import Events
from tensorboardX import SummaryWriter


class TensorboardX:
    def __init__(self, loss_name, log_dir, log_iterval):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Tensorboard log directory already exists.')
                for file in os.listdir(log_dir):
                    file_path = os.path.join(log_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception:
                        raise
            else:
                raise

        self.log_iterval = log_iterval
        self.writer = SummaryWriter(log_dir=log_dir)

    def attach(self, train_engine, val_engine=None):
        train_engine.add_event_handler(self._log_loss, Events.ITERATION_COMPLETED, 'training')
        train_engine.add_event_handler(self._log_metrics, Events.EPOCH_COMPLETED, 'training')

        if val_engine is not None:
            val_engine.add_event_handler(self._log_metrics, Events.EPOCH_COMPLETED, 'validation')

    def _log_loss(self, engine, name):
        iter = (engine.state.iteration - 1) % len(engine.state.dataloader) + 1

        if iter % self.log_iterval == 0:
            self.writer.add_scalar('{}/loss'.format(name), engine.state.output, engine.state.iteration)

    def _log_metrics(self, engine, name):
        metrics = engine.state.metrics
        for metric_name, metric_val in metrics.items():
            self.writer.add_scalar("{}/avg_{}".format(name, metric_name), metric_val, engine.state.epoch)