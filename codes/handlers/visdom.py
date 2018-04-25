import numpy as np
from visdom import Visdom as Viz

from ignite.engines import Events


class Visdom:
    def __init__(self, name, log_iterval):
        self.log_iterval = log_iterval
        self.viz = Viz()
        self.viz_windows = {}

    def attach(self, train_engine, val_engine=None):
        train_engine.add_event_handler(Events.STARTED, self._fetch_loss_window)
        train_engine.add_event_handler(Events.ITERATION_COMPLETED,
                                       self._log_loss, 'training')

        train_engine.add_event_handler(Events.STARTED,
                                       self._fetch_metrics_windows, 'training')
        train_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                       self._log_metrics, 'training')

        if val_engine is not None:
            train_engine.add_event_handler(
                Events.STARTED, self._fetch_metrics_windows, 'validation')
            val_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                         self._log_metrics, 'validation')

    def _log_loss(self, engine, name):
        iter = (engine.state.iteration - 1) % len(engine.state.dataloader) + 1

        if iter % self.log_iterval == 0:
            self.viz.line(
                X=np.array([engine.state.iteration]),
                Y=np.array([engine.state.output]),
                update='append',
                win=self.viz_windows['loss'])

    def _log_metrics(self, engine, name):
        metrics = engine.state.metrics
        for metric_name, metric_val in metrics.items():
            self.viz.line(
                X=np.array([engine.state.epoch]),
                Y=np.array([metric_val]),
                win=self.viz_windows[metric_name],
                update='append',
                name='{}_{}'.format(name, metric_name))

    def _fetch_loss_window(self, engine):
        self.viz_windows['loss'] = self.viz.line(
            X=np.array([1]),
            Y=np.array([np.nan]),
            opts=dict(
                xlabel='#Iterations', ylabel='Loss', title='Training Loss'))

    def _fetch_metrics_windows(self, engine, name):
        metrics = engine.state.metrics

        for metric_name in metrics:
            if metric_name not in self.viz_windows:
                self.viz_windows[metric_name] = self.viz.line(
                    X=np.array([1]),
                    Y=np.array([np.nan]),
                    opts=dict(
                        xlabel='#Epochs', ylabel=metric_name, showlegend=True),
                    name='{}_{}'.format(name, metric_name))
