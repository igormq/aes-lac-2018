""" Logging to Visdom server

Code adapted from pytorch/tnt repo
"""

import numpy as np

from visdom import Visdom as Viz

from .logger import BaseLogger


class BaseVisdomLogger(object):
    '''
        The base class for logging output to Visdom.

        ***THIS CLASS IS ABSTRACT AND MUST BE SUBCLASSED***

        Note that the Visdom server is designed to also handle a server architecture,
        and therefore the Visdom server must be running at all times. The server can
        be started with
        $ python -m visdom.server
        and you probably want to run it from screen or tmux.
    '''

    @property
    def viz(self):
        return self._viz

    def __init__(self,
                 fields=None,
                 win=None,
                 env=None,
                 opts={},
                 port=8097,
                 server="localhost"):
        super().__init__(fields)
        self.win = win
        self.env = env
        self.opts = opts
        self._viz = Viz(server="http://" + server, port=port)

    def log(self, *args, **kwargs):
        raise NotImplementedError(
            "log not implemented for BaseVisdomLogger, which is an abstract class."
        )

    def _viz_prototype(self, vis_fn):
        ''' Outputs a function which will log the arguments to Visdom in an appropriate way.

            Args:
                vis_fn: A function, such as self.vis.image
        '''

        def _viz_logger(*args, **kwargs):
            self.win = vis_fn(
                *args, win=self.win, env=self.env, opts=self.opts, **kwargs)

        return _viz_logger

    def save(self, *args, **kwargs):
        self.viz.save(self.env)


class VisdomGenericLogger(BaseVisdomLogger):
    '''
        A generic Visdom class that works with the majority of Visdom plot types.
    '''

    def __init__(self,
                 plot_type,
                 fields=None,
                 win=None,
                 env=None,
                 opts={},
                 port=8097,
                 server="localhost"):
        '''
            Args:
                fields: Currently unused
                plot_type: The name of the plot type, in Visdom

            Examples:
                >>> # Image example
                >>> img_to_use = skimage.data.coffee().swapaxes(0,2).swapaxes(1,2)
                >>> image_logger = VisdomGenericLogger('image')
                >>> image_logger.log(img_to_use)

                >>> # Histogram example
                >>> hist_data = np.random.rand(10000)
                >>> hist_logger = VisdomGenericLogger('histogram', , opts=dict(title='Random!', numbins=20))
                >>> hist_logger.log(hist_data)
        '''
        super().__init__(fields, win, env, opts, port, server)
        self.plot_type = plot_type
        self.chart = getattr(self.viz, plot_type)
        self.viz_logger = self._viz_prototype(self.chart)

    def log(self, *args, **kwargs):
        self.viz_logger(*args, **kwargs)


class VisdomPlotLogger(BaseVisdomLogger):
    def __init__(self,
                 plot_type,
                 fields=None,
                 win=None,
                 env=None,
                 opts={},
                 port=8097,
                 server="localhost",
                 name=None):
        '''
            Multiple lines can be added to the same plot with the "name" attribute (see example)
            Args:
                fields: Currently unused
                plot_type: {scatter, line}

            Examples:
                >>> scatter_logger = VisdomPlotLogger('line')
                >>> scatter_logger.log(stats['epoch'], loss_metric.value()[0], name="train")
                >>> scatter_logger.log(stats['epoch'], loss_metric.value()[0], name="test")
        '''
        super().__init__(fields, win, env, opts, port, server)
        valid_plot_types = {"scatter": self.viz.scatter, "line": self.viz.line}
        self.plot_type = plot_type
        # Set chart type
        if plot_type not in valid_plot_types.keys():
            raise ValueError(
                "plot_type \'{}\' not found. Must be one of {}".format(
                    plot_type, valid_plot_types.keys()))
        self.chart = valid_plot_types[plot_type]

    def log(self, *args, **kwargs):
        if self.win is not None and self.viz.win_exists(
                win=self.win, env=self.env):
            if len(args) != 2:
                raise ValueError(
                    "When logging to {}, must pass in x and y values (and optionally z).".
                    format(type(self)))
            x, y = args
            self.chart(
                X=np.array([x]),
                Y=np.array([y]),
                update='append',
                win=self.win,
                env=self.env,
                opts=self.opts,
                **kwargs)
        else:
            if self.plot_type == 'scatter':
                chart_args = {'X': np.array([args])}
            else:
                chart_args = {
                    'X': np.array([args[0]]),
                    'Y': np.array([args[1]])
                }
            self.win = self.chart(
                win=self.win, env=self.env, opts=self.opts, **chart_args)
            # For some reason, the first point is a different trace. So for now
            # we can just add the point again, this time on the correct curve.
            self.log(*args, **kwargs)


class Visdom(BaseLogger):
    ''' A class to package and visualize metrics.
    Args:
        title: The title of the Visdom. This will be used as a prefix for all plots.
        server: The uri of the Visdom server
        env: Visdom environment to log to.
        port: Port of the visdom server.
    '''

    def __init__(self, title, server="localhost", env='main', port=8097):
        super().__init__()
        self.server = server
        self.port = port
        self.env = env
        self.title = title

    def _add_logger(self, metric):
        opts = {'title': self.title + ' ' + metric}
        self.logger['Train'][metric] = VisdomPlotLogger(
            'line',
            env=self.env,
            server=self.server,
            port=self.port,
            opts=opts)
        self.logger['Val'][metric] = self.logger['Train'][metric]

    def state_dict(self):
        return super().state_dict().update({
            'server': self.server,
            'port': self.port,
            'env': self.env,
            'title': self.title
        })
