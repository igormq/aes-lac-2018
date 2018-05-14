
import os

import torch

from codes import metrics
from codes.utils import model_utils as mu
from ignite import handlers
from ignite.engine import Engine, Events


def trainer(model, optimizer, criterion, device, **kwargs):
        data_timer = handlers.Timer(average=True)
        def _update(engine, batch):
            model.train()

            engine.data_timer.resume()
            inputs, targets, input_percentages, target_sizes = batch
            inputs.to(device)
            engine.data_timer.pause()
            engine.data_timer.step()

            out = model(inputs)
            # CTC loss is batch_first = False, i.e., T x B x D
            out = out.transpose(0, 1)

            seq_length = out.shape[0]
            out_sizes = (input_percentages * seq_length).int()

            loss = criterion(out, targets, out_sizes.to('cpu'), target_sizes.to('cpu'))
            loss = loss / inputs.shape[0]  # average the loss by minibatch

            loss_sum = loss.sum()
            inf = float("inf")
            if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss.item()

            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            # Clipping the norm, avoiding gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), kwargs.get('max_norm', 400))

            # optimizer step
            optimizer.step()

            torch.cuda.synchronize()

            return loss_value

        engine = Engine(_update)
        engine.data_timer = data_timer

        return engine

def evaluator(model, metrics, device=torch.device('cuda')):
    def _inference(engine, batch):
        model.eval()

        with torch.no_grad():
            inputs, targets, input_percentages, target_sizes = batch
            inputs.to(device)

            inputs = torch.Tensor(inputs).to(device)

            out = model(inputs)  # NxTxH

            seq_length = out.shape[1]
            out_sizes = (input_percentages * seq_length).int()

            return out, targets, out_sizes, target_sizes

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

class Trainer(object):

    def __init__(self, model, optimizer, criterion, metrics, device=torch.device('cuda'), **kwargs):
        self._model = model
        self._optimizer = optimizer

        self._trainer = trainer(model, optimizer, criterion, device, **kwargs)
        self._evaluator = evaluator(model, metrics, device)

        self.train_loader, self._val_loader = None, None

    def attach(self, train_loader, val_loader, kwargs):
        self.train_loader = train_loader
        self.val_loader = val_loader
        start_epoch = 0

        # Sorta grad and shuffle
        if (not kwargs.no_shuffle and start_epoch != 0) or kwargs.no_sorta_grad:
            print("Shuffling batches for the following epochs")
            self.train_loader.batch_sampler.shuffle(start_epoch)

            @self._trainer.on(Events.STARTED)
            def sampler_on_started(engine):
                print("Shuffling batches for the following epochs")
                self.train_loader.batch_sampler.shuffle(start_epoch)

        # Learning rate schedule
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._optimizer, kwargs.config.training.learning_anneal)

        # Iteration logger

        batch_timer = handlers.Timer(average=True)
        batch_timer.attach(
            self._trainer,
            start=Events.EPOCH_STARTED,
            resume=Events.ITERATION_STARTED,
            pause=Events.ITERATION_COMPLETED,
            step=Events.ITERATION_COMPLETED)

        if not kwargs.silent:

            @self._trainer.on(Events.ITERATION_COMPLETED)
            def log_iteration(engine):

                if not kwargs.silent:
                    iter = (engine.state.iteration - 1) % len(self.train_loader) + 1
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_timer:.3f}\t'
                        'Data {data_timer:.3f}\t'
                        'Loss {loss:.4f}\t'.format(
                            (engine.state.epoch),
                            iter,
                            len(self.train_loader),
                            batch_timer=batch_timer.value(),
                            data_timer=engine.data_timer.value(),
                            loss=engine.state.output))

        # Epoch checkpoint
        ckpt_handler = handlers.ModelCheckpoint(
            os.path.join(kwargs.save_folder, kwargs.config.network.name),
            kwargs.config.network.name,
            save_interval=1,
            n_saved=kwargs.config.training.num_epochs)

        # best WER checkpoint
        best_ckpt_handler = handlers.ModelCheckpoint(
            os.path.join(kwargs.save_folder, kwargs.config.network.name),
            kwargs.config.network.name,
            score_function=lambda engine: engine.state.metrics['wer'],
            n_saved=5)

        @self._trainer.on(Events.EPOCH_COMPLETED)
        def trainer_epoch_completed(engine):
            self._evaluator.run(self.train_loader)
            train_metrics = self._evaluator.state.metrics
            print(''.join(
                ['Training Summary Epoch: [{0}]\t'.format(engine.state.epoch)
                ] + [
                    'Average {} {:.3f}\t'.format(name, metric)
                    for name, metric in train_metrics.items()
                ]))

            self._evaluator.run(val_loader)
            val_metrics = self._evaluator.state.metrics
            print(''.join([
                'Validation Summary Epoch: [{0}]\t'.format(engine.state.epoch)
            ] + [
                'Average {} {:.3f}\t'.format(name, metric)
                for name, metric in val_metrics.items()
            ]))

            ckpt_handler(
                engine, {'model' : {
                    'state_dict': mu.get_state_dict(self._model),
                    'args': kwargs,
                    'optimizer': self._optimizer,
                    'epoch': engine.state.epoch,
                    'iteration': engine.state.iteration,
                    'metrics': train_metrics,
                    'val_metrics': val_metrics,
                }})


            best_ckpt_handler(
                evaluator, {'model': {
                    'state_dict': mu.get_state_dict(self._model),
                    'args': kwargs,
                    'optimizer': self._optimizer,
                    'epoch': engine.state.epoch,
                    'iteration': engine.state.iteration,
                    'metrics': train_metrics,
                    'val_metrics': val_metrics,
                }})

            if not kwargs.no_shuffle:
                print("\nShuffling batches...")
                self.train_loader.batch_sampler.shuffle(engine.state.epoch)

            old_lr = kwargs.config.training.learning_rate * (
                kwargs.config.training.learning_anneal**engine.state.epoch)
            new_lr = kwargs.config.training.learning_rate * (kwargs.config.training.learning_anneal**
                                        (engine.state.epoch + 1))
            print('\nAnnealing learning rate from {:.5g} to {:5g}.\n'.format(
                old_lr, new_lr))
            scheduler.step()


    def train(self, max_epochs):
        return self._trainer.run(self.train_loader, max_epochs)

    def eval(self):
        return self._evaluator.run(self.val_loader)
