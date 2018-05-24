
import os

import torch

from ignite import handlers
from ignite.engine import Engine


def create_trainer(model, optimizer, criterion, device, **kwargs):
        data_timer = handlers.Timer(average=True)
        def _update(engine, batch):
            if engine.skip_n > 0:
                engine.skip_n -= 1
                print(engine.state.iteration)
                return 'Skipped'

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
        engine.skip_n = kwargs.get('skip_n', 0)

        return engine

def create_evaluator(model, metrics, device=torch.device('cuda')):
    def _inference(engine, batch):
        model.eval()

        with torch.no_grad():
            inputs, targets, input_percentages, target_sizes = batch
            inputs.to(device)

            out = model(inputs)  # NxTxH

            seq_length = out.shape[1]
            out_sizes = (input_percentages * seq_length).int()

            return out, targets, out_sizes, target_sizes

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
