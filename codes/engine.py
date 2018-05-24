
import os

import torch

from ignite import handlers
from ignite.engine import Engine


def create_trainer(model, optimizer, criterion, device, **kwargs):
        data_timer = handlers.Timer(average=True)
        max_norm = kwargs.get('max_norm', 400)

        task_weights = kwargs.get('task_weights', [1])
        is_multi_task = len(task_weights) > 1

        def _update(engine, batch):
            if engine.skip_n > 0:
                engine.skip_n -= 1
                print(engine.state.iteration)
                return 'Skipped'

            model.train()

            engine.data_timer.resume()
            if len(batch) == 4:
                inputs, targets, input_percentages, target_sizes = batch
                task = torch.zeros(inputs.shape[0], dtype=torch.int)
            else:
                inputs, targets, input_percentages, target_sizes, task = batch
            inputs.to(device)
            engine.data_timer.pause()
            engine.data_timer.step()

            if is_multi_task:
                out = model(inputs, task)
            else:
                out = model(inputs)

            # CTC loss is batch_first = False, i.e., T x B x D
            out = out.transpose(0, 1)

            seq_length = out.shape[0]
            out_sizes = (input_percentages * seq_length).int()

            loss = criterion(out, targets, out_sizes.to('cpu'), target_sizes.to('cpu'))

            total_loss = 0
            for i in enumerate(task_weights):
                task_idxs = task == i

                task_loss = loss[task_idxs]
                task_loss = task_loss / inputs[task_idxs].shape[0]  # average the loss by minibatch

                task_loss_sum = task_loss.sum()
                inf = float("inf")
                if task_loss_sum == inf or task_loss_sum == -inf:
                    print("WARNING: received an inf loss, setting loss value to 0")
                    task_loss = 0
                else:
                    task_loss = loss.item()

                total_loss += task_weights[i] * task_loss

            # compute gradient
            optimizer.zero_grad()
            total_loss.backward()

            # Clipping the norm, avoiding gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

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
