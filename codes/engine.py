import copy
import os

import torch

from ignite import handlers
from ignite.engine import Engine


def _sanitize_inputs(out, targets, input_percentages, target_sizes):
    seq_length = out.shape[1]
    return out.transpose(
        0, 1), targets.to('cpu'), (input_percentages *
                seq_length).int(), target_sizes.to('cpu')


def _sanitize_loss(criterion,
                   out,
                   targets,
                   input_percentages,
                   target_sizes,
                   average=1):
    out, targets, out_sizes, target_sizes = _sanitize_inputs(
        out, targets, input_percentages, target_sizes)

    loss = criterion(out, targets, out_sizes, target_sizes)
    loss = loss / average

    loss_sum = loss.sum()  # average the loss by minibatch

    inf = float("inf")
    if loss_sum == inf or loss_sum == -inf:
        print("WARNING: received an inf loss, setting loss value to 0")
        loss_sum = 0 * loss_sum

    return loss_sum


def create_trainer(model, optimizer, criterion, device, **kwargs):
    data_timer = handlers.Timer(average=True)
    max_norm = kwargs.get('max_norm', 400)
    task_weights = kwargs.get('task_weights', [1])

    is_multi_task = len(task_weights) > 1

    if is_multi_task and not isinstance(criterion, list):
        criterion = [
            copy.deepcopy(criterion) for _ in range(len(task_weights))
        ]

    def _update(engine, batch):
        if engine.skip_n > 0:
            engine.skip_n -= 1
            print(engine.state.iteration)
            return 'Skipped'

        model.train()

        engine.data_timer.resume()
        inputs, targets, input_percentages, target_sizes = batch
        if is_multi_task:
            valid_tasks = [idx for idx, i in enumerate(inputs) if i is not None]
            inputs = [inputs[idx] for idx in valid_tasks]
        else:
            inputs.to(device)
        engine.data_timer.pause()
        engine.data_timer.step()

        if is_multi_task:
            out = model(inputs, valid_tasks)
            total_loss = 0
            for i, task_out, task_input in zip(valid_tasks, out, inputs):
                task_loss = _sanitize_loss(
                    criterion[i],
                    task_out,
                    targets[i],
                    input_percentages[i],
                    target_sizes[i],
                    average=task_input.shape[0])
                total_loss += task_weights[i] * task_loss
        else:
            out = model(inputs)
            total_loss = _sanitize_loss(
                criterion[0],
                out,
                targets,
                input_percentages,
                target_sizes,
                average=inputs.shape[0])

        # compute gradient
        optimizer.zero_grad()
        total_loss.backward()

        # Clipping the norm, avoiding gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # optimizer step
        optimizer.step()

        torch.cuda.synchronize()

        return total_loss.item()

    engine = Engine(_update)
    engine.data_timer = data_timer
    engine.skip_n = kwargs.get('skip_n', 0)

    return engine


def create_evaluator(model, metrics, device=torch.device('cuda')):
    def _inference(engine, batch):
        model.eval()

        with torch.no_grad():
            inputs, targets, input_percentages, target_sizes = batch
            if isinstance(inputs, (list, tuple)):
                valid_tasks = [idx for idx, i in enumerate(inputs) if i is not None]
                inputs = [inputs[idx] for idx in valid_tasks]
                out = model(inputs, valid_tasks)
            else:
                inputs.to(device)
                out = model(inputs)  # NxTxH

            if isinstance(out, (list, tuple)):
                return {task:
                    _sanitize_inputs(out[i], targets[task], input_percentages[task],
                                     target_sizes[task]) for i, task in enumerate(valid_tasks)
                }
            else:
                return _sanitize_inputs(out, targets, input_percentages,
                                        target_sizes)


    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
