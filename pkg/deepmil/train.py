from .arguments import get_arguments
from .dataloader import Dataset_handler
from .models import DeepMIL
import numpy as np

# For the sklearn warnings
import warnings
warnings.filterwarnings('always')

# timer.py
import time
class TimerError(Exception):

    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")

def writes_metrics(writer, to_write, epoch):
    for key in to_write:
        if type(to_write[key]) == dict:
            writer.add_scalars(key, to_write[key], epoch)
        else:
            writer.add_scalar(key, to_write[key], epoch)

def train(model, dataloader):
    model.network.train()
    mean_loss = []
    epobatch = 1/len(dataloader) # How many epochs per batch ?
    for input_batch, target_batch in dataloader:
        model.counter["batch"] += 1
        model.counter['epoch'] += epobatch
        [scheduler.step(model.counter['epoch']) for scheduler in model.schedulers]
        loss = model.optimize_parameters(input_batch, target_batch)
        mean_loss.append(loss)
    model.mean_train_loss = np.mean(mean_loss)
    print('train_loss: {}'.format(np.mean(mean_loss)))

def val(model, dataloader):
    model.network.eval()
    mean_loss = []
    for input_batch, target_batch in dataloader:
        target_batch = target_batch.to(model.device)
        loss = model.evaluate(input_batch, target_batch)
        mean_loss.append(loss)
    model.mean_val_loss = np.mean(mean_loss)
    to_write = model.flush_val_metrics()
    writes_metrics(model.writer, to_write, model.counter['epoch']) # Writes classif metrics.
    state = model.make_state()
    print('mean val loss {}'.format(np.mean(mean_loss)))
    model.update_learning_rate(model.mean_val_loss)
    model.early_stopping(model.args.sgn_metric * to_write[model.args.ref_metric], state)

def main():
    t = Timer()
    args = get_arguments(train=True)
    model = DeepMIL(args=args, with_data=True)
    model.get_summary_writer()
    while model.counter['epoch'] < args.epochs:
        t.start()
        print("Epochs {}".format(round(model.counter['epoch'])))
        train(model=model, dataloader=model.train_loader)
        val(model=model, dataloader=model.val_loader)
        t.stop()
        if model.early_stopping.early_stop:
            break
    model.writer.close()

