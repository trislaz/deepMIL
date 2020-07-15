from deepmil-trela.arguments import get_arguments
from deepmil-trela.dataloader import Dataset_handler
from deepmil-trela.models import DeepMIL
import numpy as np

# For the sklearn warnings
import warnings
warnings.filterwarnings('always')

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
        loss = model.optimize_parameters(input_batch, target_batch)
        model.writer.add_scalar("Training_batch_loss", loss, model.counter['epoch'])
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
    model.early_stopping(-to_write['accuracy'], state)

def main():
    args = get_arguments(train=True)
    model = DeepMIL(args=args)
    model.get_summary_writer()
    data = Dataset_handler(args)
    dataloader_train, dataloader_val = data.get_loader(training=True)
    model.target_correspondance = dataloader_train.dataset.target_correspondance # Will be useful when writing the results. TODO change that.
    while model.counter['epoch'] < args.epochs:
        print("Epochs {}".format(round(model.counter['epoch'])))
        train(model=model, dataloader=dataloader_train)
        val(model=model, dataloader=dataloader_val)
        if model.early_stopping.early_stop:
            break
    model.writer.close()

