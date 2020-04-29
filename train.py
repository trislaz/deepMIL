from arguments import get_arguments
from dataloader import make_loaders
from models import DeepMIL
import sklearn.metrics as metrics
import numpy as np

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
    model.mean_train_loss = np.mean(loss)

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
    model.update_learning_rate(model.mean_val_loss)
    model.early_stopping(model.mean_val_loss, state)

def main():
    args = get_arguments()
    model = DeepMIL(args=args)
    dataloader_train, dataloader_val = make_loaders(args)
    model.target_correspondance = dataloader_train.dataset.target_correspondance # Will be useful when writing the results.

    while model.counter['epoch'] < args.epochs:
        print("Begin training")
        train(model=model, dataloader=dataloader_train)
        val(model=model, dataloader=dataloader_val)
        if model.early_stopping.early_stop:
            break
    model.writer.close()

if __name__ == '__main__':
    main()

