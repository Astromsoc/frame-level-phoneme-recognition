"""
    Training driver script
"""

import sys
import yaml
import torch
import wandb
import argparse

from src.utils import *
from src.models import MLP



def main(args):
    # load configs & device info
    configs = yaml.safe_load(open(args.config_yaml_filepath, 'r'))
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"\nCurrently running on [{device}]...\n")

    # init wandb
    wandb.init(
        project='785hw1',
        entity='astromsoc'
    )

    # load phoeneme dictionary
    phoneme_dict = load_phoneme_dict(configs['PHONEME_TXT_FILEPATH'])
    
    # create dataset instances
    train_dataset = AudioDataset(
        data_directory=configs['TRAIN_DATA_DIR'],
        phoneme_dict=phoneme_dict,
        context_len=configs['context_len']
    )
    dev_dataset = AudioDataset(
        data_directory=configs['DEV_DATA_DIR'],
        phoneme_dict=phoneme_dict,
        context_len=configs['context_len']
    )

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs['training']['batch_size'],
        shuffle=True
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=configs['training']['batch_size'],
        shuffle=True
    )

    # load model configs
    input_dim = (2 * configs['context_len'] + 1) * dev_dataset.num_features
    linear_list = [input_dim] + configs['model']['linear']
    
    # build model
    model = MLP(
        dim_list=linear_list,
        activation_list=configs['model']['activation'],
        dropout_list=configs['model']['dropout'],
        batchnorm_list=configs['model']['batchnorm'],
    )
    # initialize model weights
    model.apply(lambda l: model_weights_init(l, configs['model']['init']))

    # build the optimizer & loss function
    optimizer = (
        torch.optim.Adam(model.parameters(), **configs['optimizer']['configs'])
        if configs['optimizer']['name'] == 'Adam'
        else torch.optim.AdamW(model.parameters(), **configs['optimizer']['configs'])
        if configs['optimizer']['name'] == 'AdamW'
        else torch.optim.SGD(model.parameters(), **configs['optimizer']['configs'])
        if configs['optimizer']['name'] == 'SGD'
        else torch.optim.RMSProp(model.parameters(), **configs['optimizer']['configs'])
        # default: RMSProp
    )
    criterion = torch.nn.CrossEntropyLoss()

    # log hyperparams
    wandb.config = {
        'model_configs': configs['model'],
        'optimizer_configs': configs['optimizer'],
        'training_configs': configs['training']
    }

    # take model to the device
    model.to(device)

    num_epochs = configs['training']['epochs']
    train_loss_history = np.zeros((num_epochs, ))
    train_accu_history = np.zeros((num_epochs, ))
    dev_loss_history = [list() for _ in range(num_epochs)]
    dev_accu_history = [list() for _ in range(num_epochs)]

    # save the best model
    best_dev_loss = {
        'dev_loss': float('inf'),
        'configs': configs
    }
    best_dev_accu = {
        'dev_accu': 0.0,
        'configs': configs
    }

     # train the model
    for epoch in range(num_epochs):

        print(f"\n\nRunning Epoch #{epoch + 1}...\n")

        # record loss & accuracy
        train_count = 0
        train_loss_this_epoch = 0
        train_accu_this_epoch = 0

        # iterate through batches
        model.train()
        for batch, x_and_y in enumerate(train_loader):
            # clear previous grads
            optimizer.zero_grad()
            x, y = x_and_y[0].to(device), x_and_y[1].to(device)
            # compute loss
            y_pred = model(x)
            loss = criterion(y_pred, y)
            # compute derivatives
            loss.backward()
            # update parameters
            optimizer.step()

            # record per-batch stats
            train_loss_this_batch = loss.item()
            train_count_this_batch = y.shape[0]
            label_pred = y_pred.argmax(dim=1)
            train_accu_this_batch = (label_pred == y).sum().item()

            # update per-epoch stats
            train_loss_this_epoch += train_loss_this_batch * train_count_this_batch
            train_accu_this_epoch += train_accu_this_batch 
            train_count += train_count_this_batch

            # obtain the avg stats for training in this batch
            train_accu_this_batch /= train_count_this_batch

            # use the updated model to evaluate on dev set
            dev_count = 0
            dev_loss_this_batch = 0
            dev_accu_this_batch = 0

            model.eval()
            with torch.no_grad():
                for x_and_y in dev_loader:
                    x, y = x_and_y[0].to(device), x_and_y[1].to(device)
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    dev_loss_this_batch += loss.item() * y.shape[0]
                    dev_accu_this_batch += (y_pred.argmax(dim=1) == y).sum().item()
                    dev_count += y.shape[0]
            # obtain avg metrics
            dev_loss_this_batch /= dev_count
            dev_accu_this_batch /= dev_count
            # add to epoch stats
            dev_loss_history[epoch].append(dev_loss_this_batch)
            dev_accu_history[epoch].append(dev_accu_this_batch)
            
            # print loss
            sys.stdout.write(
                "\r" +
                f"Epoch #{epoch + 1: <3} | Batch #{batch + 1: <5}: " +
                f"train_loss = {train_loss_this_batch:.6f} | " + 
                f"train_accu = {train_accu_this_batch * 100:.4f}% || " +
                f"dev_loss = {dev_loss_history[epoch][-1]:.6f} | " + 
                f"dev_accu = {dev_accu_history[epoch][-1] * 100:.4f}%"
            )
            sys.stdout.flush()

            # log to wandb
            wandb.log({
                "train_loss_per_batch": train_loss_this_batch,
                "train_accu_per_batch": train_accu_this_batch,
                "dev_loss_per_batch": dev_loss_history[epoch][-1],
                "dev_accu_per_batch": dev_accu_history[epoch][-1]
            })

            # save the best model(s)
            if dev_loss_history[epoch][-1] < best_dev_loss['dev_loss']:
                best_dev_loss['epoch'] = epoch
                best_dev_loss['batch'] = batch
                best_dev_loss['dev_loss'] = dev_loss_history[epoch][-1]
                best_dev_loss['dev_accu'] = dev_accu_history[epoch][-1]
                best_dev_loss['model_state_dict'] = model.state_dict().copy()
                best_dev_loss['optimizer_state_dict'] = optimizer.state_dict().copy()
                best_dev_loss['train_loss'] = train_loss_this_batch
                best_dev_loss['train_accu'] = train_accu_this_batch

            if dev_accu_history[epoch][-1] > best_dev_accu['dev_accu']:
                best_dev_accu['epoch'] = epoch
                best_dev_accu['batch'] = batch
                best_dev_accu['dev_loss'] = dev_loss_history[epoch][-1]
                best_dev_accu['dev_accu'] = dev_accu_history[epoch][-1]
                best_dev_accu['model_state_dict'] = model.state_dict().copy()
                best_dev_accu['optimizer_state_dict'] = optimizer.state_dict().copy()
                best_dev_accu['train_loss'] = train_loss_this_batch
                best_dev_accu['train_accu'] = train_accu_this_batch

        train_loss_history[epoch] = train_loss_this_epoch / train_count
        train_accu_history[epoch] = train_accu_this_epoch / train_count

    # save the final best models to a folder w/ the current run name by wandb
    folder = f"checkpoints/run-{wandb.run.name}"
    os.makedirs(folder, exist_ok=True)
    torch.save(best_dev_loss, f'{folder}/best_dev_loss.pt')
    torch.save(best_dev_accu, f'{folder}/best_dev_accu.pt')




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training a MLP using given configurations.")

    parser.add_argument(
        '--config-yaml-filepath',
        type=str,
        default='./configs/sample_config.yml',
        help='Path to the configuration file.'
    )
    args = parser.parse_args()

    main(args)