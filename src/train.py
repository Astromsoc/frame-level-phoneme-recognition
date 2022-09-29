"""
    Training driver script
"""

import sys
import yaml
import torch
import wandb
import pickle
import argparse

from src.utils import *
from src.models import MLP



# whether to connect to wandb and log the process
UPLOAD_TO_WANDB = True


"""
    Main Function Starts From Here...
"""

def main(args):

    # load configs & device info
    configs = yaml.safe_load(open(args.config_yaml_filepath, 'r'))
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    # obtain random seed
    SEED = (
        configs['training']['seed'] if 'seed' in configs['training'] 
        else 11785
    )
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # init wandb for formal training
    if UPLOAD_TO_WANDB:
        wandb.init(
            project='785hw1',
            entity='astromsoc'
        )
        # save the final best models to a folder w/ the current run name by wandb
        folder = f"checkpoints/run-{wandb.run.name}"
        os.makedirs(folder, exist_ok=True)
        # copy the configs file to target output folder for more direct access
        os.system(f"cp {args.config_yaml_filepath} {folder}/configs.yml")
        # NOTE: the configs are also saved into checkpoint files for faster internal access

    # load phoeneme dictionary
    phoneme_dict = load_phoneme_dict(configs['PHONEME_TXT_FILEPATH'])
    
    # create dataset instances
    train_dataset = AudioDataset(
        data_directory=configs['TRAIN_DATA_DIR'],
        phoneme_dict=phoneme_dict,
        context_len=configs['context_len'],
        add_powers=configs['model']['add_powers']
    )
    dev_dataset = AudioDataset(
        data_directory=configs['DEV_DATA_DIR'],
        phoneme_dict=phoneme_dict,
        context_len=configs['context_len'],
        add_powers=configs['model']['add_powers']
    )

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs['training']['batch_size'],
        num_workers=configs['device_num_workers'],
        shuffle=True,
        pin_memory=True
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=configs['training']['batch_size'],
        num_workers=configs['device_num_workers'],
        shuffle=True,
        pin_memory=True
    )

    # basic information
    print(f"\nCurrently running on [{device}]...\n")
    print(f"There are [{len(train_loader)}] batches in training set and " + 
          f"[{len(dev_loader)}] batches for evaluation.\n\n")

    # build linear layer dimension list
    input_dim = (2 * configs['context_len'] + 1) * dev_dataset.num_features
    # adding powers when necessary
    input_dim = (
        input_dim * configs['model']['add_powers'] if configs['model']['add_powers'] >= 2
        else input_dim
    )
    linear_list = [input_dim, *configs['model']['linear']]
    
    # build model
    model = MLP(
        dim_list=linear_list,
        activation_list=configs['model']['activation'],
        dropout_list=configs['model']['dropout'],
        batchnorm_list=configs['model']['batchnorm'],
        noise_level=configs['training']['noise_level']
    )

    # if there a model checkpoint exists: load from previous checkpoint
    if 'init_checkpoint' in configs['training']:
        checkpoint = torch.load(configs['training']['init_checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print("\nModel checkpoint successfully loaded " +
              f"from [{configs['training']['init_checkpoint']}]\n")
    else:
        # initialize model weights
        model.apply(lambda l: model_weights_init(l, configs['model']['init']))
    
    # modify the dropout rate if needed: for resumed training only
    if 'new_dropout_rates' in configs['training']:
        print(f"\nUpdating dropout rate for dropout layers...\n")
        dropout_count = 0
        for layer in model.streamline:
            if isinstance(layer, torch.nn.Dropout):
                layer = torch.nn.Dropout(
                    configs['training']['new_dropout_rates'][dropout_count]
                )
                dropout_count += 1
    
    # show model structure & number of parameters
    print(f"\nModel Architecture:\n{model}\n")
    print(f"**PARAMETERS**: [{model.trainable_param_count}] trainable " +
          f"out of [{model.total_param_count}] total.")

    # take model to the device
    model.to(device)
    # NOTE: safer to move now and ensure both model & optimizer on the same device

    # build the optimizer & loss function
    longterm_optimizer = OPTIMIZER_MAP.get(
        configs['optimizer']['name'], OPTIMIZER_MAP['adamw']
    )(model.parameters(), configs['optimizer']['configs'])

    # use warmup training w/ adam just for a few initial epochs
    if ('adam_warmup_epochs' in configs['optimizer'] and
        configs['optimizer']['adam_warmup_epochs']):
        optimizer = OPTIMIZER_MAP['adamw'](model.parameters(), {'lr': 0.001})
    else:
        optimizer = longterm_optimizer

    # load from previous check point if existed and asked
    if ('load_optimizer_checkpoint' in configs['training']
        and configs['training']['load_optimizer_checkpoint']):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("\nOptimizer checkpoint successfully loaded " +
              f"from [{configs['training']['init_checkpoint']}]\n")

    # load loss function
    criterion = torch.nn.CrossEntropyLoss()

    # placeholder for the scheduler variable
    scheduler = (SCHEDULER_MAP.get(
        configs['scheduler']['name'], SCHEDULER_MAP['cosine_annealing_lr']
    )(optimizer, configs['scheduler']['configs']) 
    if 'scheduler' in configs else None)

    # log hyperparams
    wandb.config = {
        'model_configs': configs['model'],
        'optimizer_configs': configs['optimizer'],
        'training_configs': configs['training']
    }

    # trivial record keeping
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

    # number of batches before a full-dev evaluation
    dev_eval_batches = int(np.ceil(
        len(train_loader) / configs['training']['eval_per_epoch']
    ))

    # switch to initial training status
    model.train()

    # train the model
    for epoch in range(num_epochs):

        print(f"\n\nRunning Epoch #{epoch + 1}...\n")
        # record loss & accuracy
        train_count = 0
        train_loss_this_epoch = 0
        train_accu_this_epoch = 0

        # switch back to original optimizer
        if ('adam_warmup_epochs' in configs['optimizer'] and
            epoch == configs['optimizer']['adam_warmup_epochs']):
            optimizer = longterm_optimizer 
            # load scheduler for later optimizer as well
            if 'scheduler' in configs:
                scheduler = SCHEDULER_MAP.get(
                    configs['scheduler']['name'], SCHEDULER_MAP['cosine_annealing_lr']
                )(optimizer, configs['scheduler']['configs'])
        
        # reinitialize the optimizer if needed
        if ('restart_optim_interval' in configs['optimizer'] and 
            epoch % configs['optimizer']['restart_optim_interval'] == 0):
            optimizer = OPTIMIZER_MAP.get(
                configs['optimizer']['name'], OPTIMIZER_MAP['adamw']
            )(model.parameters(), configs['optimizer']['configs'])

        # iterate through batches
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

            # record the per-batch training loss and accuracy
            if UPLOAD_TO_WANDB:
                wandb.log({
                    "avg_train_loss_on_batch": train_loss_this_epoch / train_count,
                    "avg_train_accu_on_batch": train_accu_this_epoch / train_count,
                    "learning_rate_in_optimizer": optimizer.param_groups[0]['lr']
                })
            

            # run evaluation on dev set
            if ((batch + 1) % dev_eval_batches == 0 or 
               ((batch + 1) == len(train_loader) and (batch + 1) % dev_eval_batches != 0)):

                # use the updated model to evaluate on dev set
                dev_count = 0
                dev_loss_this_batch = 0
                dev_accu_this_batch = 0

                # switch to the evaluation mode
                model.eval()
                model.is_training = False
                with torch.no_grad():
                    for x_and_y in dev_loader:
                        x, y = x_and_y[0].to(device), x_and_y[1].to(device)
                        y_pred = model(x)
                        loss = criterion(y_pred, y)
                        dev_loss_this_batch += loss.item() * y.shape[0]
                        dev_accu_this_batch += (y_pred.argmax(dim=1) == y).sum().item()
                        dev_count += y.shape[0]
                        # release occupied memory
                        del x, y, y_pred, loss
                        torch.cuda.empty_cache()

                # obtain avg metrics
                dev_loss_this_batch /= dev_count
                dev_accu_this_batch /= dev_count
                # add to epoch stats
                dev_loss_history[epoch].append(dev_loss_this_batch)
                dev_accu_history[epoch].append(dev_accu_this_batch)

                if UPLOAD_TO_WANDB:
                    # log to wandb
                    wandb.log({
                        "dev_loss_per_milestone": dev_loss_history[epoch][-1],
                        "dev_accu_per_milestone": dev_accu_history[epoch][-1]
                    })

                # save the best model(s)
                if dev_loss_history[epoch][-1] < best_dev_loss['dev_loss']:
                    best_dev_loss['epoch'] = epoch
                    best_dev_loss['batch'] = batch
                    best_dev_loss['dev_loss'] = dev_loss_history[epoch][-1]
                    best_dev_loss['dev_accu'] = dev_accu_history[epoch][-1]
                    best_dev_loss['model_state_dict'] = model.state_dict().copy()
                    best_dev_loss['optimizer_state_dict'] = optimizer.state_dict().copy()
                    best_dev_loss['train_loss'] = train_loss_this_epoch / train_count
                    best_dev_loss['train_accu'] = train_accu_this_epoch / train_count
                    torch.save(best_dev_loss, f'{folder}/best_dev_loss.pt')

                if dev_accu_history[epoch][-1] > best_dev_accu['dev_accu']:
                    best_dev_accu['epoch'] = epoch
                    best_dev_accu['batch'] = batch
                    best_dev_accu['dev_loss'] = dev_loss_history[epoch][-1]
                    best_dev_accu['dev_accu'] = dev_accu_history[epoch][-1]
                    best_dev_accu['model_state_dict'] = model.state_dict().copy()
                    best_dev_accu['optimizer_state_dict'] = optimizer.state_dict().copy()
                    best_dev_accu['train_loss'] = train_loss_this_epoch / train_count
                    best_dev_accu['train_accu'] = train_accu_this_epoch / train_count
                    torch.save(best_dev_accu, f'{folder}/best_dev_accu.pt')
                
                # turn back to model training mode
                model.train()
                model.is_training = True

            # print loss
            sys.stdout.write("\r" +
                f"Epoch #{epoch + 1: <3} | Batch #{batch + 1: <5}: " +
                f"avg_train_loss = {train_loss_this_epoch / train_count:.6f} | " + 
                f"avg_train_accu = {train_accu_this_epoch / train_count * 100:.4f}% || " +
                f"dev_loss = {(dev_loss_history[epoch][-1] if dev_loss_history[epoch] else 0):.6f} | " + 
                f"dev_accu = {((dev_accu_history[epoch][-1] if dev_accu_history[epoch] else 0) * 100):.4f}%")
            sys.stdout.flush()
            
        train_loss_history[epoch] = train_loss_this_epoch / train_count
        train_accu_history[epoch] = train_accu_this_epoch / train_count

        if UPLOAD_TO_WANDB:
            # record the per_epoch stats
            wandb.log({
                'train_loss_per_epoch': train_loss_history[epoch],
                'train_accu_per_epoch': train_accu_history[epoch]
            })
        
        # update learning rate & log it
        if scheduler:
            scheduler.step()
            if UPLOAD_TO_WANDB:
                wandb.log({
                    "learning_rate_in_scheduler": scheduler.get_last_lr()[0]
                })
        

    # save the loss history to log file
    pickle.dump({
        'train_loss': train_loss_history,
        'train_accu': train_accu_history,
        'dev_loss': dev_loss_history,
        'dev_accu': dev_accu_history
    }, open(f'{folder}/logs.pkl', 'wb'))


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