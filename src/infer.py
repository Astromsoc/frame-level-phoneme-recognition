"""
    Inference driver script
"""

import torch
import argparse
from tqdm import tqdm

from src.utils import *
from src.models import MLP



def main(args):
    # load the torch checkpoint
    checkpoint = torch.load(args.checkpoint_filepath)
    # load relevant configs
    configs = checkpoint['configs']

    # load test dataset
    test_dataset = AudioDatasetInference(
        data_directory=args.test_data_dir,
        context_len=configs['context_len']
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=configs['inference']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    ) 

    # calculate input dimension & model linear list
    input_dim = (2 * configs['context_len'] + 1) * test_dataset.num_features
    linear_list = [input_dim] + configs['model']['linear']
    
    # build model
    model = MLP(
        dim_list=linear_list,
        activation_list=configs['model']['activation'],
        dropout_list=configs['model']['dropout'],
        batchnorm_list=configs['model']['batchnorm'],
    )
    # load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # check model device
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"\nNow running on [{device}]...\n")
    model.to(device)


    # build output filepath under the checkpoint directory
    output_filepath = args.checkpoint_filepath.replace(
        '.pt', '_output.csv'
    )

    pred_count = 0
    # start inference
    model.eval()
    # write to output file as predicting
    with open(output_filepath, 'a') as fw:
        fw.write("id,label\n")
        # turn off the grad updates
        with torch.no_grad():
            for x in tqdm(test_dataloader):
                # take inputs to the device
                x = x.to(device)
                y_raws = model(x)
                # obtain the predicted phoneme index
                y_preds = torch.argmax(y_raws, dim=1)
                # write to output file using given format
                for yp in y_preds:
                    fw.write(f"{pred_count},{yp}\n")
                    pred_count += 1

        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Giving frame-level phoneme prediction using trained MLPs.")

    parser.add_argument(
        '--checkpoint-filepath', 
        type=str, 
        default='checkpoints/run-silver-snowball-33/best_dev_loss.pt',
        help="Path to the model checkpoint."
    )
    parser.add_argument(
        '--test-data-dir', 
        type=str, 
        default='data/test-clean',
        help="Directory of the test subset."
    )
    args = parser.parse_args()

    main(args)