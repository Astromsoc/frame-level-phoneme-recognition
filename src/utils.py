"""
    Custom classes of dataset instances
        tailored for frame-level phoneme acquisition
"""

import os
import torch
import numpy as np
from tqdm import tqdm



class AudioDataset(torch.utils.data.Dataset):
    """
        Dataset instance for training & development sets
    """
    subfolders = ['transcript', 'mfcc']

    def __init__(
        self, 
        data_directory: str,
        phoneme_dict: dict,
        context_len: int = None
    ):
        # load data from the given data directory
        # ASSUMING that a 'transcript' and an 'mfcc' sub folder exist
        self.data_directory = data_directory
        # load the phoneme dictionary for transcript mapping
        self.phoneme_dict = phoneme_dict

        # save the (preprocessed) data into 2 list attributes
        self.features = list()
        self.labels = list()
        # use a list to store all possible index pairs
        self.index_map = list()

        # use context length to concatenate the results
        self.context_len = context_len if context_len else 0
        self.skip_interval = 2 * self.context_len + 1

        # load data and build index maps
        for i, basename in tqdm(enumerate(os.listdir(f"{self.data_directory}/mfcc"))):

            mfcc = np.load(
                os.path.join(f"{self.data_directory}/mfcc", basename)
            )
            # assuming the paired files feature the same basename (and file suffix)
            transcript = np.vectorize(self.phoneme_dict.get)(np.load(
                os.path.join(f"{self.data_directory}/transcript", basename)
            ))[1:-1]
            # trim the <sos> and <eos> tokens

            # check each input pair is aligned
            assert mfcc.shape[0] == transcript.shape[0]  
            # add index pairs to the index map
            self.index_map += [(i, j) for j in range(mfcc.shape[0])]

            # add a constant padding vector for repetitive use in the future
            if self.context_len and not hasattr(self, 'context_padding'):
                self.num_features = mfcc.shape[1]
                self.context_padding = np.zeros((self.context_len, self.num_features))
            # add context padding if required
            if hasattr(self, 'context_padding'):
                mfcc = np.vstack((self.context_padding, mfcc, self.context_padding))
                assert mfcc.shape[0] == 2 * self.context_len + transcript.shape[0]
            
            self.features.append(torch.tensor(mfcc, dtype=torch.float32))
            self.labels.append(torch.tensor(transcript, dtype=torch.long))

        # check size agreement
        assert len(self.features) == len(self.labels)
        # entire len: count of [valid] frames, suggested by index pairs
        self.dataset_size = len(self.index_map)


    def __len__(self):
        return self.dataset_size
    

    def __getitem__(self, index):
        i, j = self.index_map[index]
        return (
            self.features[i][j: j + self.skip_interval].flatten(),
            self.labels[i][j]
        )

    def update_context_len(self, new_context_len: int):
        raise NotImplementedError("Not implemented yet")



class AudioDatasetInference(torch.utils.data.Dataset):
    """
        Dataset instance for training & development sets
    """

    def __init__(
        self, 
        data_directory: str,
        context_len: int = None
    ):
        # load data from the given data directory
        # ASSUMING that a 'transcript' and an 'mfcc' sub folder exist
        self.data_directory = data_directory

        # save the (preprocessed) data into 2 list attributes
        self.features = list()
        # use a list to store all possible index pairs
        self.index_map = list()

        # use context length to concatenate the results
        self.context_len = context_len if context_len else 0
        self.skip_interval = 2 * self.context_len + 1

        # load data and build index maps
        for i, basename in tqdm(enumerate(os.listdir(f"{self.data_directory}/mfcc"))):
            mfcc = np.load(
                os.path.join(f"{self.data_directory}/mfcc", basename)
            )
            # add index pairs to the index map
            self.index_map += [(i, j) for j in range(mfcc.shape[0])]

            # add a constant padding vector for repetitive use in the future
            if self.context_len and not hasattr(self, 'context_padding'):
                self.num_features = mfcc.shape[1]
                self.context_padding = np.zeros((self.context_len, self.num_features))
            # add context padding if required
            if hasattr(self, 'context_padding'):
                mfcc = np.vstack((self.context_padding, mfcc, self.context_padding))
            self.features.append(torch.tensor(mfcc, dtype=torch.float32))

        # entire len: count of [valid] frames, suggested by index pairs
        self.dataset_size = len(self.index_map)


    def __len__(self):
        return self.dataset_size
    

    def __getitem__(self, index):
        i, j = self.index_map[index]
        return self.features[i][j: j + self.skip_interval].flatten()
            
            
    def update_context_len(self, new_context_len: int):
        raise NotImplementedError("Not implemented yet")
        


def load_phoneme_dict(
    phoneme_dict_path: str
):
    """
        Function to load phoneme dictionary
    """
    return {
        p: i for i, p in enumerate([f.strip() for f in open(phoneme_dict_path, 'r') if f])
    }


def model_weights_init(
    layer, 
    init_method: str = 'xavier'
):
    """
        Function to initialize model weights
    """
    if isinstance(layer, torch.nn.Linear):
        if init_method == 'xavier':
            torch.nn.init.xavier_uniform_(layer.weight)
        elif init_method == 'kaiming':
            torch.nn.init.kaiming_uniform_(layer.weight)
        elif init_method == 'normal':
            torch.nn.init.normal_(layer.weight)
        else:
            torch.nn.init.zeros_(layer.weight)
        # init the bias from 0: avoid disturbance / divergence
        torch.nn.init.zeros_(layer.bias)


# if __name__ == '__main__':

#     phoneme_dict = load_phoneme_dict('./src/phonemes.txt')

#     train_dataset = AudioDataset(
#         data_directory='./tiny/train-clean-100',
#         phoneme_dict=phoneme_dict,
#         context_len=2
#     )

#     dev_dataset = AudioDataset(
#         data_directory='./tiny/dev-clean',
#         phoneme_dict=phoneme_dict,
#         context_len=2
#     )

#     print(dev_dataset[0][0].shape)
#     print(dev_dataset[765])
#     print(dev_dataset[766])

