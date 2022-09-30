"""
    Custom classes of dataset instances
        tailored for frame-level phoneme acquisition
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter



"""
    Mapping from Configuration Specifications
        to Corresponding Classes / Functions
"""

WEIGHT_INIT_MAP = {
    'xavier': torch.nn.init.xavier_uniform_,
    'kaiming': torch.nn.init.kaiming_uniform_,
    'normal': torch.nn.init.normal_,
    'zeros': torch.nn.init.zeros_
}

# dict to build optimizer based on config specifications
OPTIMIZER_MAP = {
    'adam': lambda prm, cfg: torch.optim.Adam(prm, **cfg),
    'adamw': lambda prm, cfg: torch.optim.AdamW(prm, **cfg),
    'adagrad': lambda prm, cfg: torch.optim.Adagrad(prm, **cfg),
    'nadam': lambda prm, cfg: torch.optim.NAdam(prm, **cfg),
    'sgd': lambda prm, cfg: torch.optim.SGD(prm, **cfg),
    'rmsprop': lambda prm, cfg: torch.optim.RMSprop(prm, **cfg)
}

# dict to build scheduler based on config specifications
SCHEDULER_MAP = {
    'cosine_annealing_warm_restarts': lambda opt, cfg: 
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, **cfg),
    'cosine_annealing_lr': lambda opt, cfg: 
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, **cfg),
    'reduce_lr_on_plateau': lambda opt, cfg: 
        torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **cfg),
    'exponential_lr': lambda opt, cfg: 
        torch.optim.lr_scheduler.ExponentialLR(opt, **cfg),
    'one_cycle_lr': lambda opt, cfg: 
        torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **cfg),
    'multiplicative_lr': lambda opt, cfg: 
        torch.optim.lr_scheduler.MultiplicativeLR(opt, **cfg),
    'multistep_lr': lambda opt, cfg: 
        torch.optim.lr_scheduler.MultiStepLR(opt, **cfg),
    'lambda_lr': lambda opt, cfg: 
        torch.optim.lr_scheduler.LambdaLR(opt, **cfg),
    'cyclic_lr': lambda opt, cfg: 
        torch.optim.lr_scheduler.CyclicLR(opt, **cfg)
}



class AudioDataset(torch.utils.data.Dataset):
    """
        Dataset instance for training & development sets
    """
    subfolders = ['transcript', 'mfcc']

    def __init__(
        self, 
        data_directory: str,
        phoneme_dict: dict,
        context_len: int=None,
        add_powers: int=0,
        add_log:bool=False,
        show_label_stats: bool=False
    ):
        # load data from the given data directory
        # ASSUMING that a 'transcript' and an 'mfcc' sub folder exist
        self.data_directory = data_directory
        # load the phoneme dictionary for transcript mapping
        self.phoneme_dict = phoneme_dict
        
        # record the pow count if needed
        self.add_powers = int(add_powers) if add_powers >= 2 else 0
        self.add_log = add_log

        # save the (preprocessed) data into 2 list attributes
        self.features = list()
        self.labels = list()
        # use a list to store all possible index pairs
        self.index_map = list()

        # use context length to concatenate the results
        self.context_len = context_len if context_len else 0
        self.window_len = 2 * self.context_len + 1

        # initialize counters for basic stats on inputs
        self.label_counter = Counter()

        # load data and build index maps
        for i, basename in tqdm(enumerate(os.listdir(f"{self.data_directory}/mfcc"))):

            mfcc = np.load(
                os.path.join(f"{self.data_directory}/mfcc", basename)
            )
            # assuming the paired files feature the same basename (and file suffix)
            transcript = np.vectorize(self.phoneme_dict.get)(np.load(
                os.path.join(f"{self.data_directory}/transcript", basename)
            ))[1:-1] # trim the <sos> and <eos> tokens
            # update the counts by phoneme index
            self.label_counter.update(Counter(transcript))

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
            # convert to torch tensor
            mfcc = torch.tensor(mfcc, dtype=torch.float32)
            # add powers to mfcc if specified
            if self.add_powers:
                mfcc = torch.hstack((mfcc, *(torch.pow(mfcc, i) for i in range(2, self.add_powers + 1))))
            
            # add processed results to internal data storage
            self.features.append(mfcc)
            self.labels.append(torch.tensor(transcript, dtype=torch.long))

        # check size agreement
        assert len(self.features) == len(self.labels)
        # entire len: count of [valid] frames, suggested by index pairs
        self.dataset_size = len(self.index_map)

        # show the result of counter stats
        if show_label_stats:
            for k, v in self.label_counter.items():
                print(f"{k:>4}: {v}")


    def __len__(self):
        return self.dataset_size
    

    def __getitem__(self, index):
        i, j = self.index_map[index]
        return (
            self.features[i][j: j + self.window_len],
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
        context_len: int=None,
        add_powers: int=0,
        add_log: bool=False
    ):
        # load data from the given data directory
        # ASSUMING that a 'transcript' and an 'mfcc' sub folder exist
        self.data_directory = data_directory

        # record the pow count if needed
        self.add_powers = add_powers if add_powers >= 2 else 0
        self.add_log = add_log

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
            # convert to torch tensor
            mfcc = torch.tensor(mfcc, dtype=torch.float32)
            # add powers to mfcc if specified
            if self.add_powers:
                mfcc = torch.hstack((mfcc, *(torch.pow(mfcc, i) for i in range(2, self.add_powers + 1))))

        # entire len: count of [valid] frames, suggested by index pairs
        self.dataset_size = len(self.index_map)


    def __len__(self):
        return self.dataset_size
    

    def __getitem__(self, index):
        i, j = self.index_map[index]
        return (
            self.features[i][j: j + self.window_len],
            self.labels[i][j]
        )
            
            
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
        # init the weight given init_method; default: xavier
        WEIGHT_INIT_MAP.get(init_method, 'xavier')(layer.weight)
        # init the bias from 0: avoid disturbance / divergence
        torch.nn.init.zeros_(layer.bias)
