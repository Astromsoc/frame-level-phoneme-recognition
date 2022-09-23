# set up the running environment for this project

# step 1. conda env set up
conda create -n flpr python=3.8
conda activate flpr

# step 2. install necessary packages
pip3 install -r requirements.txt

# step 3. log in for several platforms
wandb login
# get your wandb API token ready

# others: remember to copy your kaggle API key to ~/.kaggle/kaggle.json 
# as well to download datasets & submit results to the competition
