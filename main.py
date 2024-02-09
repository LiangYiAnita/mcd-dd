import argparse
import yaml
import torch
import numpy as np
from dataset_loader import WindowedDataset
from encoder import Encoder
from mcd import MCD
from torch.utils.data import DataLoader
import os

def main(config_file, seed=1111,device='cpu'):
    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Load configuration file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Check if dataset path exists
    if not os.path.exists(config['dataset_path']):
        raise FileNotFoundError(f"can not find dataset in {config['dataset_path']}.")

    # Load dataset
    dataset = np.load(config['dataset_path'], allow_pickle=True)  
    dataset = torch.tensor(dataset, dtype=torch.float32)  

    # Calculate required parameters based on the loaded dataset
    T = dataset.shape[0]
    col_length = dataset.shape[1]
    window_size = config['win_size']
    slide = int(window_size / config['sub_window_num'])

    for exp in range(config['n_experiments']):
        print(f"Running experiment {exp+1}/{config['n_experiments']}")
        # Resetting seed for each experiment
        np.random.seed(seed + exp)
        torch.manual_seed(seed + exp)

        windowed_dataset = WindowedDataset(dataset, window_size=window_size, slide=slide)
        dataloader = DataLoader(windowed_dataset, batch_size=1, shuffle=False)

        # Initialize your model, optimizer here
        model = Encoder(col_length, config['hidden_size'], config['output_size']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        mcd = MCD(model, optimizer, config['epochs'], config['sub_window_num'], config['m'], config['k'], config['eps_small'], config['eps_big'], config['temperature'], config['lamb'], config['percentile'], device)

        threshold = 0
        first_window = True
        # Training loop for each window of data
        for i, window_data in enumerate(dataloader):
            window_data = window_data.to(device).squeeze(0)  
            if not first_window:
                distances = mcd.test(window_data)
                if distances[-1] > threshold:
                    # Drift detection
                    drift_detected_start = max(0, i * slide + window_size - slide)
                    drift_detected_end = min(len(dataset), i * slide + window_size)
                    print(f"Drift detected between {drift_detected_start} and {drift_detected_end}")
            threshold = mcd.train(window_data) 
            first_window = False 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", type=int, default=1111, help="Random seed for reproducibility")
    parser.add_argument("--config_file", type=str,default="configs/para_simulated_data.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    main(config_file=args.config_file, seed=args.seed, device=device)
    #Use python main.py --config_file path/to/config.yaml to run the main code