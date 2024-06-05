# MCD-DD
Primary code and datasets for the paper: _Online Drift Detection with Maximum Concept Discrepancy_

## Data sets
All datasets are located in the `data` folder, with synthetic simulated data in `data/toy_data`, including 7 datasets such as GM_Sud, GM_Rec, etc. Descriptions of the synthetic datasets can be found in the paper's appendix. <br>
Real-world datasets are in `data/real_world_data`, and the specific data meanings are described in the paper's appendix and the following websites: [INSECTS](https://sites.google.com/view/uspdsrepository) and [EEG](https://archive.ics.uci.edu/dataset/264/eeg+eye+state).

## Code Information
All code is written in Python 3.10, utilizing PyTorch 2.1.0+cu122. The functionality of each section is as follows: <br>
The `configs` folder contains parameters that need to be specified, such as the dataset path and the logic for constructing positive and negative sample pairs. For detailed parameter settings, please refer to the paper. <br>
The `dataset_loader.py` file loads data according to the sliding window logic. <br>
`mcd.py` defines the main functions of the MCD-DD method. <br>
`encoder.py` outlines the form of the sample set encoder, which can be modified.

## How to run
After specifying all parameters in the YAML file, we run the program using the code below to conduct online drift detection on a specific dataset.
```python
python main.py --config_file path/to/config.yaml
```

