import torch

class WindowedDataset(torch.utils.data.Dataset):
    """A dataset class for windowed data processing."""
    def __init__(self, dataset, window_size, slide, transform=None):
        """
        Initializes the dataset with windowing parameters.

        Args:
            dataset (torch.Tensor): The dataset to window.
            window_size (int): The size of each data window.
            slide (int): The sliding distance between windows.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset = dataset
        self.window_size = window_size
        self.slide = slide
        self.transform = transform

    def __len__(self):
        return max(1, ((len(self.dataset) - self.window_size) // self.slide) + 1)

    def __getitem__(self, idx):
        start = idx * self.slide
        end = start + self.window_size
        data = self.dataset[start:end]
        if self.transform:
            data = self.transform(data)
        return data
