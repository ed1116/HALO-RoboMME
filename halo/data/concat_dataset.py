import torch
from typing import List, Dict, Any


class CustomConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, *args, **kwargs):
        self.weight_by_dataset = kwargs.pop("weight_by_dataset", None)
        super(CustomConcatDataset, self).__init__(*args, **kwargs)
        if self.weight_by_dataset is None:
            self.weight_by_dataset = [1.0] * len(self.datasets)
        assert len(self.weight_by_dataset) == len(self.datasets), "Weight by dataset must be the same length as the number of datasets"
        self._balanced_sampling_weights = self.weights_for_balanced_classes()

    
    @property
    def balanced_sampling_weights(self):
        return self._balanced_sampling_weights

    
    def save_split(self, *args, **kwargs):
        for dataset in self.datasets:
            dataset.save_split(*args, **kwargs)
        return


    def shuffle_dataset(self, seed=0):
        """
        Shuffle the dataset according to the seed
        """
        for dataset in self.datasets:
            dataset.shuffle_dataset(seed=seed)
        # after shuffling, the dataset size might change.
        self.cumulative_sizes = self.cumsum(self.datasets)
        self._balanced_sampling_weights = self.weights_for_balanced_classes()
        return


    def weights_for_balanced_classes(self):
        """
        Return the weights for balanced classes
            - Weights are inversely proportional to the size of the dataset
        """
        weights = [(1.0/len(dataset)) * w for dataset, w in zip(self.datasets, self.weight_by_dataset) for _ in range(len(dataset))]
        return weights


if __name__ == "__main__":
    import numpy as np
    # create a fake datasets and custom concat dataset.
    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, size: int, value: int):
            self.size = size
            self.value = value
        def __len__(self):
            return self.size
        def __getitem__(self, index: int):
            return torch.tensor(self.value)

    # load the sampler from halo
    from halo.util.misc import DistributedWeightedSubEpochSampler

    dataset1 = FakeDataset(1000, 1)
    dataset2 = FakeDataset(2000, 2)
    dataset3 = FakeDataset(3000, 3)
    custom_concat_dataset = CustomConcatDataset([dataset1, dataset2, dataset3])
    sampler = DistributedWeightedSubEpochSampler(custom_concat_dataset, num_replicas=1, rank=0, shuffle=True, split_epoch=1, seed=0, weights=custom_concat_dataset.balanced_sampling_weights)
    dataloader = torch.utils.data.DataLoader(custom_concat_dataset, sampler=sampler, batch_size=1000)
    # dataloader = torch.utils.data.DataLoader(custom_concat_dataset, batch_size=1000, shuffle=True)

    for batch in dataloader:
        value1 = torch.sum(batch == 1)
        value2 = torch.sum(batch == 2)
        value3 = torch.sum(batch == 3)
        print(f"Value 1: {value1}, Value 2: {value2}, Value 3: {value3}")
        