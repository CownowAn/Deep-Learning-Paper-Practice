# coding=utf-8
import numpy as np
import torch
from setting import args


class PrototypicalBatchSampler(object):

    def __init__(self, labels, Nc, num_samples, iterations):
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.Nc = Nc
        self.num_samples = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        self.index = range(len(self.labels))
        self.indices_by_class = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indices_by_class = torch.Tensor(self.indices_by_class)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            class_idx = np.argwhere(self.classes == label).item()
            self.indices_by_class[class_idx, np.where(np.isnan(self.indices_by_class[class_idx]))[0][0]] = idx
            self.numel_per_class[class_idx] += 1

    def __iter__(self):
        for it in range(self.iterations):
            batch_size = self.num_samples * self.Nc
            batch = torch.LongTensor(batch_size)
            selected_class_indices = torch.randperm(len(self.classes))[:self.Nc]
            for i, c in enumerate(self.classes[selected_class_indices]):
                batch_indices = slice(i * self.num_samples, (i + 1) * self.num_samples)
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_indices = torch.randperm(self.numel_per_class[label_idx])[:self.num_samples]
                batch[batch_indices] = self.indices_by_class[label_idx][sample_indices]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        return self.iterations