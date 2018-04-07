import numpy as np

class ClassificationDataSet(object):

    """Dataset for classification problems."""

    def __init__(self,
                 images,
                 labels,
                 label_names):
        assert images.shape[0] == labels.shape[0]
        assert images.ndim == 4
        assert labels.ndim == 1

        self.num_examples = images.shape[0]

        self.images = images
        self.labels = labels
        self.label_names = label_names
        self.epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        assert batch_size <= self.num_examples

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self.images[start:end], self.labels[start:end]