import warnings

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing.label import LabelEncoder, column_or_1d
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
class OrderedLabelEncoder(LabelEncoder):
    def fit(self, y):
        """Fit label encoder
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        Returns
        -------
        self : returns an instance of self.
        """
        y = column_or_1d(y, warn=True)
        _, idx = np.unique(y, return_index=True)
        self.classes_ = y[np.sort(idx)]
        self.map_classes_ = {c: i for i, c in enumerate(self.classes_)}
        return self

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels
        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.
        Returns
        -------
        y : array-like of shape [n_samples]
        """
        self.fit(y)

        return self.transform(y)

    def transform(self, y):
        """Transform labels to normalized encoding.
        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.
        Returns
        -------
        y : array-like of shape [n_samples]
        """
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        classes = np.unique(y)
        if len(np.intersect1d(classes, self.classes_)) < len(classes):
            diff = np.setdiff1d(classes, self.classes_)
            raise ValueError("y contains new labels: %s" % str(diff))

        return np.asarray([self.map_classes_[v] for v in y]).squeeze()
