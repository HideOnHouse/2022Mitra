import numpy as np
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import f1_score

from utils import *


class F1(Metric):
    def __init__(self) -> None:
        super().__init__()
        self._name = 'f1-score'
        self._maximize = True

    def __call__(self, y_true, y_score):
        y_pred = np.argmax(y_score, axis=1)
        return f1_score(y_true, y_pred)


def main():
    train_dataset = GroupDataset(train=True)
    valid_dataset = GroupDataset(train=False)
    x_train, y_train = train_dataset.get_xy()
    x_valid, y_valid = valid_dataset.get_xy()
    feature_names = train_dataset.get_feature_names()

    ordinal_cols = train_dataset.df.nunique()
    ordinal_cols = ordinal_cols[ordinal_cols < 100].to_dict()
    del ordinal_cols['fwd_seg_size_min']

    cat_idxs = [idx for idx in range(
        len(feature_names)) if feature_names[idx] in ordinal_cols]
    cat_dims = [ordinal_cols[i] for i in feature_names if i in ordinal_cols]

    model = TabNetClassifier(
        cat_idxs=cat_idxs, cat_dims=cat_dims, seed=GLOBAL_SEED, device_name='cuda:2')
    model.fit(x_train, y_train, eval_set=[
              (x_valid, y_valid)], batch_size=128, num_workers=2, eval_metric=[F1])
    model.save_model("./ckpt/tabnet_group.pt")


if __name__ == '__main__':
    main()
