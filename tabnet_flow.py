import sys

import numpy as np
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import f1_score, classification_report

from utils import *


class F1(Metric):
    def __init__(self) -> None:
        super().__init__()
        self._name = 'f1-score'
        self._maximize = True

    def __call__(self, y_true, y_score):
        y_pred = np.argmax(y_score, axis=1)
        return f1_score(y_true, y_pred)


def main(args):
    train = True if args[1] == 'train' else False
    train_dataset = FlowDataset(train=True)
    valid_dataset = FlowDataset(train=False)
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
        cat_idxs=cat_idxs, cat_dims=cat_dims, seed=GLOBAL_SEED, device_name='cuda:1')
    if train:
        print("Training model")
        model.fit(x_train, y_train, eval_set=[
                (x_valid, y_valid)], batch_size=128, num_workers=2, eval_metric=[F1])
        model.save_model("./ckpt/tabnet_flow.pt")
    else:
        model.load_model("./ckpt/tabnet_flow.pt.zip")
    print("Validate Model")
    pred = model.predict(x_valid)
    f1 = f1_score(y_valid, pred)
    print(f"f1-score: {f1:.5f}")
    print(classification_report(y_valid, pred))


if __name__ == '__main__':
    main(sys.argv)
