import numpy as np


class Metric:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        self.cur_tp = 0
        self.cur_tn = 0
        self.cur_fp = 0
        self.cur_fn = 0

    def update(self, pred, label):
        tp = ((pred >= 0.5) & (label == 1)).astype(np.int32).sum()
        tn = ((pred < 0.5) & (label == 0)).astype(np.int32).sum()
        fp = ((pred < 0.5) & (label == 1)).astype(np.int32).sum()
        fn = ((pred >= 0.5) & (label == 0)).astype(np.int32).sum()

        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn

        self.cur_tp = tp
        self.cur_tn = tn
        self.cur_fp = fp
        self.cur_fn = fn

    def metric(self):
        return self.calulate_metric(self.tp, self.tn, self.fp, self.fn)

    def cur_metric(self):
        return self.calulate_metric(self.cur_tp, self.cur_tn, self.cur_fp, self.cur_fn)

    def calulate_metric(self, tp, tn, fp, fn):
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        fpr = fp / (tn + fp + 1e-10)

        return {'precision': precision,
                'recall': recall,
                'f1': f1,
                'fpr': fpr}