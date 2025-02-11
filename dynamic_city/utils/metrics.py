import torch
import torch.distributed as dist

import dynamic_city.utils.constants as C
from dynamic_city.utils.dist_utils import distributed


class Metrics:
    def __init__(self, n_classes, device):
        self.n_classes = n_classes
        self.device = device
        self.cls_tp = self.cls_fp = self.cls_fn = None
        self.bin_tp = self.bin_fp = self.bin_fn = None
        self.reset()

    def reset(self):
        self.cls_tp = torch.zeros(self.n_classes, dtype=torch.int64, device=self.device)
        self.cls_fp = torch.zeros(self.n_classes, dtype=torch.int64, device=self.device)
        self.cls_fn = torch.zeros(self.n_classes, dtype=torch.int64, device=self.device)

        self.bin_tp = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.bin_fp = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.bin_fn = torch.zeros(1, dtype=torch.int64, device=self.device)

    def update(self, pred, gt):
        pred = pred.reshape(-1)
        gt = gt.reshape(-1)

        mask = pred == gt
        tp = torch.bincount(pred[mask], minlength=self.n_classes)
        fp = torch.bincount(pred, minlength=self.n_classes) - tp
        fn = torch.bincount(gt, minlength=self.n_classes) - tp

        self.cls_tp += tp
        self.cls_fp += fp
        self.cls_fn += fn

        bin_pred = pred.clone()
        bin_gt = gt.clone()
        bin_pred[bin_pred != 0] = 1
        bin_gt[bin_gt != 0] = 1
        bin_mask = bin_pred == bin_gt
        bin_tp = torch.bincount(bin_pred[bin_mask], minlength=2)
        bin_fp = torch.bincount(bin_pred, minlength=2) - bin_tp
        bin_fn = torch.bincount(bin_gt, minlength=2) - bin_tp

        self.bin_tp += bin_tp[1]
        self.bin_fp += bin_fp[1]
        self.bin_fn += bin_fn[1]

        return self.get_metrics(
            tp, fp, fn,
            torch.tensor([bin_tp[1]]), torch.tensor([bin_fp[1]]), torch.tensor([bin_fn[1]])
        )

    def get_metrics(self, tp=None, fp=None, fn=None, bin_tp=None, bin_fp=None, bin_fn=None, mask=False):
        tp = tp if tp is not None else self.cls_tp
        fp = fp if fp is not None else self.cls_fp
        fn = fn if fn is not None else self.cls_fn
        bin_tp = bin_tp if bin_tp is not None else self.bin_tp
        bin_fp = bin_fp if bin_fp is not None else self.bin_fp
        bin_fn = bin_fn if bin_fn is not None else self.bin_fn

        cls_iou = (tp / (tp + fp + fn + C.EPSILON)).cpu().numpy()

        if mask:
            cls_iou_mask = ((tp + fn) != 0).cpu().numpy()[1:]
            miou = cls_iou[1:][cls_iou_mask].mean()
        else:
            miou = cls_iou[1:].mean()

        bin_iou = (bin_tp / (bin_tp + bin_fp + bin_fn + C.EPSILON)).cpu().numpy().item()
        return {
            'cls_iou': cls_iou * 100.,
            'miou': miou * 100.,
            'bin_iou': bin_iou * 100.,
        }

    def get_metrics_dist(self):
        tp_all = self.cls_tp.clone()
        fp_all = self.cls_fp.clone()
        fn_all = self.cls_fn.clone()
        bin_tp_all = self.bin_tp.clone()
        bin_fp_all = self.bin_fp.clone()
        bin_fn_all = self.bin_fn.clone()

        if distributed():
            dist.all_reduce(tp_all, op=dist.ReduceOp.SUM)
            dist.all_reduce(fp_all, op=dist.ReduceOp.SUM)
            dist.all_reduce(fn_all, op=dist.ReduceOp.SUM)
            dist.all_reduce(bin_tp_all, op=dist.ReduceOp.SUM)
            dist.all_reduce(bin_fp_all, op=dist.ReduceOp.SUM)
            dist.all_reduce(bin_fn_all, op=dist.ReduceOp.SUM)

        return self.get_metrics(tp_all, fp_all, fn_all, bin_tp_all, bin_fp_all, bin_fn_all, mask=True)
