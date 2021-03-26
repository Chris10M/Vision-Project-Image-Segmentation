import torch
import collections
import torch.nn.functional as F
import numpy as np
import cv2
import utils
from model import model
from arg_parser import evaluate

from tabulate import tabulate
from sklearn import metrics
from torch.utils.data import DataLoader
from cityscapes import CityScapes
from tqdm import tqdm


class Evaluate:
    def __init__(self, dataset, net):
        self.net = net
        self.ds = dataset

        self.n_classes = dataset.n_classes

        self.steps = 0
        self.metrics = collections.defaultdict(dict)
        self.ROC = collections.defaultdict(dict)
        self.auc = collections.defaultdict(float)
        self.jaccard_similarity = collections.defaultdict(float)
        self.dice_score = collections.defaultdict(float)

        self.class_metrics = dict()
        self.macro_metrics = dict()

    def __call__(self, im, lb):
        self.net.eval()

        with torch.no_grad():
            out = self.net(im)

            preds = out.argmax(dim=1).cpu().numpy()
            probs = out.softmax(dim=1).cpu().numpy()

            lb = lb.cpu().numpy()

        lb_flat = np.reshape(lb, -1)
        preds_flat = np.reshape(preds, -1)

        self.append_class_wise_similarity(lb_flat, preds_flat)
        self.append_class_wise(lb_flat, preds_flat)
        # self.append_class_wise_ROC(lb_flat, probs)

        self.steps += 1

        self.net.train()

    def append_class_wise(self, lb_flat, preds_flat):
        gto = (lb_flat == self.ds.ignore_lb)
        gtn = (lb_flat != self.ds.ignore_lb)
        
        for class_id in range(0, self.n_classes):
            gt = (lb_flat == class_id)
            pred = (preds_flat == class_id)

            eq = np.logical_and(gt, pred)
            ne = np.not_equal(gt, pred)

            tp = np.count_nonzero(np.logical_and(eq, gtn))
            tn = np.count_nonzero(np.logical_and(eq, gto))
            
            fp = np.count_nonzero(np.logical_and(ne, gtn))
            fn = np.count_nonzero(np.logical_and(ne, gto))

            try: self.metrics[class_id]['tn'] += tn
            except KeyError: self.metrics[class_id]['tn'] = tn

            try: self.metrics[class_id]['fp'] += fp
            except KeyError: self.metrics[class_id]['fp'] = fp

            try: self.metrics[class_id]['fn'] += fn
            except KeyError: self.metrics[class_id]['fn'] = fn

            try: self.metrics[class_id]['tp'] += tp
            except KeyError: self.metrics[class_id]['tp'] = tp

    def append_class_wise_ROC(self, lb_flat, probs):
        for class_id in range(0, self.n_classes):
            fpr, tpr, thresholds = metrics.roc_curve(lb_flat, np.reshape(probs[:, class_id, :, :], -1), pos_label=class_id)
            auc = metrics.auc(fpr, tpr)
            
            self.auc[class_id] += auc 

            for threshold in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
                idx = utils.find_nearest_index(thresholds, threshold)

                fr = fpr[idx] if not np.isnan(fpr[idx]) else 0
                tr = tpr[idx] if not np.isnan(tpr[idx]) else 0
                
                try:
                    self.ROC[class_id][threshold]['fpr'] += fr
                    self.ROC[class_id][threshold]['tpr'] += tr
                except KeyError: self.ROC[class_id][threshold] = {'fpr': fr, 'tpr': tr}
                
    def append_class_wise_similarity(self, lb_flat, preds_flat):
        for class_id in range(0, self.n_classes):
            gt = (lb_flat == class_id)
            pred = (preds_flat == class_id)

            inter = np.count_nonzero(np.logical_and(pred, gt))
            union = np.count_nonzero(np.logical_or(pred, gt))

            magnitude = np.count_nonzero(pred) + np.count_nonzero(gt)

            ds = 2 * (inter / (magnitude + 1e-6))
            js = (inter / (union + 1e-6))

            self.dice_score[class_id] += ds
            self.jaccard_similarity[class_id] += js

    def compute_metrics(self):
        macro_metrics = {
            'accuracy': 0,
            'f1-score': 0,
            'sensitivity': 0,
            'jaccardSimilarity': 0,
            'diceScore': 0,
            'IoU': 0
        }
        class_metrics = dict()
        for class_id in range(0, self.n_classes):  
            tp = self.metrics[class_id]['tp']
            fp = self.metrics[class_id]['fp']
            fn = self.metrics[class_id]['fn']
            tn = self.metrics[class_id]['tn']

            try: accuracy = (tp + tn) / (tp + tn + fp + fn)
            except ZeroDivisionError: accuracy = 0

            try: f1_score = tp / (tp + 0.5 * (fp + fn))
            except ZeroDivisionError: f1_score = 0

            try: sensitivity = tp / (tp + fn)
            except ZeroDivisionError: sensitivity = 0

            try: iou = tp / (tp + fp) 
            except ZeroDivisionError: iou = 0

            jaccard_similarity = self.jaccard_similarity[class_id] / self.steps
            dice_score = self.dice_score[class_id] / self.steps

            class_info = self.ds.get_class_info(class_id)

            class_metrics[class_info['name']] = {
                'accuracy': accuracy,
                'f1-score': f1_score,
                'sensitivity': sensitivity,
                'jaccardSimilarity': jaccard_similarity,
                'diceScore': dice_score,
                'IoU': iou
            }
            
            macro_metrics['accuracy'] += accuracy  
            macro_metrics['f1-score'] += f1_score  
            macro_metrics['sensitivity'] += sensitivity  
            macro_metrics['jaccardSimilarity'] += jaccard_similarity  
            macro_metrics['diceScore'] += dice_score  
            macro_metrics['IoU'] += iou  

        macro_metrics['accuracy'] /= self.n_classes  
        macro_metrics['f1-score'] /= self.n_classes  
        macro_metrics['sensitivity'] /= self.n_classes  
        macro_metrics['jaccardSimilarity'] /= self.n_classes  
        macro_metrics['diceScore'] /= self.n_classes  
        macro_metrics['IoU'] /= self.n_classes  

        self.macro_metrics = macro_metrics
        self.class_metrics = class_metrics

    def __str__(self):
        self.compute_metrics()

        macro_table = [["mean", *[round(f, 2) for f in self.macro_metrics.values()]]]
        macro_table = tabulate(macro_table, headers=self.macro_metrics.keys(), tablefmt="pretty")
    
        micro_table = [[k, *[round(f, 2) for f in v.values()]] for k, v in self.class_metrics.items()]
        micro_table = tabulate(micro_table, headers=self.macro_metrics.keys(), tablefmt="pretty")

        table = f'{macro_table}\n\n{micro_table}'

        return table


def evaluate_net(args, net):
    scale = 0.5
    cropsize = [int(2048 * scale), int(1024 * scale)]

    ds = CityScapes(args.cityscapes_path, cropsize=cropsize, mode='val')

    dl = DataLoader(ds,
                    batch_size=8,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True)

    evaluate = Evaluate(ds, net)

    print('Evaluate model')
    for im, lb in tqdm(dl):
        with torch.no_grad():
            if torch.cuda.is_available():
                im = im.cuda()

            evaluate(im, lb)

    return str(evaluate)


def main(args):
    scale = 1
    cropsize = [int(2048 * scale), int(1024 * scale)]

    ds = CityScapes(args.cityscapes_path, cropsize=cropsize, mode='val', demo=True)
    n_classes = ds.n_classes
    
    dl = DataLoader(ds,
                    batch_size=4,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True,
                    drop_last=True)

    net = model.get_network(n_classes)

    saved_path = args.saved_model

    loaded_model = torch.load(saved_path, map_location=torch.device('cuda') if torch.cuda.is_available() else 'cpu')
    state_dict = loaded_model['state_dict']
    net.load_state_dict(state_dict, strict=False)

    if torch.cuda.is_available():
        net.cuda()

    evaluate = Evaluate(ds, net)

    for images, im, lb in tqdm(dl):
        if torch.cuda.is_available():
            im = im.cuda()

        evaluate(im, lb)
        
    print(evaluate)


if __name__ == "__main__":
    args = evaluate()
    main(args)
