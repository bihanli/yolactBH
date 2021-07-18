"""
Runs the coco-supplied cocoeval script to evaluate detections
outputted by using the output_coco_json flag in eval.py.
"""
import copy
import argparse
from data import get_label_map
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import numpy as np
from data import cfg
parser = argparse.ArgumentParser(description='COCO Detections Evaluator')
parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str)
parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str)
parser.add_argument('--gt_ann_file',   default='data/coco/annotations/datacocoaug_valid.json', type=str)#noairninecocotest_validLabelWY
#parser.add_argument('--gt_ann_file',   default='data/coco/annotations/testes.json', type=str)
parser.add_argument('--eval_type',     default='both', choices=['bbox', 'mask', 'both'], type=str)
args = parser.parse_args()

def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """
    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else '{:0.2f}'.format(iouThr)
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        #print(aind,mind)
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            #print(111,s.shape)
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
                #print(t)
            if isinstance(catId, int):
                #print(222,s.shape)
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        #print(333,s.shape)
        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string
    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
    print_info = "\n".join(print_list)
    if not self.eval:
        raise Exception('Please run accumulate() first')
    return stats, print_info
if __name__ == '__main__':

	eval_bbox = (args.eval_type in ('bbox', 'both'))
	eval_mask = (args.eval_type in ('mask', 'both'))

	print('Loading annotations...')
	gt_annotations = COCO(args.gt_ann_file)
	if eval_bbox:
		bbox_dets = gt_annotations.loadRes(args.bbox_det_file)
	if eval_mask:
		mask_dets = gt_annotations.loadRes(args.mask_det_file)

	if eval_bbox:
		print('\nEvaluating BBoxes:')
		bbox_eval = COCOeval(gt_annotations, bbox_dets, 'bbox')
		bbox_eval.evaluate()
		bbox_eval.accumulate()
		bbox_eval.summarize()
		#print(bbox_eval.stats)
		voc=[]
		#category_index=['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'fire hydrant','stop sign','parking meter', 'bird', 'cat','dog']
		#category_index=['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'cat','dog']
		category_index=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus','train', 'truck', 'traffic light', 'fire hydrant','stop sign', 'bird', 'cat']
		#category_index=['person', 'car', 'motorcycle',  'truck', 'traffic light']
		"""
		category_index = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']
                """
		for i in range(len(category_index)):
				stats,_=summarize(bbox_eval,catId=i)
				voc.append(" {:15}: {}".format(category_index[i], stats[1]))
		print_voc = "\n".join(voc)
		print(print_voc)
	if eval_mask:
		print('\nEvaluating Masks:')
		bbox_eval = COCOeval(gt_annotations, mask_dets, 'segm')
		bbox_eval.evaluate()
		bbox_eval.accumulate()
		bbox_eval.summarize()
		voc=[]
		
		for i in range(len(category_index)):
				stats,_=summarize(bbox_eval,catId=i)
				voc.append(" {:15}: {}".format(category_index[i], stats[1]))
		print_voc = "\n".join(voc)
		print(print_voc)

