import logging
import os.path as osp
import tempfile
from tabulate import tabulate
from PIL import Image

import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmdet.core import eval_recalls
from mmdet.utils import print_log
from .custom import CustomDataset 
from .registry import DATASETS

def dice_coefficient(pred_mask, gt_mask):
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    if union == 0:
        return 0.0  # Handle case when both masks are empty
    else:
        return 2.0 * intersection / union

def mean_dice_coefficient(predictions, ground_truths, num_classes):
    dice_scores = []
    for class_id in range(num_classes):
        class_dice_scores = []
        for pred_mask, gt_mask in zip(predictions, ground_truths):
            class_pred_mask = (pred_mask == class_id).astype(np.uint8)
            class_gt_mask = (gt_mask == class_id).astype(np.uint8)
            class_dice_score = dice_coefficient(class_pred_mask, class_gt_mask)
            class_dice_scores.append(class_dice_score)
        mean_class_dice_score = np.mean(class_dice_scores)
        dice_scores.append(mean_class_dice_score)
    mean_dice = np.mean(dice_scores)
    return (dice_scores,mean_dice)

def compute_iou(gt_mask, pred_mask, num_classes):
    ious = []
    for class_id in range(num_classes):
        # Get pixels for the current class in both masks
        gt_class_pixels = (gt_mask == class_id)
        pred_class_pixels = (pred_mask == class_id)

        # Calculate intersection and union
        intersection = np.logical_and(gt_class_pixels, pred_class_pixels).sum()
        union = np.logical_or(gt_class_pixels, pred_class_pixels).sum()

        # Avoid division by zero
        if union == 0:
            iou = 0
        else:
            iou = intersection / union
        ious.append(iou)

    # Compute mean IoU
    mean_iou = np.mean(ious)
    return (ious,mean_iou)

@DATASETS.register_module
class ForestDataset(CustomDataset):
    global CLASSES
    CLASSES={'Road','Grass','Vegetation','Tree','Sky','Obstacle'}

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0] + 1,
            _bbox[3] - _bbox[1] + 1,
        ]

    def _proposal2json(self, results):
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results
    
    def display(self,batch_ious, batch_dices):
        headers = ['Class', 'IoU', 'Dice']
        data = []
        for class_idx in range(len(batch_ious)):
            data.append([class_idx, batch_ious[class_idx], batch_dices[class_idx]])
        formatted_data = []
        for class_data in data:
            class_idx = class_data[0]
            iou = class_data[1]
            dice = class_data[2]
            formatted_data.append([class_idx, f'{iou:.4f}', f'{dice:.4f}'])
        print()
        print(tabulate(formatted_data, headers=headers, floatfmt=".4f"))

    def _det2json(self, results):
        json_results = []
        counts=np.zeros((len(CLASSES)))
        batch_ious=np.zeros((len(CLASSES)))
        batch_dices=np.zeros((len(CLASSES)))
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            label=0
            mask = result[label][0]
            mask=np.array(mask)
            filename=result[label][1][0]['filename']
            new_filename = filename.replace("val", "freiburg_panoptic_val")
            img=Image.open(new_filename)
            img = img.resize((mask.shape[1], mask.shape[0]), Image.ANTIALIAS)
            img=np.array(img)
    
            color_mapping = {
                (170, 170, 170): 0,   # Road
                (0, 255, 0): 1,       # Grass
                (102, 102, 51): 2,    # Vegetation
                (0, 60, 0): 3,        # Tree
                (0, 120, 255): 4,     # Sky
                (0, 0, 0): 5          # Obstacle
            }

            # gt = np.zeros_like(img[:, :, 0], dtype=np.uint8)
            # for color, label in color_mapping.items():
            #     gt[(img == color).all(axis=-1)] = label

            pixels = img.reshape(-1, 3)

            # Expand the color mapping to a 3D array to enable broadcasting
            color_mapping_array = np.array(list(color_mapping.keys()))

            # Calculate the Euclidean distances for all pixels and color mapping
            distances = np.sqrt(np.sum((pixels[:, None] - color_mapping_array) ** 2, axis=-1))

            # Find the index of the closest color for each pixel
            closest_color_indices = np.argmin(distances, axis=1)

            # Map the closest color indices to the corresponding labels
            labels = np.array(list(color_mapping.values()))
            gt_flat = labels[closest_color_indices]

            # Reshape the labels to match the original image shape
            gt = gt_flat.reshape(img.shape[:2])
            # print(gt)

            ious,mean_iou = compute_iou(gt, mask, num_classes=len(CLASSES))
            ious=np.array(ious)
            cnt=np.count_nonzero(ious,axis=0)
            mean_iou=np.sum(ious)/cnt
            dices,mean_dice=mean_dice_coefficient(mask,gt,num_classes=len(CLASSES))
            dices=np.array(dices)
            cnt=np.count_nonzero(dices,axis=0)
            mean_dice=np.sum(dices)/cnt
            present=np.array(ious>0)
            counts+=present
            batch_ious+=ious
            batch_dices+=dices
        for i in range(len(counts)):
            if(counts[i]):
                batch_ious[i]/=counts[i]
                batch_dices[i]/=counts[i]
        self.display(batch_ious,batch_dices)
        return json_results

    def _segm2json(self, results):
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and
                values are corresponding filenames.
        """
        result_files = dict()
        json_results = self._det2json(results)
        result_files['segm'] = '{}.{}.json'.format(outfile_prefix, 'segm')
        # result_files['proposal'] = '{}.{}.json'.format(
        #     outfile_prefix, 'bbox')
        mmcv.dump(json_results, result_files['segm'])
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.getAnnIds(imgIds=self.img_ids[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        """Evaluation in COCO protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str: float]
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError('metric {} is not supported'.format(metric))

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = {}
        cocoGt = self.coco
        for metric in metrics:
            msg = 'Evaluating {}...'.format(metric)
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
                    log_msg.append('\nAR@{}\t{:.4f}'.format(num, ar[i]))
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError('{} is not in results'.format(metric))
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = self.img_ids
            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.params.maxDets = list(proposal_nums)
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
                    'AR_l@1000'
                ]
                for i, item in enumerate(metric_items):
                    val = float('{:.3f}'.format(cocoEval.stats[i + 6]))
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    pass  # TODO
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]
                for i in range(len(metric_items)):
                    key = '{}_{}'.format(metric, metric_items[i])
                    val = float('{:.3f}'.format(cocoEval.stats[i]))
                    eval_results[key] = val
                eval_results['{}_mAP_copypaste'.format(metric)] = (
                    '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
