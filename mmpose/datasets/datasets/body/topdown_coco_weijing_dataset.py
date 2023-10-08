import os

import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval


from ...builder import DATASETS


from .coco_dataset import TopDownCocoDataset
from .pixeldiseval import PixelDisEval

@DATASETS.register_module()
class TopDownCocoWeijingDataset(TopDownCocoDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info = None,
                 test_mode=False):
        self.datasetFolder = os.path.dirname(ann_file)
        super(TopDownCocoDataset, self).__init__(
            ann_file, img_prefix, data_cfg, pipeline, dataset_info, test_mode=test_mode)

        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        # self.image_thr = data_cfg['image_thr']
        if 'isdraw_gt_dt' in data_cfg:
            self.isdraw_gt_dt = data_cfg['isdraw_gt_dt']
        else:
            self.isdraw_gt_dt = False

        self.soft_nms = data_cfg['soft_nms']
        self.use_nms = data_cfg.get('use_nms', True)
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']
        # self.bbox_thr = data_cfg['bbox_thr']

        # self.ann_info['flip_pairs'] = flip_pairs

        self.ann_info['upper_body_ids'] = ()
        self.ann_info['lower_body_ids'] = ()

        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['joint_weights'] = \
            np.ones((self.ann_info['num_joints'], 1), dtype=np.float32)

        num_joints = data_cfg['num_joints']
        self.sigmas = np.linspace(0.5, 0.5, num_joints)

        self.coco = COCO(ann_file)

        cats = [
            cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())
        ]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            (self._class_to_coco_ind[cls], self._class_to_ind[cls])
            for cls in self.classes[1:])
        self.img_ids = self.coco.getImgIds()
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)
        self.dataset_name = 'coco'

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

        pass
    def draw_gt_dt(self, cocoGt:COCO, cocoDt:COCO):
        saveFolder = "evalImages/"

        pass
    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)
        coco_eval = PixelDisEval(self.coco, coco_det, 'keypoints', self.dataset_info, self.img_prefix, self.datasetFolder, self.isdraw_gt_dt)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP95', 'AP', 'precision', 'recall'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str