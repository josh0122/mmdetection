import torch
from mmdet.apis import init_detector, inference_detector
import mmcv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

from mmcv import Config

from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import mmcv
import os.path as osp
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# config 파일을 설정하고, 다운로드 받은 pretrained 모델을 checkpoint로 설정. 
config_file = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'


@DATASETS.register_module(force=True)
class NucleusDataset(CocoDataset):
  CLASSES = ['body', 'right_hand', 'left_hand', 'left_foot', 'right_foot', 'right_thigh', 'left_thigh', 'right_calf', 'left_calf', 
     'left_arm', 'right_arm', 'left_forearm',  'right_forearm' , 'head']

cfg = Config.fromfile(config_file)

# dataset에 대한 환경 파라미터 수정. 
cfg.dataset_type = 'NucleusDataset' # 'NucleusDataset'
cfg.data_root = '/home/seonghwan/PyTorch-Simple-MaskRCNN/heroes_data/train_dataset' #'/content/coco_output/'

# train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정. 
cfg.data.train.type =  'NucleusDataset' #'NucleusDataset'
cfg.data.train.data_root = '/home/seonghwan/PyTorch-Simple-MaskRCNN/heroes_data/train_dataset/' #'/content/coco_output/'
cfg.data.train.ann_file = 'new_ai_2014_train=mask.json' #'train_coco.json'
cfg.data.train.img_prefix = '2014_train=mask'

cfg.data.val.type = 'NucleusDataset' # 'NucleusDataset'
cfg.data.val.data_root = '/home/seonghwan/PyTorch-Simple-MaskRCNN/heroes_data/train_dataset/'#'/content/coco_output/'
cfg.data.val.ann_file ='new_ai_2014_val=mask.json' # 'val_coco.json'
cfg.data.val.img_prefix = '2014_val=mask'

# class의 갯수 수정. 
cfg.model.roi_head.bbox_head.num_classes = 14 # 1
cfg.model.roi_head.mask_head.num_classes = 14 # 1

# pretrained 모델
cfg.load_from = 'checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth' #'/content/mmdetection/checkpoints/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth'

# 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정. 
cfg.work_dir = './tutorial_exps'

# 학습율 변경 환경 파라미터 설정. 
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# CocoDataset의 경우 metric을 bbox로 설정해야 함.(mAP아님. bbox로 설정하면 mAP를 iou threshold를 0.5 ~ 0.95까지 변경하면서 측정)
cfg.evaluation.metric = ['bbox', 'segm']
cfg.evaluation.interval = 12
cfg.checkpoint_config.interval = 12

# epochs 횟수는 36으로 증가 
cfg.runner.max_epochs = 80

# 두번 config를 로드하면 lr_config의 policy가 사라지는 오류로 인하여 설정. 
cfg.lr_config.policy='step'
# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# train, valid 용 Dataset 생성. 
datasets_train = [build_dataset(cfg.data.train)]
datasets_val = [build_dataset(cfg.data.val)]

model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets_train[0].CLASSES



mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# epochs는 config의 runner 파라미터로 지정됨. 기본 12회 
train_detector(model, datasets_train, cfg, distributed=False, validate=True)