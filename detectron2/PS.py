import torch, torchvision
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CocoTrainer(DefaultTrainer):
    
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)
register_coco_instances("vehicle_train_PS", {}, 
                        "/home/super/Desktop/yh/train_PS50.json", 
                        "/home/super/Desktop/yh/train_PS")
register_coco_instances("vehicle_test_PS", {}, 
                        "/home/super/Desktop/yh/test_PS10.json", 
                        "/home/super/Desktop/yh/test_PS")

vehicle_train_metadata = MetadataCatalog.get("vehicle_train_PS")
dataset_dicts = DatasetCatalog.get("vehicle_train_PS")

vehicle_test_metadata = MetadataCatalog.get("vehicle_test_PS")
dataset_dicts = DatasetCatalog.get("vehicle_test_PS")

# DatasetCatalog.remove('vehicle_train_BB')
# MetadataCatalog.remove("vehicle_train_BB")


# for d in random.sample(dataset_dicts, 3):
#   img = cv2.imread(d["file_name"])
#   visualizer = Visualizer(img[:, :, ::-1], metadata=vehicle_train_metadata, scale=0.5)
#   vis = visualizer.draw_dataset_dict(d)
#   cv2.imshow("random_image",vis.get_image()[:, :, ::-1])
#   cv2.waitKey(0)

cfg = get_cfg()
cfg.merge_from_file("/home/super/Desktop/yh/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("vehicle_train_PS",)
cfg.DATASETS.TEST = ("vehicle_test_PS",)
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("/home/super/Desktop/yh/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "/home/super/Desktop/yh/detectron2/detectron2/output_BB/model_final.pth"


cfg.DATALOADER.NUM_WORKERS = 40
cfg.SOLVER.IMS_PER_BATCH = 10
# cfg.SOLVER.IMS_PER_BATCH = 8  #gpu memory 25000
# cfg.SOLVER.IMS_PER_BATCH = 2  #gpu memory 10000 12sec 15sec
cfg.SOLVER.BASE_LR = 0.0003
cfg.SOLVER.WARMUP_ITERS = 40000
cfg.SOLVER.MAX_ITER =  300000  # 300 iterations seems good enough, but you can certainly train longer
cfg.SOLVER.STEPS = (40000, 80000, 120000,160000,200000)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.TEST.EVAL_PERIOD = 20000
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()