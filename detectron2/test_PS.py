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
                        "../datasets/train_PS50.json", 
                        "../datasets/train_PS")
register_coco_instances("vehicle_test_PS", {}, 
                        "../datasets/test_PS10.json", 
                        "../datasets/test_PS")

vehicle_train_metadata = MetadataCatalog.get("vehicle_train_PS")
dataset_dicts = DatasetCatalog.get("vehicle_train_PS")

vehicle_test_metadata = MetadataCatalog.get("vehicle_test_PS")
dataset_dicts = DatasetCatalog.get("vehicle_test_PS")

cfg = get_cfg()
cfg.OUTPUT_DIR = "./output_PS"
cfg.merge_from_file("../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("vehicle_train_PS",)
cfg.DATASETS.TEST = ("vehicle_test_PS",)
cfg.MODEL.WEIGHTS = "../weights/model_final_PS.pth"

cfg.DATALOADER.NUM_WORKERS = 40
cfg.SOLVER.IMS_PER_BATCH = 10

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs("./output_PS", exist_ok=True)

trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=True)

evaluator = COCOEvaluator("vehicle_test_PS", ("bbox", "segm"), False, output_dir="./output_PS/")
test_loader = build_detection_test_loader(cfg, "vehicle_test_PS")
print(inference_on_dataset(trainer.model, test_loader, evaluator))
