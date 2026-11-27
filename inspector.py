from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms.functional import to_tensor

CLASSES = ["background", "dent", "scratch", "crack"]


def make_model(pretrained=False):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None,
                                  weights_backbone=None, min_size=640, max_size=960)
    heads = model.roi_heads
    heads.box_predictor = FastRCNNPredictor(heads.box_predictor.cls_score.in_features, len(CLASSES))
    heads.mask_predictor = MaskRCNNPredictor(heads.mask_predictor.conv5_mask.in_channels, 256, len(CLASSES))
    return model


class DamageData(torch.utils.data.Dataset):
    def __init__(self, root, split):
        folder = Path(root)
        self.folder = folder / f"{split}2017"
        self.coco = COCO(str(folder / "annotations" / f"instances_{split}2017.json"))
        self.labels = {c["id"]: CLASSES.index(c["name"].strip().lower())
                       for c in self.coco.cats.values() if c["name"].strip().lower() in CLASSES[1:]}
        if set(self.labels.values()) != {1, 2, 3}:
            raise ValueError(f"Expected dent, scratch and crack in {self.coco.cats}")
        self.ids = sorted(self.coco.imgs)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        info = self.coco.imgs[image_id]
        with Image.open(self.folder / info["file_name"]) as source:
            image = to_tensor(source.convert("RGB"))
        h, w = image.shape[1:]
        masks, boxes, labels = [], [], []
        for ann in self.coco.imgToAnns.get(image_id, []):
            if ann["category_id"] not in self.labels or ann.get("iscrowd"):
                continue
            mask = self.coco.annToMask(ann)
            if mask.shape != (h, w):
                raise ValueError(f"Image and annotation dimensions differ for {image_id}")
            ys, xs = np.nonzero(mask)
            if len(xs):
                boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
                masks.append(mask)
                labels.append(self.labels[ann["category_id"]])
        return image, dict(boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                           masks=torch.as_tensor(np.asarray(masks, dtype=np.uint8).reshape(-1, h, w)),
                           labels=torch.tensor(labels, dtype=torch.int64))


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != CLASSES:
        raise ValueError("Checkpoint classes differ from the inspector classes.")
    model = make_model()
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()
