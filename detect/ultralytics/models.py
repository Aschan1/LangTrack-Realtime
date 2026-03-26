# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.engine.model import Model
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.nn.tasks import DetectionModel, SegmentationModel, WorldModel, YOLOEModel, YOLOESegModel
from ultralytics.utils import DEFAULT_CFG, ROOT, YAML, nms, ops


class DetectionPredictor(BasePredictor):
    """Minimal detection predictor for the vendored YOLO/YOLOE models."""

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])


class SegmentationPredictor(DetectionPredictor):
    """Segmentation predictor for vendored YOLOE segmentation models."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def construct_result(self, pred, img, orig_img, img_path, proto=None):
        if pred.shape[0] == 0:
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])
        else:
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

        if masks is not None:
            keep = masks.amax((-2, -1)) > 0
            if not all(keep):
                pred, masks = pred[keep], masks[keep]

        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)

    def postprocess(self, preds, img, orig_imgs):  # type: ignore[override]
        protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        preds = nms.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        return [
            self.construct_result(pred, img, orig_img, img_path, proto)
            for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
        ]


class YOLO(Model):
    """Basic YOLO wrapper with support for local YOLOE and YOLO-World weights."""

    def __init__(self, model: str | Path = "yolo26n.pt", task: str | None = None, verbose: bool = False):
        path = Path(model if isinstance(model, (str, Path)) else "")
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        elif "yoloe" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:
            new_instance = YOLOE(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        return {
            "detect": {"model": DetectionModel, "predictor": DetectionPredictor},
            "segment": {"model": SegmentationModel, "predictor": SegmentationPredictor},
        }


class YOLOWorld(Model):
    """YOLO-World wrapper for the vendored package."""

    def __init__(self, model: str | Path = "yolov8s-world.pt", verbose: bool = False) -> None:
        super().__init__(model=model, task="detect", verbose=verbose)
        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        return {"detect": {"model": WorldModel, "predictor": DetectionPredictor}}

    def set_classes(self, classes: list[str]) -> None:
        self.model.set_classes(classes)
        if " " in classes:
            classes = [c for c in classes if c != " "]
        self.model.names = classes
        if self.predictor:
            self.predictor.model.names = classes


class YOLOE(Model):
    """YOLOE wrapper that restores the missing local ultralytics.models API."""

    def __init__(self, model: str | Path = "yoloe-11s-seg.pt", task: str | None = None, verbose: bool = False) -> None:
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        return {
            "detect": {"model": YOLOEModel, "predictor": DetectionPredictor},
            "segment": {"model": YOLOESegModel, "predictor": SegmentationPredictor},
        }

    def get_text_pe(self, texts):
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_text_pe(texts)

    def get_visual_pe(self, img, visual):
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_visual_pe(img, visual)

    def set_vocab(self, vocab: list[str], names: list[str]) -> None:
        assert isinstance(self.model, YOLOEModel)
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_vocab(names)

    def set_classes(self, classes: list[str], embeddings: torch.Tensor | None = None) -> None:
        assert isinstance(self.model, YOLOEModel)
        if embeddings is None:
            embeddings = self.get_text_pe(classes)
        self.model.set_classes(classes, embeddings)
        assert " " not in classes
        self.model.names = classes
        if self.predictor:
            self.predictor.model.names = classes


class NAS:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Vendored ultralytics.models.NAS is not available in this checkout.")


class SAM:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Vendored ultralytics.models.SAM is not available in this checkout.")


class FastSAM:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Vendored ultralytics.models.FastSAM is not available in this checkout.")


class RTDETR:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Vendored ultralytics.models.RTDETR is not available in this checkout.")


__all__ = ("YOLO", "YOLOWorld", "YOLOE", "NAS", "SAM", "FastSAM", "RTDETR")
