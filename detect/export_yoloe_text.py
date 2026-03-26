"""
Export YOLOE with explicit text embedding input for Triton/TensorRT deployment.

Usage:
    cd /home/chen/workplace/LangTrack-Realtime
    python ultralytics/export_yoloe_text.py

This creates:
    yoloe-26l-seg-text.onnx  (validate with onnxruntime)

Then convert to TensorRT engine:
    trtexec \
      --onnx=yoloe-26l-seg-text.onnx \
      --saveEngine=model_repository/yoloe/1/yoloe-26l-seg-text.plan \
      --minShapes=image:1x3x640x640,text_emb:1x1x512 \
      --optShapes=image:1x3x640x640,text_emb:1x4x512 \
      --maxShapes=image:1x3x640x640,text_emb:1x80x512 \
      --fp16
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ultralytics"))

import torch
import torch.nn as nn
from ultralytics import YOLOE
from ultralytics.nn.modules.head import YOLOEDetect


class YOLOETextWrapper(nn.Module):
    """Wraps YOLOEModel to expose raw text embeddings as an explicit ONNX input.

    The standard YOLOE export ignores the tpe (text positional embedding) argument,
    baking class embeddings into the model weights.  This wrapper keeps the reprta
    normalisation layer inside the graph so that callers can pass raw MobileClip
    embeddings at inference time without recompiling the engine per query.

    Args:
        yoloe_model: Loaded ultralytics YOLOE instance (not the inner nn.Module).
    """

    def __init__(self, yoloe_model: YOLOE):
        super().__init__()
        self.inner = yoloe_model.model  # the actual YOLOEModel (nn.Module)

    def forward(self, image: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Run YOLOE detection with dynamic text embeddings.

        Args:
            image:    Float32 [B, 3, 640, 640]
            text_emb: Float32 [1, num_classes, 512]  — raw MobileClip output,
                      L2-normalised but NOT yet passed through reprta.
                      The reprta + L2-normalisation step runs inside this graph.

        Returns:
            Float32 detection tensor (export mode: [B, 4+nc, num_anchors])
        """
        return self.inner.predict(image, tpe=text_emb)


def export_onnx(
    weights: str = "yoloe-26l-seg.pt",
    output: str = "yoloe-26l-seg-text.onnx",
    imgsz: int = 640,
    nc_opt: int = 4,       # optimum number of classes (for shape hints)
    nc_max: int = 80,      # maximum number of classes
    opset: int = 17,
):
    print(f"[1/4] Loading {weights} ...")
    model = YOLOE(weights)
    model.eval()

    # Put the detection head into export mode so it returns a plain tensor.
    head = model.model.model[-1]
    assert isinstance(head, YOLOEDetect), "Last layer must be YOLOEDetect"
    head.export = True
    head.format = "onnx"

    wrapper = YOLOETextWrapper(model)
    wrapper.eval()

    device = next(wrapper.parameters()).device
    dummy_image = torch.zeros(1, 3, imgsz, imgsz, device=device)
    dummy_text  = torch.zeros(1, nc_opt, 512, device=device)

    print(f"[2/4] Tracing with dummy shapes: image={list(dummy_image.shape)}, "
          f"text_emb={list(dummy_text.shape)}")

    print(f"[3/4] Exporting to {output} ...")
    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_text),
        output,
        input_names=["image", "text_emb"],
        output_names=["output0"],
        dynamic_axes={
            "image":    {0: "batch"},
            "text_emb": {1: "num_classes"},
            "output0":  {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"[3/4] ONNX saved → {output}")

    print("[4/4] Validating with onnxruntime ...")
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
        dummy_img_np  = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        dummy_text_np = np.zeros((1, nc_opt, 512),     dtype=np.float32)
        out = sess.run(None, {"image": dummy_img_np, "text_emb": dummy_text_np})
        print(f"[4/4] ORT validation passed — output shape: {out[0].shape}")
    except ImportError:
        print("[4/4] onnxruntime not installed, skipping validation.")

    print("\nNext step — build TensorRT engine:")
    print(f"  trtexec \\")
    print(f"    --onnx={output} \\")
    print(f"    --saveEngine=model_repository/yoloe/1/yoloe-26l-seg-text.plan \\")
    print(f"    --minShapes=image:1x3x{imgsz}x{imgsz},text_emb:1x1x512 \\")
    print(f"    --optShapes=image:1x3x{imgsz}x{imgsz},text_emb:1x{nc_opt}x512 \\")
    print(f"    --maxShapes=image:1x3x{imgsz}x{imgsz},text_emb:1x{nc_max}x512 \\")
    print(f"    --fp16")


if __name__ == "__main__":
    export_onnx()
