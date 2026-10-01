# Image classification models

`scripts/download_models.sh` downloads the ONNX files into this directory and verifies their SHA-256 checksums. Model binaries and the downloaded labels file are intentionally ignored by git.

| Model | Source | License | Input | Normalization |
|---|---|---|---|---|
| ResNet-18 v1-7 | [ONNX Model Zoo](https://huggingface.co/onnxmodelzoo/resnet18-v1-7) | Apache-2.0 | RGB NCHW, 224×224 | mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)` |
| MobileNet v2-7 | [ONNX Model Zoo](https://huggingface.co/onnxmodelzoo/mobilenetv2-7) | Apache-2.0 | RGB NCHW, 224×224 | mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)` |

Both produce 1,000 ImageNet class scores. The pipeline decodes JPEG/PNG, scales the shorter side to 256 pixels, center-crops to 224×224, then normalizes RGB values. These are versioned ONNX Model Zoo exports; no model training happens in this project.
