# MaxSight architecture

This document describes the model and system architecture end to end.

## Overview

MaxSight is a multi-task vision (and optional audio) model for accessibility. It uses a shared backbone and feature pyramid, then many task-specific heads. Input is typically RGB images `[B, 3, 224, 224]` and optionally audio features; outputs include detections, urgency, distance, depth, motion, therapy state, scene description, and more.

## Tiers and capabilities

The codebase uses capability tiers (T0–T5) that control which components are enabled:

- **T0 (baseline):** ResNet50 + FPN + core detection heads only.
- **T1:** Adds attention (CBAM, SE).
- **T2:** Adds hybrid CNN–ViT backbone.
- **T3:** Adds cross-task attention.
- **T4:** Adds cross-modal (audio–visual) fusion.
- **T5:** Adds temporal encoder (ConvLSTM / TimeSformer) and full retrieval.

Stage A (ResNet50 + FPN) is always used for safety-critical, low-latency features; Stage B is tier-dependent and can add ViT, temporal, and retrieval.

## Backbone and FPN

- **Backbone:** ResNet50 (or hybrid CNN–ViT in T2+) in `ml/models/backbone/`. Produces multi-scale feature maps.
- **FPN:** Feature Pyramid Network in the main model builds C2–C5 and P2–P5 for multi-scale detection.
- **Dynamic convolution:** Optional adaptive convolutions in `ml/models/backbone/dynamic_conv.py` for condition-adaptive processing.

## Heads

Heads live under `ml/models/heads/` and consume shared features:

- **Detection:** Objectness, classification, box regression (anchor-free, FCOS-style). COCO 91 classes plus accessibility classes.
- **Urgency and distance:** Urgency levels (e.g. 0–3), distance zones (near / medium / far).
- **Therapy-related:** Contrast head, depth head, motion head, fatigue head, therapy state head.
- **Scene and text:** Scene description head, OCR head, ROI priority head.
- **Multimodal:** Sound event head when audio is used; personalization and predictive alert heads.

Heads are gated by tier and condition mode so only the relevant subset runs per run.

## Fusion and temporal

- **Multimodal fusion:** `ml/models/fusion/multimodal_fusion.py` combines visual and audio features for T4+.
- **Temporal encoder:** `ml/models/temporal/temporal_encoder.py` (ConvLSTM and/or TimeSformer) for T5 temporal modeling.
- **Scene graph:** `ml/models/scene_graph/scene_graph_encoder.py` for relational reasoning; often stubbed or disabled for export.

## Retrieval

- **Encoders:** Patch, region, global, OCR, depth, audio in `ml/retrieval/encoders/`.
- **Indexing:** `ml/retrieval/indexing/` builds and manages neural indexes.
- **Retrieval:** Two-stage (e.g. ANN then rerank) in `ml/retrieval/retrieval/`.

## Condition-specific behavior

The model supports multiple vision conditions (e.g. glaucoma, AMD, cataracts, CVI). Condition affects:

- Preprocessing (e.g. in `ml/utils/preprocessing.py`): contrast, blur, central/peripheral emphasis.
- Which heads are emphasized and how outputs are scheduled (e.g. `ml/utils/output_scheduler.py`).

## Data flow (inference)

1. Input: images (and optionally audio features).
2. Preprocessing: condition-specific normalization and augmentation (if any).
3. Backbone + FPN: multi-scale features.
4. Heads: run in parallel where enabled; produce raw logits and scores.
5. Post-processing: NMS, thresholding, scheduling (urgency, distance, etc.).
6. Output: dictionary of tensors and optional scene/OCR/therapy outputs.

## Export and deployment

- **Export:** `ml/training/export.py` supports JIT, ExecuTorch (.pte), CoreML, ONNX. Dict outputs are flattened or stubbed for trace-friendly formats.
- **iOS bundle:** `export_ios_bundle()` produces a folder with model, configs, and a processing reference for Xcode integration.

For more on training layout and data flow, see `docs/training_architecture.md` and `docs/training-data-loading.md`.
