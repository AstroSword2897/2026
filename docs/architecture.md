# EnviroSight CNN Architecture Design

## Overview
Multi-level CNN architecture for environmental scene understanding, designed to support 10 vision conditions with condition-specific adaptations.

## Architecture Levels

### Level 1: Low-Level Features (Backbone)
- **ResNet50** (pretrained on ImageNet)
- Extracts edges, textures, and basic visual features
- Output: 2048-d feature vector per spatial location
- Spatial resolution: 7x7 (from 224x224 input)

### Level 2: Object Detection
- **Feature Pyramid Network (FPN)** on ResNet50 features
- Multi-scale object detection
- Handles objects of various sizes
- Output: Feature maps at multiple scales

### Level 3: Scene Semantics
- **Multi-Head Attention** mechanism
- Context understanding (indoor/outdoor, room type)
- Danger level assessment
- Output: Scene context embedding (512-d)

### Level 4: Structured Outputs
Five parallel output heads:

1. **Object Classification Head**
   - Input: FPN features
   - Architecture: 2x Linear(2048 → 1024 → 48)
   - Output: 48 environmental classes
   - Activation: Softmax

2. **Spatial Localization Head**
   - Input: FPN features
   - Architecture: 2x Linear(2048 → 1024 → 4)
   - Output: Bounding boxes (x, y, w, h)
   - Activation: Sigmoid (normalized coordinates)

3. **Scene Description Head**
   - Input: Attention context + FPN features
   - Architecture: Linear(2048+512 → 512)
   - Output: 512-d embedding for TTS
   - Activation: Tanh

4. **Urgency Scoring Head**
   - Input: FPN features + scene context
   - Architecture: Linear(2048+512 → 4)
   - Output: 4 urgency levels (safe, caution, warning, danger)
   - Activation: Softmax

5. **Distance Estimation Head**
   - Input: FPN features + bounding box size
   - Architecture: Linear(2048+4 → 3)
   - Output: 3 distance zones (near, medium, far)
   - Activation: Softmax

## Audio Fusion Branch (Optional)
- **Input**: MFCC features (128-d)
- **Architecture**: Linear(128 → 256) → BatchNorm → ReLU → Dropout(0.5) → Linear(256 → 512)
- **Fusion**: Concatenate with visual features before attention
- **Output**: Audio-enhanced scene context

## Condition-Specific Adaptations

### Glaucoma Mode
- Separate attention for center (0-40% width) vs peripheral (40-100% width)
- Peripheral objects get +0.2 urgency boost

### AMD Mode
- Dual-path: high-res center crop (40% region) + full-frame edge detection
- Edge objects prioritized in output

### Color Blindness Mode
- Additional color classification head: Linear(2048 → 12)
- 12 color categories output per object

### CVI Mode
- Single-object focus: only top-1 detection
- Simplified class names

### Low-Light Mode (Retinitis Pigmentosa)
- Preprocessing: Gamma correction + histogram stretching
- Enhanced feature extraction

## Parameter Count
- ResNet50 backbone: ~23M parameters
- FPN: ~2M parameters
- Attention: ~500K parameters
- Output heads: ~1M parameters
- **Total: ~26.5M parameters**

## Input/Output Specifications

### Input
- **Image**: 224x224x3 (RGB, normalized with ImageNet stats)
- **Audio** (optional): 128-d MFCC features

### Output
- **Classifications**: [batch, 48] (48 object classes)
- **Bounding boxes**: [batch, N, 4] (N detections, x,y,w,h)
- **Scene embedding**: [batch, 512]
- **Urgency scores**: [batch, N, 4]
- **Distance zones**: [batch, N, 3]

## Training Strategy
- **Backbone**: Frozen for first 5 epochs, then fine-tuned
- **Learning rate**: 1e-4 for backbone, 1e-3 for heads
- **Optimizer**: AdamW with weight decay 1e-4
- **Scheduler**: Cosine annealing
- **Batch size**: 16 (adjust based on GPU memory)

## Inference Performance
- **Target latency**: <500ms on Apple Silicon
- **Model size**: <50MB after quantization
- **Memory**: <250MB RAM
