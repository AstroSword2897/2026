"""
OCR Integration Module for MaxSight
Sprint 1 Day 5: OCR Integration & Text Reading

PROJECT PHILOSOPHY & APPROACH:
This module implements "Reads Environment" capability - detecting and reading text from the
environment. This directly addresses the problem statement's requirement for text reading
support, enabling users to read signs, labels, and documents.

WHY OCR INTEGRATION MATTERS:
Text is everywhere in the environment - signs, labels, documents, menus. Users with vision
impairments need this text read aloud. This module provides that capability by:
1. Detecting text regions in images
2. Extracting text using OCR
3. Providing text-to-speech for reading aloud

This supports "Clear Multimodal Communication" by making textual information accessible
through audio, enabling users to access information that would otherwise be inaccessible.

HOW IT CONNECTS TO THE PROBLEM STATEMENT:
The problem asks: "What are ways that those who cannot see... be able to interact with the
world like those can?" OCR integration answers by providing access to textual information
through audio, enabling users to read signs, labels, and documents independently.

RELATIONSHIP TO BARRIER REMOVAL METHODS:
1. ENVIRONMENTAL STRUCTURING: Makes textual information accessible and understandable
2. CLEAR MULTIMODAL COMMUNICATION: Converts visual text to audio
3. SKILL DEVELOPMENT: Supports reading skills through practice

TECHNICAL DESIGN DECISIONS:
- Cross-platform: iOS Vision framework for production, pytesseract for development
- Adaptive preprocessing: Improves OCR accuracy in low-contrast images
- Confidence scoring: Combines region detection and OCR confidence for reliability
- DBSCAN clustering: Efficient pixel clustering for large images
- Line/block grouping: Prevents splitting connected text

This module provides:
- Text region detection (using model's text_head)
- OCR text extraction (iOS Vision framework integration)
- Text-to-speech pipeline for reading aloud

iOS VISION FRAMEWORK INTEGRATION PLAN:

Phase 1: Model-Based Text Region Detection (Current - Python)
- Uses MaxSightCNN text_head to detect text regions
- Provides bounding boxes and confidence scores
- Efficient for real-time processing

Phase 2: iOS Vision Framework Integration (Production)
- Replace pytesseract with VNRecognizeTextRequest
- Use Vision framework for OCR on detected regions
- Leverage iOS Neural Engine for hardware acceleration

Implementation Steps for iOS:

1. Swift/iOS Integration:
   ```swift
   import Vision
   
   func recognizeText(in image: CIImage, regions: [CGRect]) -> [VNRecognizedTextObservation] {
       let request = VNRecognizeTextRequest { request, error in
           // Handle results
       }
       request.recognitionLevel = .accurate
       request.usesLanguageCorrection = true
       
       let handler = VNImageRequestHandler(ciImage: image, options: [:])
       try? handler.perform([request])
       
       return request.results as? [VNRecognizedTextObservation] ?? []
   }
   ```

2. Bridge Python Model Outputs to iOS:
   - Export text_head outputs (text_scores, boxes) from .pte model
   - Convert to CGRect regions in Swift
   - Pass regions to Vision framework for OCR

3. Data Flow:
   ```
   Camera Frame → MaxSightCNN → text_head → text_regions (CGRect[])
   → VNRecognizeTextRequest → recognized_text → TTS → Audio Output
   ```

4. Performance Optimization:
   - Use Vision framework's on-device processing (Neural Engine)
   - Batch process multiple regions in single request
   - Cache results for static text (signs, labels)

5. Fallback Strategy:
   - If Vision framework fails, use model's text_head confidence
   - Provide bounding box coordinates for manual review
   - Log failures for model improvement

6. Integration Points:
   - MaxSightCNN.forward() → text_regions output
   - OCRIntegration.process_image_for_ocr() → iOS Vision wrapper
   - DescriptionGenerator → includes text in scene descriptions
   - CrossModalScheduler → prioritizes text reading alerts

7. Testing:
   - Unit tests: Mock Vision framework responses
   - Integration tests: Real device testing with various text types
   - Performance tests: Measure latency on iPhone Neural Engine

Benefits of iOS Vision Framework:
- Hardware acceleration via Neural Engine
- High accuracy (Apple's trained models)
- Language support (50+ languages)
- Real-time performance (<100ms for typical text)
- Privacy (on-device processing)

Migration Path:
1. Keep Python implementation for development/testing
2. Implement iOS wrapper in Swift
3. A/B test Python vs iOS for accuracy
4. Switch to iOS in production once validated
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

class OCRIntegration:
    """
    OCR integration for MaxSight - reads text from detected regions.
    
    For iOS: Uses Vision framework VNRecognizeTextRequest
    For Python: Uses fallback text extraction methods
    """
    
    def __init__(self, text_threshold: float = 0.5, confidence_threshold: float = 0.3):
        """
        Initialize OCR integration.
        
        Arguments:
            text_threshold: Threshold for text region detection from model
            confidence_threshold: Minimum confidence for OCR text recognition
        """
        self.text_threshold = text_threshold
        self.confidence_threshold = confidence_threshold
    
    def detect_text_regions_from_model(
        self,
        text_scores: torch.Tensor,
        boxes: torch.Tensor,
        image_size: Tuple[int, int] = (224, 224)
    ) -> List[Dict]:
        """
        Detect text regions from model's text_head output.
        
        Arguments:
            text_scores: Text probability scores [N] or [H, W]
            boxes: Bounding boxes [N, 4] in center format (cx, cy, w, h)
            image_size: Image dimensions (height, width)
        
        Returns:
            List of text region dicts with 'box', 'confidence', 'region_id'
        """
        text_regions = []
        
        # Handle different input shapes
        if text_scores.dim() == 2:  # [H, W] - spatial map
            # Find regions above threshold
            h, w = text_scores.shape
            y_coords, x_coords = torch.where(text_scores > self.text_threshold)
            
            if len(y_coords) > 0:
                # Group nearby pixels into regions (simple clustering)
                regions = self._cluster_text_pixels(x_coords, y_coords, h, w)
                
                for region_id, (x_min, y_min, x_max, y_max) in enumerate(regions):
                    # Convert to center format and normalize
                    cx = ((x_min + x_max) / 2) / w
                    cy = ((y_min + y_max) / 2) / h
                    width = (x_max - x_min) / w
                    height = (y_max - y_min) / h
                    
                    # Get average confidence
                    region_scores = text_scores[y_min:y_max+1, x_min:x_max+1]
                    confidence = float(region_scores.mean().item())
                    
                    text_regions.append({
                        'box': [cx, cy, width, height],
                        'confidence': confidence,
                        'region_id': region_id
                    })
        
        elif text_scores.dim() == 1 and boxes.shape[0] == text_scores.shape[0]:
            # [N] scores with matching boxes
            text_mask = text_scores > self.text_threshold
            text_boxes = boxes[text_mask]
            text_confidences = text_scores[text_mask]
            
            for i, (box, conf) in enumerate(zip(text_boxes, text_confidences)):
                text_regions.append({
                    'box': box.tolist() if isinstance(box, torch.Tensor) else box,
                    'confidence': float(conf.item()) if isinstance(conf, torch.Tensor) else conf,
                    'region_id': i
                })
        
        return text_regions
    
    def _cluster_text_pixels(
        self,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
        h: int,
        w: int,
        cluster_distance: int = 10,
        use_dbscan: bool = True
    ) -> List[Tuple[int, int, int, int]]:
        """
        Cluster text pixels into regions using DBSCAN (improved) or simple distance-based method.
        
        WHY DBSCAN:
        DBSCAN is more efficient (O(N log N) vs O(N²)) and handles irregularly shaped regions
        better than simple distance-based clustering. For large images with many text pixels,
        this provides significant performance improvement while maintaining accuracy.
        
        HOW IT SUPPORTS THE PROBLEM STATEMENT:
        Efficient text region detection enables real-time text reading, supporting the "Reads
        Environment" feature. Users need text detected quickly for practical use, not just
        accurate detection.
        
        Arguments:
            x_coords: X coordinates of text pixels
            y_coords: Y coordinates of text pixels
            h: Image height
            w: Image width
            cluster_distance: Maximum distance for clustering
            use_dbscan: Use DBSCAN for better performance (requires scikit-learn)
        
        Returns:
            List of (x_min, y_min, x_max, y_max) bounding boxes
        """
        if len(x_coords) == 0:
            return []
        
        # Convert to numpy for easier processing
        coords = torch.stack([x_coords, y_coords], dim=1).cpu().numpy()
        
        if use_dbscan:
            try:
                from sklearn.cluster import DBSCAN  # type: ignore
            except ImportError:
                raise RuntimeError(
                    "scikit-learn required for text region clustering. "
                    "Install: pip install scikit-learn"
                )
            
            # Use DBSCAN for efficient clustering
            # eps = cluster_distance in normalized coordinates
            # min_samples = 2 (at least 2 pixels per cluster)
            dbscan = DBSCAN(eps=cluster_distance, min_samples=2, metric='euclidean')
            labels = dbscan.fit_predict(coords)
            
            # Group pixels by cluster label
            regions = []
            unique_labels = set(labels)
            unique_labels.discard(-1)  # Remove noise label
            
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_coords = coords[cluster_mask]
                
                if len(cluster_coords) > 0:
                    x_min = int(cluster_coords[:, 0].min())
                    y_min = int(cluster_coords[:, 1].min())
                    x_max = int(cluster_coords[:, 0].max())
                    y_max = int(cluster_coords[:, 1].max())
                    
                    # Add padding
                    padding = 2
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w - 1, x_max + padding)
                    y_max = min(h - 1, y_max + padding)
                    
                    if x_max > x_min and y_max > y_min:
                        regions.append((x_min, y_min, x_max, y_max))
            
            return regions
        
        # Optimized distance-based clustering (fallback) using cKDTree for O(N log N) performance
        # Vectorized approach: use scipy.spatial.cKDTree if available, otherwise simple O(N²) fallback
        try:
            from scipy.spatial import cKDTree  # type: ignore
            
            # Build KD-tree for efficient nearest neighbor search
            tree = cKDTree(coords)
            regions = []
            used = set()
            
            for i, (x, y) in enumerate(coords):
                if i in used:
                    continue
                
                # Start new region
                cluster = [i]
                used.add(i)
                x_min, y_min, x_max, y_max = x, y, x, y
                
                # Find all neighbors within cluster_distance using KD-tree (O(log N) per query)
                neighbors = tree.query_ball_point((x, y), cluster_distance)
                
                for j in neighbors:
                    if j in used or j == i:
                        continue
                    cluster.append(j)
                    used.add(j)
                    x2, y2 = coords[j]
                    x_min = min(x_min, x2)
                    y_min = min(y_min, y2)
                    x_max = max(x_max, x2)
                    y_max = max(y_max, y2)
                
                # Add padding
                padding = 2
                x_min = max(0, int(x_min) - padding)
                y_min = max(0, int(y_min) - padding)
                x_max = min(w - 1, int(x_max) + padding)
                y_max = min(h - 1, int(y_max) + padding)
                
                if x_max > x_min and y_max > y_min:
                    regions.append((x_min, y_min, x_max, y_max))
            
            return regions
            
        except ImportError:
            # Fallback to simple O(N²) clustering if scipy not available
            regions = []
            used = set()
            
            for i, (x, y) in enumerate(coords):
                if i in used:
                    continue
                
                # Start new region
                cluster = [i]
                used.add(i)
                x_min, y_min, x_max, y_max = x, y, x, y
                
                # Find nearby pixels (O(N) per pixel)
                for j, (x2, y2) in enumerate(coords):
                    if j in used or j == i:
                        continue
                    
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if distance < cluster_distance:
                        cluster.append(j)
                        used.add(j)
                        x_min = min(x_min, x2)
                        y_min = min(y_min, y2)
                        x_max = max(x_max, x2)
                        y_max = max(y_max, y2)
                
                # Add padding
                padding = 2
                x_min = max(0, int(x_min) - padding)
                y_min = max(0, int(y_min) - padding)
                x_max = min(w - 1, int(x_max) + padding)
                y_max = min(h - 1, int(y_max) + padding)
                
                if x_max > x_min and y_max > y_min:
                    regions.append((x_min, y_min, x_max, y_max))
            
            return regions
    
    def extract_text_from_region(
        self,
        image: Image.Image,
        region_box: List[float],
        use_vision_framework: bool = False
    ) -> Tuple[Optional[str], float]:
        """
        Extract text from a specific image region with confidence score.
        
        WHY CONFIDENCE SCORING:
        Combining region detection confidence with OCR engine confidence provides a more
        meaningful measure of text readability. This enables filtering unreliable text
        extractions, supporting "Practical Usability" by ensuring only reliable information
        is presented to users.
        
        Arguments:
            image: PIL Image
            region_box: Bounding box [cx, cy, w, h] in normalized coordinates
            use_vision_framework: If True, use iOS Vision framework (requires iOS)
        
        Returns:
            Tuple of (extracted text, confidence score 0-1)
        """
        # Crop region from image
        # PIL Image.size is (width, height), not (height, width)
        w, h = image.size
        cx, cy, width, height = region_box
        
        # Convert center format to corner format
        x1 = int((cx - width / 2) * w)
        y1 = int((cy - height / 2) * h)
        x2 = int((cx + width / 2) * w)
        y2 = int((cy + height / 2) * h)
        
        # Clamp to image bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return (None, 0.0)
        
        # Crop region
        region_image = image.crop((x1, y1, x2, y2))
        
        if use_vision_framework:
            # iOS Vision framework integration (for iOS app)
            # This would call VNRecognizeTextRequest in Swift
            return self._extract_text_vision_framework(region_image)
        else:
            # Python fallback: OCR using pytesseract with confidence
            return self._extract_text_fallback(region_image)
    
    def _extract_text_vision_framework(self, image: Image.Image) -> Tuple[Optional[str], float]:
        """
        Extract text using iOS Vision framework.
        This is a placeholder - actual implementation in iOS app.
        
        WHY VISION FRAMEWORK:
        iOS Vision framework provides high-quality OCR with confidence scores, making it ideal
        for production use. This interface enables Python tests to simulate iOS Vision output
        for cross-platform unit testing.
        
        Returns:
            Tuple of (extracted text, confidence score 0-1)
        """
        # In iOS app, this would be:
        # let request = VNRecognizeTextRequest { request, error in
        #     guard let observations = request.results else { return }
        #     // Extract text from observations with confidence
        #     let confidence = observation.confidence
        # }
        # request.recognitionLevel = .accurate
        # try? VNImageRequestHandler(cgImage: image.cgImage!).perform([request])
        
        return (None, 0.0)  # Placeholder
    
    def _extract_text_fallback(
        self,
        image: Image.Image,
        use_adaptive_threshold: bool = True
    ) -> Tuple[Optional[str], float]:
        """
        Fallback text extraction for Python (development/testing).
        Uses pytesseract if available, otherwise returns placeholder.
        
        WHY ADAPTIVE PREPROCESSING:
        Adaptive thresholding and CLAHE improve OCR accuracy in low-contrast images, which
        are common in real-world scenarios (poor lighting, reflections, etc.). This directly
        supports "Practical Usability" by ensuring text extraction works in varied conditions.
        
        Arguments:
            image: PIL Image to extract text from
            use_adaptive_threshold: Use adaptive thresholding for better OCR
        
        Returns:
            Tuple of (extracted text, confidence score)
        """
        try:
            import pytesseract  # type: ignore
            
            # Preprocess image for better OCR
            gray = image.convert('L')
            
            if use_adaptive_threshold:
                try:
                    import cv2  # type: ignore
                    # Use adaptive thresholding for better results
                    gray_array = np.array(gray)
                    adaptive = cv2.adaptiveThreshold(
                        gray_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2
                    )
                    processed_image = Image.fromarray(adaptive)
                except (ImportError, Exception):
                    # Fallback to CLAHE or simple threshold
                    try:
                        import cv2  # type: ignore
                        gray_array = np.array(gray)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        enhanced = clahe.apply(gray_array)
                        processed_image = Image.fromarray(enhanced)
                    except (ImportError, Exception):
                        # Final fallback: simple threshold
                        threshold = 128
                        def threshold_func(p: int) -> int:
                            return 255 if p > threshold else 0
                        processed_image = gray.point(threshold_func, mode='1')
            else:
                # Simple threshold
                threshold = 128
                def threshold_func(p: int) -> int:
                    return 255 if p > threshold else 0
                processed_image = gray.point(threshold_func, mode='1')
            
            # Extract text with confidence
            try:
                # Try to get confidence scores
                ocr_data = pytesseract.image_to_data(
                    processed_image,
                    output_type=pytesseract.Output.DICT,
                    config='--psm 7'
                )
                
                # Extract text and confidence
                texts = [t for t in ocr_data['text'] if t.strip()]
                confidences = [c for c, t in zip(ocr_data['conf'], ocr_data['text']) if t.strip()]
                
                if texts:
                    text = ' '.join(texts)
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    # Normalize confidence to 0-1
                    confidence = max(0.0, min(1.0, avg_confidence / 100.0))
                    return (text.strip() if text.strip() else None, confidence)
                else:
                    return (None, 0.0)
            
            except Exception:
                # Fallback to simple extraction
                text = pytesseract.image_to_string(processed_image, config='--psm 7')
                return (text.strip() if text.strip() else None, 0.5)  # Default confidence
        
        except ImportError:
            # pytesseract not available - return placeholder
            return ("[Text detected - install pytesseract for extraction]", 0.0)
        except Exception as e:
            print(f"OCR extraction error: {e}")
            return (None, 0.0)
    
    def process_image_for_ocr(
        self,
        image: Image.Image,
        text_scores: torch.Tensor,
        boxes: torch.Tensor,
        max_regions: int = 10
    ) -> List[Dict]:
        """
        Complete OCR pipeline: detect regions and extract text.
        
        Arguments:
            image: PIL Image
            text_scores: Text detection scores from model
            boxes: Bounding boxes from model
            max_regions: Maximum number of text regions to process
        
        Returns:
            List of dicts with 'box', 'text', 'confidence', 'region_id'
        """
        # Detect text regions
        text_regions = self.detect_text_regions_from_model(text_scores, boxes)
        
        # Sort by confidence and limit
        text_regions.sort(key=lambda x: x['confidence'], reverse=True)
        text_regions = text_regions[:max_regions]
        
        # Extract text from each region with combined confidence (parallel processing)
        results = []
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.extract_text_from_region, image, region['box']): region
                for region in text_regions
            }
            
            for future in as_completed(futures):
                region = futures[future]
                try:
                    text, ocr_confidence = future.result(timeout=5.0)
                    if text:
                        # Combine region detection confidence with OCR confidence
                        # WHY: More meaningful confidence measure - both detection and extraction matter
                        combined_confidence = (region['confidence'] * 0.5 + ocr_confidence * 0.5)
                        
                        results.append({
                            'box': region['box'],
                            'text': text,
                            'confidence': combined_confidence,  # Combined confidence
                            'detection_confidence': region['confidence'],
                            'ocr_confidence': ocr_confidence,
                            'region_id': region['region_id']
                        })
                except Exception as e:
                    import logging
                    logging.warning(f"OCR failed for region {region.get('region_id', 'unknown')}: {e}")
        
        return results

def create_text_description(text_results: List[Dict], verbosity: str = 'normal') -> str:
    """
    Create natural language description of detected text with line/block grouping.
    
    WHY LINE/BLOCK GROUPING:
    Connected text (lines, paragraphs) should be described together, not split into
    individual words. This provides more natural descriptions and supports better
    understanding of text context.
    
        Arguments:
        text_results: List of OCR results from process_image_for_ocr
        verbosity: 'brief', 'normal', or 'detailed'
    
    Returns:
        Natural language description
    """
    if not text_results:
        return "No text detected"
    
    # Group text by proximity (line/block grouping)
    # WHY: Prevents splitting connected text into multiple regions
    grouped_texts = _group_text_by_proximity(text_results)
    
    if verbosity == 'brief':
        return f"Text detected: {len(grouped_texts)} region(s)"
    
    elif verbosity == 'normal':
        if len(grouped_texts) == 1:
            return f"Text: {grouped_texts[0]['text']}"
        else:
            texts = [g['text'] for g in grouped_texts[:3]]
            return f"Text detected: {', '.join(texts)}"
    
    else:  # detailed
        descriptions = []
        for i, group in enumerate(grouped_texts[:5], 1):
            pos = "left" if group['box'][0] < 0.33 else ("right" if group['box'][0] > 0.67 else "center")
            distance = "near" if group['box'][1] < 0.33 else ("far" if group['box'][1] > 0.67 else "center")
            descriptions.append(f"Text {i} ({pos}, {distance}): {group['text']}")
        return "; ".join(descriptions)

def _group_text_by_proximity(text_results: List[Dict], proximity_threshold: float = 0.1) -> List[Dict]:
    """
    Group text regions by spatial proximity (line/block grouping).
    
    WHY THIS FUNCTION:
    Prevents splitting connected text (lines, paragraphs) into multiple regions. This
    provides more natural text descriptions and better context understanding.
    
        Arguments:
        text_results: List of OCR results
        proximity_threshold: Maximum distance for grouping (normalized)
    
    Returns:
        List of grouped text results
    """
    if not text_results:
        return []
    
    groups = []
    used = set()
    
    for i, result in enumerate(text_results):
        if i in used:
            continue
        
        # Start new group
        group = [result]
        used.add(i)
        box1 = result['box']
        cx1, cy1 = box1[0], box1[1]
        
        # Find nearby text regions (likely same line/block)
        for j, other in enumerate(text_results):
            if j in used or j == i:
                continue
            
            box2 = other['box']
            cx2, cy2 = box2[0], box2[1]
            
            # Check if vertically aligned (same line) or horizontally close (same block)
            vertical_distance = abs(cy1 - cy2)
            horizontal_distance = abs(cx1 - cx2)
            
            if vertical_distance < proximity_threshold or horizontal_distance < proximity_threshold:
                group.append(other)
                used.add(j)
        
        # Combine text in group
        combined_text = ' '.join([r['text'] for r in group])
        avg_confidence = sum(r['confidence'] for r in group) / len(group)
        
        # Calculate group center
        avg_cx = sum(r['box'][0] for r in group) / len(group)
        avg_cy = sum(r['box'][1] for r in group) / len(group)
        
        groups.append({
            'text': combined_text,
            'confidence': avg_confidence,
            'box': [avg_cx, avg_cy, 0.1, 0.1],  # Approximate group size
            'region_count': len(group)
        })
    
    return groups

def read_text_aloud(text: str) -> None:
    """
    Read text aloud using TTS (text-to-speech).
    
    WHY TTS INTEGRATION:
    Text-to-speech enables users to hear text content, supporting "Clear Multimodal
    Communication" by making textual information accessible through audio. This is critical
    for users with vision impairments who cannot read text visually.
    
    NOTE: In iOS app, this would use AVSpeechSynthesizer. This Python version is for
    testing/development purposes.
    
        Arguments:
        text: Text to read aloud
    """
    try:
        import pyttsx3  # type: ignore
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except ImportError:
        # pyttsx3 not available - print text instead
        print(f"TTS: {text}")
    except Exception as e:
        print(f"TTS error: {e}")
        print(f"Text: {text}")

if __name__ == "__main__":
    # Test OCR integration
    print("OCR Integration Module Test")
    print("=" * 50)
    
    ocr = OCRIntegration(text_threshold=0.5)
    
    # Create dummy text detection output
    dummy_text_scores = torch.rand(14, 14) * 0.3  # Low scores
    dummy_text_scores[5:8, 5:8] = 0.8  # Text region
    dummy_boxes = torch.tensor([[0.5, 0.5, 0.1, 0.1]])  # Center box
    
    regions = ocr.detect_text_regions_from_model(dummy_text_scores, dummy_boxes)
    print(f"Detected {len(regions)} text regions")
    
    for region in regions:
        print(f"  Region {region['region_id']}: confidence={region['confidence']:.2f}, box={region['box']}")

