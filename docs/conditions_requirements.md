# Vision Conditions Requirements Mapping

This document maps each of the 10 vision conditions to specific technical requirements and environmental structuring approaches.

## 1. Refractive Errors (Myopia, Hyperopia, Astigmatism, Presbyopia)

### Vision Impact
- Blurry vision at certain distances
- Cannot see fine details clearly
- Eye shape issues (too long/short) or irregular cornea

### Technical Requirements
- **Distance Detection**: Provide accurate distance estimates for objects
- **Fine Detail Enhancement**: High-resolution object detection for small items
- **Spatial Descriptions**: Clear verbal descriptions of object locations ("2 meters ahead", "arm's length away")
- **Magnification Support**: Optional zoom for text reading

### Environmental Structuring
- Prioritize distance information in descriptions
- Use consistent distance zones (near/medium/far)
- Emphasize fine details that may be missed

### Multimodal Output
- **Audio**: Distance-first descriptions ("Door 2 meters ahead")
- **Haptic**: Distance-based vibration intensity (closer = stronger)
- **Visual**: Distance overlay on detected objects

---

## 2. Cataracts

### Vision Impact
- Reduced visual acuity
- Cloudy/blurry vision
- Difficulty with contrast

### Technical Requirements
- **High-Contrast Overlays**: Increase contrast in visual displays
- **Enhanced Edge Detection**: Emphasize object boundaries
- **Brightness Adjustment**: Auto-adjust for optimal visibility
- **Text Enhancement**: High-contrast text rendering

### Environmental Structuring
- Use high-contrast color schemes
- Emphasize object boundaries and edges
- Provide clear spatial relationships

### Multimodal Output
- **Audio**: Clear, detailed descriptions
- **Visual**: High-contrast bounding boxes and labels
- **Haptic**: Standard patterns

---

## 3. Glaucoma

### Vision Impact
- Gradual loss of peripheral (side) vision
- Tunnel vision in advanced stages
- Can cause blindness if untreated

### Technical Requirements
- **Peripheral Priority Detection**: Prioritize objects in side vision zones
- **Side Obstacle Alerts**: Enhanced alerts for objects approaching from sides
- **Field-of-View Division**: Center vs. peripheral zone detection
- **Spatial Audio**: Directional audio for side objects

### Environmental Structuring
- Divide field of view into zones (center/peripheral)
- Prioritize peripheral obstacle descriptions
- Warn about side-approaching objects first

### Multimodal Output
- **Audio**: "Obstacle on your left", "Person approaching from right"
- **Haptic**: Left/right directional patterns for side objects
- **Visual**: Highlight peripheral detections prominently

---

## 4. AMD (Age-related Macular Degeneration)

### Vision Impact
- Blurred or dark central vision
- Straight lines appear wavy (wet AMD)
- Difficulty with detailed central vision

### Technical Requirements
- **Central Magnification**: Pinch-to-zoom for center of view
- **Edge Priority**: Prioritize describing objects at screen edges
- **Text Region Detection**: Auto-detect and magnify text areas
- **Peripheral Enhancement**: Boost detection of edge objects

### Environmental Structuring
- Magnify center region for detailed viewing
- Describe edge objects first (where vision is better)
- Provide text reading with magnification

### Multimodal Output
- **Audio**: "Text detected, magnifying center", "Object at screen edge"
- **Visual**: Auto-zoom on text regions, edge object highlighting
- **Haptic**: Standard patterns

---

## 5. Diabetic Retinopathy

### Vision Impact
- Blurry or spotty vision
- Dark patches or "floaters"
- Possible blindness in late stages

### Technical Requirements
- **Floater Compensation**: Detect and ignore dark spot artifacts
- **Confidence Boosting**: Increase detection confidence in affected regions
- **Verbal Emphasis**: More detailed verbal descriptions (reduce visual reliance)
- **Artifact Filtering**: Filter out false positives from floaters

### Environmental Structuring
- Compensate for visual artifacts
- Provide comprehensive verbal scene descriptions
- Boost confidence in detection despite visual noise

### Multimodal Output
- **Audio**: Detailed scene descriptions, object locations
- **Visual**: Filtered, artifact-free overlays
- **Haptic**: Standard patterns

---

## 6. Retinitis Pigmentosa

### Vision Impact
- Night blindness
- Gradual loss of side vision (tunnel vision)
- Difficulty in low light

### Technical Requirements
- **Low-Light Mode**: Automatic ISO/exposure adjustment
- **Brightness Enhancement**: Boost image brightness in dark conditions
- **Night Vision Simulation**: Enhanced contrast in low light
- **Audio Boost**: Increase audio alert volume in low light

### Environmental Structuring
- Auto-detect ambient light levels
- Enhance low-light image processing
- Prioritize audio alerts when vision is limited

### Multimodal Output
- **Audio**: Enhanced volume, clear descriptions
- **Visual**: Brightness-enhanced, high-contrast display
- **Haptic**: Stronger patterns in low light

---

## 7. Color Blindness

### Vision Impact
- Inability to distinguish certain colors accurately
- Commonly red-green mix-up
- Color confusion

### Technical Requirements
- **Color Detection**: Identify object colors in bounding boxes
- **Color Announcement**: Explicitly state colors ("Red car", "Green light")
- **Color Correction Filter**: Optional filter to adjust colors
- **Color Mapping**: Map RGB values to color names

### Environmental Structuring
- Always announce colors explicitly
- Use color information for navigation cues
- Provide color correction options

### Multimodal Output
- **Audio**: "Red stop sign", "Green traffic light"
- **Visual**: Color-coded labels, optional correction filter
- **Haptic**: Standard patterns

---

## 8. CVI (Cortical Visual Impairment)

### Vision Impact
- Inconsistent vision (some days clear, others not)
- Difficulty recognizing objects, faces, or movement
- Brain processing issues, not eye damage

### Technical Requirements
- **Simplified Descriptions**: Reduce complexity, use consistent format
- **Consistent Structure**: Same sentence patterns always
- **Single-Focus Mode**: Limit to one alert at a time
- **Visual Simplification**: Reduce overlay complexity

### Environmental Structuring
- Use simple, consistent language
- One object/alert at a time
- Predictable, structured format

### Multimodal Output
- **Audio**: Simple, consistent descriptions ("Door ahead. 2 meters.")
- **Visual**: Minimal, simplified overlays
- **Haptic**: Single pattern at a time

---

## 9. Amblyopia (Lazy Eye)

### Vision Impact
- Blurry or weak vision in one eye
- Depth perception loss
- Reduced 3D vision

### Technical Requirements
- **Explicit Depth Cues**: Verbal depth information
- **Distance Zones**: Clear near/medium/far classifications
- **Spatial Relationships**: Detailed spatial descriptions
- **No Visual Depth Assumption**: Don't rely on user's depth perception

### Environmental Structuring
- Provide explicit depth information
- Use clear distance zones
- Describe spatial relationships verbally

### Multimodal Output
- **Audio**: "Object very close", "Object arm's length", "Object far away"
- **Visual**: Distance overlays
- **Haptic**: Distance-based intensity

---

## 10. Strabismus (Crossed Eyes)

### Vision Impact
- Eyes point in different directions
- Double vision
- Suppression of one eye

### Technical Requirements
- **Single-Object Focus**: Clear, single-object descriptions
- **Explicit Depth Cues**: Verbal depth information (same as amblyopia)
- **No Double Vision Assumption**: Provide clear single focus
- **Spatial Clarity**: Clear left/right/center positioning

### Environmental Structuring
- Focus on one object at a time
- Provide clear spatial positioning
- Use explicit depth cues

### Multimodal Output
- **Audio**: "Single door ahead, 2 meters, handle on left"
- **Visual**: Single-object focus, clear positioning
- **Haptic**: Standard patterns

---

## Cross-Condition Requirements

### Universal Features
1. **Adjustable Verbosity**: Brief/Normal/Detailed modes
2. **Personal Labeling**: Custom object recognition
3. **Routine Adaptation**: Learn from usage patterns
4. **Multimodal Output**: Audio + Visual + Haptic always available
5. **Accessibility Compliance**: VoiceOver, Dynamic Type, High Contrast

### Priority System
- **Danger**: Vehicles, stairs, obstacles → Immediate alerts
- **Warning**: People, doors, signs → Moderate priority
- **Info**: Furniture, landmarks → Lower priority

### Performance Requirements
- **Latency**: <500ms inference time
- **Accuracy**: >85% object detection in varied environments
- **Battery**: <12% per hour normal use
- **Offline**: All core features work without internet
