"""
Schema Validator, Downgrade Policy, and Stress Tests

Validates outputs against schema v1.1, enforces safety rules, and runs stress tests.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Validates outputs against accessibility output schema v1.1.
    Enforces safety rules and semantic clarity.
    """
    
    def __init__(self, strict: bool = True):
        """
        Initialize validator.
        
        Arguments:
            strict: If True, enforce all rules strictly. If False, allow warnings.
        """
        self.strict = strict
    
    def validate(self, outputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate outputs against schema v1.1.
        
        Arguments:
            outputs: Output dictionary to validate
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        required_fields = ['frame_id', 'timestamp', 'detections', 'scene_analysis']
        for field in required_fields:
            if field not in outputs:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Validate detections
        det_errors = self._validate_detections(outputs.get('detections', []))
        errors.extend(det_errors)
        
        # Validate scene_analysis
        scene_errors = self._validate_scene_analysis(outputs.get('scene_analysis', {}))
        errors.extend(scene_errors)
        
        # Validate functional_vision (if present)
        if 'functional_vision' in outputs:
            func_errors = self._validate_functional_vision(outputs['functional_vision'])
            errors.extend(func_errors)
        
        # Validate output_recommendations (if present)
        if 'output_recommendations' in outputs:
            output_errors = self._validate_output_recommendations(outputs['output_recommendations'])
            errors.extend(output_errors)
        
        # Check semantic clarity (no duplicate field names)
        clarity_errors = self._check_semantic_clarity(outputs)
        errors.extend(clarity_errors)
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _validate_detections(self, detections: List[Dict[str, Any]]) -> List[str]:
        """Validate detections array."""
        errors = []
        
        for i, det in enumerate(detections):
            # Check required fields
            required = ['class', 'class_name', 'confidence', 'box', 'distance', 'urgency', 'priority']
            for field in required:
                if field not in det:
                    errors.append(f"Detection {i}: Missing required field '{field}'")
            
            # Check confidence range
            if 'confidence' in det:
                conf = det['confidence']
                if not (0 <= conf <= 1):
                    errors.append(f"Detection {i}: confidence {conf} out of range [0, 1]")
            
            # Check confidence_sources (optional but recommended)
            if 'confidence_sources' in det:
                sources = det['confidence_sources']
                for source_name, source_conf in sources.items():
                    if not (0 <= source_conf <= 1):
                        errors.append(
                            f"Detection {i}: confidence_sources.{source_name} {source_conf} out of range [0, 1]"
                        )
        
        return errors
    
    def _validate_scene_analysis(self, scene: Dict[str, Any]) -> List[str]:
        """Validate scene_analysis."""
        errors = []
        
        # Check required fields
        required = ['scene_type', 'lighting_class']
        for field in required:
            if field not in scene:
                errors.append(f"scene_analysis: Missing required field '{field}'")
        
        # Check for old field names (warn or error)
        old_fields = {
            'contrast_sensitivity': 'scene_contrast_demand',
            'glare_risk_level': 'scene_glare_risk',
            'motion_perception_difficulty': 'scene_motion_difficulty'
        }
        
        for old_name, new_name in old_fields.items():
            if old_name in scene:
                if self.strict:
                    errors.append(
                        f"scene_analysis: Old field name '{old_name}' found. Use '{new_name}' instead."
                    )
                else:
                    logger.warning(
                        f"scene_analysis: Old field name '{old_name}' found. Use '{new_name}' instead."
                    )
        
        # Check for embedded images (should be references)
        if 'hazard_density_heatmap' in scene:
            errors.append(
                "scene_analysis: 'hazard_density_heatmap' embedded. Use 'hazard_density_heatmap_ref' instead."
            )
        
        return errors
    
    def _validate_functional_vision(self, functional: Dict[str, Any]) -> List[str]:
        """Validate functional_vision."""
        errors = []
        
        # Check for old field names
        old_fields = {
            'contrast_sensitivity': 'user_contrast_capacity',
            'glare_risk_level': 'user_glare_sensitivity',
            'motion_perception_difficulty': 'user_motion_capacity'
        }
        
        for old_name, new_name in old_fields.items():
            if old_name in functional:
                if self.strict:
                    errors.append(
                        f"functional_vision: Old field name '{old_name}' found. Use '{new_name}' instead."
                    )
                else:
                    logger.warning(
                        f"functional_vision: Old field name '{old_name}' found. Use '{new_name}' instead."
                    )
        
        return errors
    
    def _validate_output_recommendations(self, outputs: Dict[str, Any]) -> List[str]:
        """Validate output_recommendations with safety gating."""
        errors = []
        
        # output_validity is REQUIRED
        if 'output_validity' not in outputs:
            errors.append(
                "output_recommendations: Missing required field 'output_validity' "
                "(required for all action-oriented outputs)"
            )
            return errors  # Can't validate further without validity
        
        validity = outputs['output_validity']
        
        # Check required fields in output_validity
        if 'confidence' not in validity:
            errors.append("output_validity: Missing required field 'confidence'")
        if 'safe_to_act' not in validity:
            errors.append("output_validity: Missing required field 'safe_to_act'")
        
        # Enforce safety rules
        if 'confidence' in validity and 'safe_to_act' in validity:
            confidence = validity['confidence']
            safe_to_act = validity['safe_to_act']
            uncertainty = validity.get('uncertainty', 0.0)
            
            # Rule 1: safe_to_act must be false if confidence < 0.5
            if confidence < 0.5 and safe_to_act:
                errors.append(
                    f"output_validity: safe_to_act=true but confidence={confidence:.2f} < 0.5 "
                    "(violates safety rule)"
                )
            
            # Rule 2: safe_to_act must be false if uncertainty > 0.7
            if uncertainty > 0.7 and safe_to_act:
                errors.append(
                    f"output_validity: safe_to_act=true but uncertainty={uncertainty:.2f} > 0.7 "
                    "(violates safety rule)"
                )
        
        return errors
    
    def _check_semantic_clarity(self, outputs: Dict[str, Any]) -> List[str]:
        """Check for semantic clarity (no duplicate field names with different meanings)."""
        errors = []
        
        scene = outputs.get('scene_analysis', {})
        functional = outputs.get('functional_vision', {})
        
        # Check for field name overlap (excluding intentionally shared names)
        scene_fields = set(scene.keys())
        functional_fields = set(functional.keys())
        
        # These are intentionally different (scene vs user)
        allowed_overlap = set()  # No overlap allowed in v1.1
        
        overlap = (scene_fields & functional_fields) - allowed_overlap
        
        if overlap:
            errors.append(
                f"Semantic clarity violation: Field name overlap between scene_analysis and functional_vision: {overlap}. "
                "Use scene_* vs user_* prefixes for clarity."
            )
        
        return errors


class SchemaDowngrader:
    """
    Downgrades outputs to safe state when validation fails or heads are missing.
    
    Implements graceful degradation policy.
    """
    
    def __init__(self, min_confidence: float = 0.5, max_uncertainty: float = 0.7):
        """
        Initialize downgrader.
        
        Arguments:
            min_confidence: Minimum confidence for safe outputs
            max_uncertainty: Maximum uncertainty for safe outputs
        """
        self.min_confidence = min_confidence
        self.max_uncertainty = max_uncertainty
    
    def downgrade(self, outputs: Dict[str, Any], reason: str = "validation_failed") -> Dict[str, Any]:
        """
        Downgrade outputs to safe state.
        
        Arguments:
            outputs: Output dictionary to downgrade
            reason: Reason for downgrade
        
        Returns:
            Downgraded outputs dictionary
        """
        downgraded = outputs.copy()
        
        # Ensure output_validity exists and is safe
        if 'output_recommendations' not in downgraded:
            downgraded['output_recommendations'] = {}
        
        output_validity = downgraded['output_recommendations'].get('output_validity', {})
        
        # Set safe defaults
        output_validity['confidence'] = min(
            output_validity.get('confidence', 0.0),
            self.min_confidence - 0.1  # Force below threshold
        )
        output_validity['safe_to_act'] = False
        output_validity['uncertainty'] = max(
            output_validity.get('uncertainty', 0.0),
            self.max_uncertainty + 0.1  # Force above threshold
        )
        output_validity['downgrade_reason'] = reason
        
        downgraded['output_recommendations']['output_validity'] = output_validity
        
        # Remove action-oriented outputs
        if 'audio' in downgraded['output_recommendations']:
            downgraded['output_recommendations']['audio'] = {}
        if 'haptic' in downgraded['output_recommendations']:
            downgraded['output_recommendations']['haptic'] = {}
        
        logger.warning(f"Outputs downgraded: {reason}")
        
        return downgraded
    
    def downgrade_on_missing_heads(
        self,
        outputs: Dict[str, Any],
        missing_heads: List[str]
    ) -> Dict[str, Any]:
        """
        Downgrade outputs when heads are missing.
        
        Arguments:
            outputs: Output dictionary
            missing_heads: List of missing head names
        
        Returns:
            Downgraded outputs dictionary
        """
        downgraded = outputs.copy()
        
        # Update output_validity
        if 'output_recommendations' not in downgraded:
            downgraded['output_recommendations'] = {}
        
        output_validity = downgraded['output_recommendations'].get('output_validity', {})
        
        # Mark degraded modes
        if 'degraded_modes' not in output_validity:
            output_validity['degraded_modes'] = []
        output_validity['degraded_modes'].extend(missing_heads)
        
        # Increase uncertainty
        current_uncertainty = output_validity.get('uncertainty', 0.0)
        output_validity['uncertainty'] = min(1.0, current_uncertainty + 0.2 * len(missing_heads))
        
        # Check if we need to set safe_to_act = false
        if output_validity['uncertainty'] > self.max_uncertainty:
            output_validity['safe_to_act'] = False
        
        downgraded['output_recommendations']['output_validity'] = output_validity
        
        return downgraded


def validate_and_downgrade(
    outputs: Dict[str, Any],
    strict: bool = True,
    auto_downgrade: bool = True
) -> Tuple[Dict[str, Any], bool, List[str]]:
    """
    Convenience function to validate and optionally downgrade outputs.
    
    Arguments:
        outputs: Output dictionary to validate
        strict: Whether to enforce strict validation
        auto_downgrade: Whether to automatically downgrade on failure
    
    Returns:
        Tuple of (outputs, is_valid, errors)
    """
    validator = SchemaValidator(strict=strict)
    is_valid, errors = validator.validate(outputs)
    
    if not is_valid and auto_downgrade:
        downgrader = SchemaDowngrader()
        outputs = downgrader.downgrade(outputs, reason="validation_failed")
        is_valid = True  # Downgraded outputs are "valid" (safe)
    
    return outputs, is_valid, errors


# ============================================================================
# Schema Stress Tests
# ============================================================================

@dataclass
class SchemaStressTestResult:
    """Result of a schema stress test."""
    test_name: str
    passed: bool
    metrics: Dict[str, Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class SchemaStressTester:
    """Schema-specific stress tests for accessibility output schema."""
    
    def __init__(self, schema_path: Optional[str] = None):
        self.schema_path = schema_path
    
    def test_partial_head_dropout(
        self,
        outputs: Dict[str, Any],
        missing_heads: List[str]
    ) -> SchemaStressTestResult:
        """Test graceful degradation when heads are missing."""
        errors = []
        warnings = []
        metrics = {}
        
        test_outputs = self._remove_heads(outputs.copy(), missing_heads)
        
        if not self._validate_basic_structure(test_outputs):
            errors.append("Schema validation failed after head dropout")
        
        if 'detections' in test_outputs:
            for det in test_outputs['detections']:
                if 'confidence_sources' in det:
                    sources = det['confidence_sources']
                    for missing_head in missing_heads:
                        if missing_head in sources:
                            errors.append(f"Missing head '{missing_head}' still present in confidence_sources")
        
        if 'output_recommendations' in test_outputs:
            output_validity = test_outputs['output_recommendations'].get('output_validity')
            if output_validity:
                if 'degraded_modes' not in output_validity:
                    warnings.append("degraded_modes not set when heads are missing")
                if 'uncertainty' in output_validity:
                    metrics['uncertainty'] = output_validity['uncertainty']
                    if output_validity['uncertainty'] < 0.3:
                        warnings.append("Uncertainty did not increase after head dropout")
            else:
                errors.append("output_validity missing (required for output_recommendations)")
        
        return SchemaStressTestResult(
            test_name="Partial Head Dropout",
            passed=len(errors) == 0,
            metrics=metrics,
            errors=errors,
            warnings=warnings
        )
    
    def test_contradictory_signals(self, outputs: Dict[str, Any]) -> SchemaStressTestResult:
        """Test safety rules with contradictory signals."""
        errors = []
        warnings = []
        metrics = {}
        
        test_outputs = outputs.copy()
        if 'detections' in test_outputs and len(test_outputs['detections']) > 0:
            det = test_outputs['detections'][0]
            det['urgency'] = 3
            det['confidence'] = 0.3
        
        if 'output_recommendations' in test_outputs:
            output_validity = test_outputs['output_recommendations'].get('output_validity')
            if not output_validity:
                errors.append("output_validity missing")
            else:
                safe_to_act = output_validity.get('safe_to_act', True)
                confidence = output_validity.get('confidence', 1.0)
                uncertainty = output_validity.get('uncertainty', 0.0)
                
                if confidence < 0.5 and safe_to_act:
                    errors.append(f"safe_to_act=true but confidence={confidence:.2f} < 0.5")
                if uncertainty > 0.7 and safe_to_act:
                    errors.append(f"safe_to_act=true but uncertainty={uncertainty:.2f} > 0.7")
                
                metrics.update({'safe_to_act': safe_to_act, 'confidence': confidence, 'uncertainty': uncertainty})
        
        return SchemaStressTestResult(
            test_name="Contradictory Signals",
            passed=len(errors) == 0,
            metrics=metrics,
            errors=errors,
            warnings=warnings
        )
    
    def test_payload_size_explosion(self, outputs: Dict[str, Any]) -> SchemaStressTestResult:
        """Test payload size and serialization time."""
        errors = []
        warnings = []
        metrics = {}
        
        json_str = json.dumps(outputs)
        size_bytes = len(json_str.encode('utf-8'))
        size_kb = size_bytes / 1024
        metrics['size_bytes'] = size_bytes
        metrics['size_kb'] = size_kb
        
        start_time = time.perf_counter()
        for _ in range(100):
            json.dumps(outputs)
        serialization_time_ms = (time.perf_counter() - start_time) / 100 * 1000
        metrics['serialization_time_ms'] = serialization_time_ms
        
        if 'scene_analysis' in outputs and 'hazard_density_heatmap' in outputs['scene_analysis']:
            errors.append("hazard_density_heatmap embedded (should use hazard_density_heatmap_ref)")
        if 'spatial_mapping' in outputs and 'semantic_segmentation' in outputs['spatial_mapping']:
            errors.append("semantic_segmentation embedded (should use semantic_segmentation_ref)")
        
        if size_kb > 150:
            errors.append(f"Payload size {size_kb:.1f}KB exceeds 150KB threshold")
        elif size_kb > 100:
            warnings.append(f"Payload size {size_kb:.1f}KB exceeds 100KB recommendation")
        
        return SchemaStressTestResult(
            test_name="Payload Size Explosion",
            passed=len(errors) == 0,
            metrics=metrics,
            errors=errors,
            warnings=warnings
        )
    
    def test_semantic_drift_over_time(
        self,
        outputs_sequence: List[Dict[str, Any]],
        stable_scene: bool = True
    ) -> SchemaStressTestResult:
        """Test semantic stability over time."""
        errors = []
        warnings = []
        metrics = {}
        
        if len(outputs_sequence) < 10:
            errors.append("Need at least 10 frames")
            return SchemaStressTestResult(test_name="Semantic Drift", passed=False, errors=errors)
        
        user_contrast_values = []
        for output in outputs_sequence:
            if 'functional_vision' in output:
                functional = output['functional_vision']
                if 'user_contrast_capacity' in functional:
                    user_contrast_values.append(functional['user_contrast_capacity'])
        
        if len(user_contrast_values) > 1:
            user_variance = self._variance(user_contrast_values)
            metrics['user_contrast_variance'] = user_variance
            if user_variance > 0.05:
                errors.append(f"User contrast capacity variance {user_variance:.3f} too high")
        
        return SchemaStressTestResult(
            test_name="Semantic Drift Over Time",
            passed=len(errors) == 0,
            metrics=metrics,
            errors=errors,
            warnings=warnings
        )
    
    def _remove_heads(self, outputs: Dict[str, Any], missing_heads: List[str]) -> Dict[str, Any]:
        """Remove specified heads from outputs."""
        test_outputs = outputs.copy()
        if 'detections' in test_outputs:
            for det in test_outputs['detections']:
                if 'distance' in missing_heads:
                    det.pop('distance', None)
                if 'findability' in missing_heads or 'accessibility' in missing_heads:
                    det.pop('findability', None)
                    det.pop('recognizability', None)
        if 'output_recommendations' in test_outputs:
            output_validity = test_outputs['output_recommendations'].get('output_validity', {})
            if 'degraded_modes' not in output_validity:
                output_validity['degraded_modes'] = []
            output_validity['degraded_modes'].extend(missing_heads)
        return test_outputs
    
    def _validate_basic_structure(self, outputs: Dict[str, Any]) -> bool:
        """Validate basic schema structure."""
        required_fields = ['frame_id', 'timestamp', 'detections', 'scene_analysis']
        return all(field in outputs for field in required_fields)
    
    def _variance(self, values: List[float]) -> float:
        """Compute variance of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def run_all_tests(
        self,
        outputs: Dict[str, Any],
        outputs_sequence: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, SchemaStressTestResult]:
        """Run all schema stress tests."""
        results = {}
        results['partial_head_dropout'] = self.test_partial_head_dropout(
            outputs, missing_heads=['distance', 'findability']
        )
        results['contradictory_signals'] = self.test_contradictory_signals(outputs)
        results['payload_size'] = self.test_payload_size_explosion(outputs)
        if outputs_sequence:
            results['semantic_drift'] = self.test_semantic_drift_over_time(outputs_sequence)
        return results
    
    def generate_report(self, results: Dict[str, SchemaStressTestResult]) -> Dict[str, Any]:
        """Generate stress test report."""
        return {
            'summary': {
                'total_tests': len(results),
                'passed': sum(1 for r in results.values() if r.passed),
                'failed': sum(1 for r in results.values() if not r.passed),
                'total_errors': sum(len(r.errors) for r in results.values()),
                'total_warnings': sum(len(r.warnings) for r in results.values())
            },
            'tests': {
                name: {
                    'passed': result.passed,
                    'metrics': result.metrics,
                    'errors': result.errors,
                    'warnings': result.warnings
                }
                for name, result in results.items()
            }
        }

