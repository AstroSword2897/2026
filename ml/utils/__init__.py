# MaxSight Utils Module - Data preprocessing and utility functions.

from ml.utils.preprocessing import ImagePreprocessor, DistanceEstimator, TextRegionDetector
from ml.utils.logging_config import setup_logging, get_logger
from ml.utils.clip_utils import clip_image_features_to_tensor

try:
    from ml.utils.ocr_integration import OCRIntegration, create_text_description, read_text_aloud
    from ml.utils.description_generator import DescriptionGenerator, create_description_generator
    from ml.utils.spatial_memory import SpatialMemory, SpatialObject
    from ml.utils.output_scheduler import CrossModalScheduler, OutputConfig, OutputChannel, AlertFrequency
    from ml.utils.semantic_grouping import SemanticGrouper, group_detections_semantically
    from ml.utils.adaptive_assistance import AdaptiveAssistance, create_adaptive_assistance_from_session
    from ml.utils.path_planning import PathPlanner, PathDirection, create_path_planner
    
    __all__ = [
        'ImagePreprocessor',
        'DistanceEstimator',
        'TextRegionDetector',
        'OCRIntegration',
        'create_text_description',
        'read_text_aloud',
        'DescriptionGenerator',
        'create_description_generator',
        'SpatialMemory',
        'SpatialObject',
        'CrossModalScheduler',
        'OutputConfig',
        'OutputChannel',
        'AlertFrequency',
        'SemanticGrouper',
        'group_detections_semantically',
        'AdaptiveAssistance',
        'create_adaptive_assistance_from_session',
        'PathPlanner',
        'PathDirection',
        'create_path_planner',
        'clip_image_features_to_tensor',
        'setup_logging',
        'get_logger'
    ]
except ImportError:
    __all__ = [
        'ImagePreprocessor',
        'DistanceEstimator',
        'TextRegionDetector',
        'clip_image_features_to_tensor',
        'setup_logging',
        'get_logger'
    ]






