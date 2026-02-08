"""Custom exceptions for MaxSight system. Provides structured error handling with clear error messages and recovery guidance."""


class MaxSightError(Exception):
    """Base exception for MaxSight system."""
    pass


class PreprocessingError(MaxSightError):
    """Image preprocessing failed."""
    pass


class OCRError(MaxSightError):
    """Text extraction failed."""
    pass


class ModelError(MaxSightError):
    """Model inference failed."""
    pass


class SpatialMemoryError(MaxSightError):
    """Spatial memory operation failed."""
    pass


class OutputSchedulingError(MaxSightError):
    """Output scheduling failed."""
    pass


