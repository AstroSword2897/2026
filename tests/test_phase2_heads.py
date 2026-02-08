"""Comprehensive Tests for Phase 2: Advanced Multi-Task Heads."""

import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path

# Add parent directory to path.
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestOCRHead:
    """Test Transformer-Based OCR Head."""
    
    def test_ocr_head_import(self):
        """Test that OCR Head can be imported."""
        from ml.models.heads.ocr_head import TransformerOCRHead
        assert TransformerOCRHead is not None
    
    def test_ocr_head_forward(self):
        """Test OCR Head forward pass."""
        from ml.models.heads.ocr_head import TransformerOCRHead
        
        ocr_head = TransformerOCRHead(
            input_dim=256,
            vocab_size=100,
            max_text_length=50
        )
        ocr_head.eval()
        
        features = torch.randn(2, 10, 256)  # [B, N_regions, D].
        outputs = ocr_head(features)
        
        assert outputs is not None
        assert isinstance(outputs, dict)


class TestSceneDescriptionHead:
    """Test Scene Description Head."""
    
    def test_scene_desc_import(self):
        """Test that Scene Description Head can be imported."""
        from ml.models.heads.scene_description_head import SceneDescriptionHead
        assert SceneDescriptionHead is not None
    
    def test_scene_desc_forward(self):
        """Test Scene Description Head forward pass."""
        from ml.models.heads.scene_description_head import SceneDescriptionHead
        
        desc_head = SceneDescriptionHead(
            global_dim=512,
            region_dim=256,
            ocr_dim=256,
            vocab_size=5000,
            max_length=100
        )
        desc_head.eval()
        
        global_features = torch.randn(2, 512)
        region_features = torch.randn(2, 10, 256)
        ocr_features = torch.randn(2, 5, 256)
        
        description_logits = desc_head(global_features, region_features, ocr_features)
        
        assert description_logits is not None


class TestSoundEventHead:
    """Test Sound Event Classification Head."""
    
    def test_sound_event_import(self):
        """Test that Sound Event Head can be imported."""
        from ml.models.heads.sound_event_head import SoundEventHead
        assert SoundEventHead is not None
    
    def test_sound_event_forward(self):
        """Test Sound Event Head forward pass."""
        from ml.models.heads.sound_event_head import SoundEventHead
        
        sound_head = SoundEventHead(
            freq_bins=128,
            num_classes=15,  # Match actual default.
            num_directions=4,  # Match actual default.
            embed_dim=256
        )
        sound_head.eval()
        
        # Input is spectrogram: [B, T, freq_bins].
        spectrogram = torch.randn(2, 10, 128)  # [B, T, freq_bins].
        outputs = sound_head(spectrogram)
        
        assert outputs is not None
        assert isinstance(outputs, dict)
        assert 'sound_logits' in outputs
        assert 'sound_probs' in outputs
        assert 'direction_logits' in outputs
        assert 'priority' in outputs
        assert 'urgency' in outputs


class TestPersonalizationHead:
    """Test Personalization Head."""
    
    def test_personalization_import(self):
        """Test that Personalization Head can be imported."""
        from ml.models.heads.personalization_head import PersonalizationHead
        assert PersonalizationHead is not None
    
    def test_personalization_forward(self):
        """Test Personalization Head forward pass."""
        from ml.models.heads.personalization_head import PersonalizationHead
        
        personal_head = PersonalizationHead(
            input_dim=512,
            num_features=10,
            num_alert_types=5,
            embed_dim=128
        )
        personal_head.eval()
        
        scene_features = torch.randn(2, 512)  # [B, input_dim].
        user_id = torch.LongTensor([0, 1])  # [B] - required parameter.
        
        outputs = personal_head(scene_features, user_id)
        
        assert outputs is not None
        assert isinstance(outputs, dict)
        assert 'attention_logits' in outputs or 'attention' in outputs


class TestPredictiveAlertHead:
    """Test Predictive Alert Head."""
    
    def test_predictive_alert_import(self):
        """Test that Predictive Alert Head can be imported."""
        from ml.models.heads.predictive_alert_head import PredictiveAlertHead
        assert PredictiveAlertHead is not None
    
    def test_predictive_alert_forward(self):
        """Test Predictive Alert Head forward pass."""
        from ml.models.heads.predictive_alert_head import PredictiveAlertHead
        
        alert_head = PredictiveAlertHead(
            input_dim=512,
            motion_dim=256,
            num_hazard_types=10
        )
        alert_head.eval()
        
        scene_features = torch.randn(2, 512)
        motion_features = torch.randn(2, 256)  # [B, D] not [B, T, D].
        
        outputs = alert_head(scene_features, motion_features)
        
        assert outputs is not None
        assert isinstance(outputs, dict)


def run_all_tests():
    """Run all Phase 2 tests."""
    print("=" * 60)
    print("Phase 2: Advanced Multi-Task Heads Tests")
    print("=" * 60)
    
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_all_tests()







