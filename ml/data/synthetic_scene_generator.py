import torch
import torch.nn as nn
from typing import Tuple


class SyntheticSceneGenerator(nn.Module):

    def __init__(
        self,
        latent_dim: int = 128,
        output_size: Tuple[int, int] = (224, 224),
        normalize_output: bool = True  # Scale outputs to [0, 1].
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self._output_size = output_size
        self.normalize_output = normalize_output

        # Simple fully connected generator.
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_size[0] * output_size[1] * 3),
            nn.Tanh()  # Outputs in [-1, 1].
        )

    @property
    def output_size(self) -> Tuple[int, int]:
        """Get output size as (height, width)."""
        return self._output_size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Ensure batch dimension.
        if z.dim() == 1:
            z = z.unsqueeze(0)
        assert z.shape[1] == self.latent_dim, f"Expected latent dim {self.latent_dim}, got {z.shape[1]}"

        batch_size = z.size(0)
        x = self.generator(z).view(batch_size, 3, self._output_size[0], self._output_size[1])

        if self.normalize_output:
            # Scale from [-1,1] to [0,1].
            x = (x + 1.0) / 2.0

        return x


if __name__ == "__main__":
    # Example usage.
    generator = SyntheticSceneGenerator(latent_dim=128, output_size=(224, 224))
    z = torch.randn(16, 128)  # Batch of 16 latent vectors.
    images = generator(z)
    print(f"Generated images shape: {images.shape}")  # (16, 3, 224, 224)
    print(f"Min/Max pixel values: {images.min().item():.3f}/{images.max().item():.3f}")






