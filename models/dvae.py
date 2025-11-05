"""
Decoupled Variational Autoencoder (DVAE) for 3D Volume Artifact Correction

This model separates content features from artifact features in the latent space,
allowing reconstruction of artifact-free volumes from laminography input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """3D Convolutional block with BatchNorm and activation"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv3DTransposeBlock(nn.Module):
    """3D Transpose Convolutional block for upsampling"""

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    """
    3D Encoder that produces separate content and artifact feature representations

    Architecture progressively downsamples the input volume while extracting features.
    The final layer splits into two paths: content features and artifact features.
    """

    def __init__(self, in_channels=1, base_channels=32, latent_dim=128):
        super().__init__()

        # Shared encoder path
        self.enc1 = Conv3DBlock(in_channels, base_channels, stride=1)
        self.enc2 = Conv3DBlock(base_channels, base_channels * 2, stride=2)
        self.enc3 = Conv3DBlock(base_channels * 2, base_channels * 4, stride=2)
        self.enc4 = Conv3DBlock(base_channels * 4, base_channels * 8, stride=2)

        # Decoupled paths for content and artifact features
        self.content_path = nn.Sequential(
            Conv3DBlock(base_channels * 8, base_channels * 8),
            nn.AdaptiveAvgPool3d(1)
        )

        self.artifact_path = nn.Sequential(
            Conv3DBlock(base_channels * 8, base_channels * 8),
            nn.AdaptiveAvgPool3d(1)
        )

        # VAE latent space parameterization (mean and log variance)
        self.fc_content_mu = nn.Linear(base_channels * 8, latent_dim)
        self.fc_content_logvar = nn.Linear(base_channels * 8, latent_dim)

        self.fc_artifact_mu = nn.Linear(base_channels * 8, latent_dim)
        self.fc_artifact_logvar = nn.Linear(base_channels * 8, latent_dim)

    def forward(self, x):
        # Shared encoding
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoupled feature extraction
        content_feat = self.content_path(e4).squeeze(-1).squeeze(-1).squeeze(-1)
        artifact_feat = self.artifact_path(e4).squeeze(-1).squeeze(-1).squeeze(-1)

        # VAE parameterization
        content_mu = self.fc_content_mu(content_feat)
        content_logvar = self.fc_content_logvar(content_feat)

        artifact_mu = self.fc_artifact_mu(artifact_feat)
        artifact_logvar = self.fc_artifact_logvar(artifact_feat)

        return content_mu, content_logvar, artifact_mu, artifact_logvar


class Decoder(nn.Module):
    """
    3D Decoder that reconstructs volumes from ONLY content features

    This ensures the output is artifact-free, as artifact features are not used.
    """

    def __init__(self, latent_dim=128, base_channels=32, out_channels=1, spatial_size=8):
        super().__init__()

        self.spatial_size = spatial_size
        self.base_channels = base_channels

        # Project latent vector to initial spatial dimensions
        self.fc = nn.Linear(latent_dim, base_channels * 8 * spatial_size**3)

        # Upsampling path
        self.dec1 = Conv3DTransposeBlock(base_channels * 8, base_channels * 4)
        self.dec2 = Conv3DTransposeBlock(base_channels * 4, base_channels * 2)
        self.dec3 = Conv3DTransposeBlock(base_channels * 2, base_channels)

        # Final output layer
        self.final = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z_content):
        # Project and reshape
        x = self.fc(z_content)
        x = x.view(-1, self.base_channels * 8, self.spatial_size, self.spatial_size, self.spatial_size)

        # Upsample
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)

        # Output
        x = self.final(x)

        return x


class DVAE(nn.Module):
    """
    Complete Decoupled Variational Autoencoder for 3D volume artifact correction

    Args:
        in_channels: Number of input channels (1 for grayscale volumes)
        base_channels: Base number of feature channels
        latent_dim: Dimension of latent space
        patch_size: Expected input patch size (used to calculate decoder spatial size)
    """

    def __init__(self, in_channels=1, base_channels=32, latent_dim=128, patch_size=64):
        super().__init__()

        # Calculate spatial size after encoder downsampling
        # 3 stride-2 conv layers: patch_size -> patch_size/2 -> patch_size/4 -> patch_size/8
        spatial_size = patch_size // 8

        self.encoder = Encoder(in_channels, base_channels, latent_dim)
        self.decoder = Decoder(latent_dim, base_channels, in_channels, spatial_size)

    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        content_mu, content_logvar, artifact_mu, artifact_logvar = self.encoder(x)

        # Reparameterize
        z_content = self.reparameterize(content_mu, content_logvar)
        z_artifact = self.reparameterize(artifact_mu, artifact_logvar)

        # Decode using ONLY content features
        x_recon = self.decoder(z_content)

        return x_recon, content_mu, content_logvar, artifact_mu, artifact_logvar

    def encode(self, x):
        """Encode input to latent space"""
        content_mu, content_logvar, artifact_mu, artifact_logvar = self.encoder(x)
        z_content = self.reparameterize(content_mu, content_logvar)
        z_artifact = self.reparameterize(artifact_mu, artifact_logvar)
        return z_content, z_artifact

    def decode(self, z_content):
        """Decode from content latent vector"""
        return self.decoder(z_content)


def test_dvae():
    """Test DVAE architecture with a sample input"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = DVAE(in_channels=1, base_channels=32, latent_dim=128, patch_size=64).to(device)

    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 1, 64, 64, 64).to(device)

    # Forward pass
    x_recon, content_mu, content_logvar, artifact_mu, artifact_logvar = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_recon.shape}")
    print(f"Content latent shape: {content_mu.shape}")
    print(f"Artifact latent shape: {artifact_mu.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


if __name__ == "__main__":
    test_dvae()
