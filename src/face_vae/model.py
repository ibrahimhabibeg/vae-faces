import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    A simple convolutional block: Conv2d -> BatchNorm2d -> LeakyReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvEncoderNet(nn.Module):
    """
    A convolutional encoder network that takes in an image and outputs a feature vector.
    Expected input shape: (batch_size, 3, H, W) where H and W are typically 64 for CelebA
    Output shape: (batch_size, 256 * H//16 * W//16)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(
            3, 32, kernel_size=4, stride=2, padding=1
        )  # 3x64x64 -> 32x32x32
        self.conv2 = ConvBlock(
            32, 64, kernel_size=4, stride=2, padding=1
        )  # 32x32x32 -> 64x16x16
        self.conv3 = ConvBlock(
            64, 128, kernel_size=4, stride=2, padding=1
        )  # 64x16x16 -> 128x8x8
        self.conv4 = ConvBlock(
            128, 256, kernel_size=4, stride=2, padding=1
        )  # 128x8x8 -> 256x4x4
        self.flatten = nn.Flatten()

    def forward(self, x):
        # x.shape should be (batch_size, 3, H, W) where H and W are expected to be 64 but not strictly required
        assert x.shape[1] == 3, "Expected input with 3 channels (RGB)"
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        return x


class Encoder(nn.Module):
    """
    Encoder module for VAE that takes in an image and outputs a latent vector.
    Defines q(z|x) = N(z; mu(x), diag(var(x))) where mu and var are predicted by the network.
    """

    def __init__(
        self,
        encoder_net: nn.Module,
        latent_dim: int,
        encoder_output_dim: int = 256 * 4 * 4,
    ):
        """
        Encoder module for VAE.

        Args:
            encoder_net (nn.Module): The convolutional encoder network.
            latent_dim (int): Dimension of the latent space.
            encoder_output_dim (int): Dimension of the encoder output. Default is 256 * 4 * 4.
        """
        super().__init__()
        self.encoder_net = encoder_net
        self.head_mu = nn.Linear(encoder_output_dim, latent_dim)
        # Predict log var instead of var because var must be positive while log var can be any real number.
        self.head_logvar = nn.Linear(encoder_output_dim, latent_dim)

    def reparametrize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) using N(0, 1).
        Necessary for backpropagation through the sampling step.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(std.device)  # eps ~ N(0, 1)
        return mu + eps * std  # Return sampled latent vector (z ~ N(mu, var))

    def forward(self, x):
        assert x.shape[1] == 3, "Expected input with 3 channels (RGB)"
        assert x.dim() == 4, "Expected input with shape (batch_size, 3, H, W)"
        enc_out = self.encoder_net(x)
        mu = self.head_mu(enc_out)
        logvar = self.head_logvar(enc_out)
        z = self.reparametrize(mu, logvar)
        return (
            z,
            mu,
            logvar,
        )  # Return the sampled latent vector (z) for reconstruction and mean and log variance for loss calculation

    def sample(self, x=None, mu=None, logvar=None):
        """
        Sample a latent vector from the encoder. Can be used for both training and inference.
        If x is provided, it will encode x to get mu and logvar. Otherwise, it will use the provided mu and logvar.
        """
        if x is not None:
            z, mu, logvar = self.forward(x)
            return z
        assert mu is not None and logvar is not None, (
            "Must provide either x or both mu and logvar"
        )
        z = self.reparametrize(mu, logvar)
        return z


class UpConvBlock(nn.Module):
    """
    A simple up-convolutional block: ConvTranspose2d -> BatchNorm2d -> LeakyReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.upconv(x)))


class DecoderNet(nn.Module):
    """
    A convolutional decoder network that takes in a latent vector and outputs an image.
    Expected input shape: (batch_size, latent_dim)
    Output shape: (batch_size, 3, 64, 64)
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.upconv1 = UpConvBlock(256, 128)  # 256x4x4 -> 128x8x8
        self.upconv2 = UpConvBlock(128, 64)  # 128x8x8 -> 64x16x16
        self.upconv3 = UpConvBlock(64, 32)  # 64x16x16 -> 32x32x32
        # No Relu or BatchNorm in the last layer
        self.upconv4 = nn.ConvTranspose2d(
            32, 3, kernel_size=4, stride=2, padding=1
        )  # 32x32x32 -> 3x64x64

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)  # Reshape to (batch_size, 256, 4, 4)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        return x


class Decoder(nn.Module):
    """
    Decoder module for VAE that takes in a latent vector and outputs a reconstructed image.
    p(x|z) definition depends on the loss function used.
    For binary cross-entropy loss, p(x|z) = Bernoulli(x; decoder(z)) where decoder(z) outputs the probability of each pixel being 1.
    For mean squared error loss, p(x|z) = Gaussian(x; decoder(z), I) where decoder(z) outputs the mean and the mean is bounded to [0, 1] using sigmoid activation.
    Categorical distribution isn't used due to computational constraints
    """

    def __init__(self, decoder_net: nn.Module):
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        assert z.dim() == 2, "Expected input with shape (batch_size, latent_dim)"
        recon_x = self.decoder_net(z)
        recon_x = torch.sigmoid(recon_x)  # Apply sigmoid to get pixel values in [0, 1]
        return recon_x  # Return recon_x with shape (batch_size, 3, 64, 64) and values in [0, 1]


class Prior(nn.Module):
    """
    Prior module for VAE that defines the prior distribution p(z).
    Uses a standard normal distribution N(0, I) as the prior.
    """

    def __init__(self):
        super().__init__()

    def forward(self, batch_size, latent_dim, device):
        return self.sample(batch_size, latent_dim, device)

    def sample(self, batch_size, latent_dim, device):
        return torch.randn(batch_size, latent_dim, device=device)


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class that combines the Encoder, Decoder, and Prior modules.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, prior: Prior):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def sample(self, batch_size, latent_dim, device):
        """
        Sample from the prior and decode to get a new image.
        """
        z = self.prior.sample(batch_size, latent_dim, device)
        recon_x = self.decoder(z)
        return recon_x

    def encode(self, x):
        """
        Encode an image to get its latent representation (mu and logvar).
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        assert x.shape[1] == 3, "Expected input with 3 channels (RGB)"
        _, mu, logvar = self.encoder(x)
        return mu.squeeze(0), logvar.squeeze(0)  # Remove batch dimension for single image input

    def decode(self, z):
        """
        Decode a latent vector to get the reconstructed image.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)  # Add batch dimension if missing
        assert z.dim() == 2, "Expected input with shape (batch_size, latent_dim)"
        recon_x = self.decoder(z)
        return recon_x.squeeze(0)  # Remove batch dimension for single image output
