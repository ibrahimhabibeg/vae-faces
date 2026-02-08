from simple_parsing import ArgumentParser
from dataclasses import dataclass
import torch
from face_vae.model import VAE, Encoder, Decoder, Prior, ConvEncoderNet, DecoderNet
from face_vae.morph import morph_images
from torchvision import transforms
from PIL import Image

@dataclass
class MorphConfig:
    """
    Configuration for morphing between two images in the latent space of the VAE.
    """

    img1_path: str = (
        "../data/celebamask-hq-processed/02978.png"  # Path to the first image
    )
    img2_path: str = (
        "../data/celebamask-hq-processed/08543.png"  # Path to the second image
    )
    fps: int = 30  # Frames per second for the output morphing sequence
    duration: int = 5  # Duration of the morphing sequence in seconds
    output_path: str = "morph.gif"  # Path to save the output morphing GIF
    vae_checkpoint: str = (
        "../checkpoints/vae_epoch_20.pt"  # Path to the trained VAE checkpoint
    )
    img_size: int = 64  # Size to which input images should be resized. Must match the size used during VAE training.


def load_model(config: MorphConfig) -> VAE:
    with torch.serialization.safe_globals([]):
        checkpoint = torch.load(config.vae_checkpoint, weights_only=False)
    print(f"Loaded VAE checkpoint from {config.vae_checkpoint}")
    latent_dim = checkpoint["model_state_dict"]["encoder.head_mu.weight"].shape[0]
    encoder_net = ConvEncoderNet()
    encoder = Encoder(
        encoder_net=encoder_net,
        latent_dim=latent_dim,
    )

    decoder_net = DecoderNet(latent_dim=latent_dim)
    decoder = Decoder(decoder_net=decoder_net)

    prior = Prior()

    vae = VAE(encoder=encoder, decoder=decoder, prior=prior)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()
    return vae


def load_and_preprocess_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ]
    )
    return transform(image)  # type: ignore


def save_morph_gif(morph_sequence: list[torch.Tensor], output_path: str, fps: int):
    images = []
    for img_tensor in morph_sequence:
        img_array = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        images.append(Image.fromarray(img_array))
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=1000 / fps,
        loop=0,
    )


def create_morph_gif(config: MorphConfig):
    """
    Create a morphing GIF between two images using a trained VAE model.
    Args:
        config: MorphConfig object containing all necessary parameters for morphing.
    """
    vae = load_model(config)

    img1 = load_and_preprocess_image(config.img1_path)
    img2 = load_and_preprocess_image(config.img2_path)

    steps = config.fps * config.duration
    morph_sequence = morph_images(vae, img1, img2, steps)

    save_morph_gif(morph_sequence, config.output_path, config.fps)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(MorphConfig, dest="config")
    args = parser.parse_args()
    create_morph_gif(args.config)
