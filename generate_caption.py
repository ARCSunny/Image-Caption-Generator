import argparse
import torch
from PIL import Image

from data.dataset import Vocabulary, eval_transform
from model.caption_model import ImageCaptionModel


def load_model(checkpoint_path: str, vocab_path: str, device):
    vocab = Vocabulary.load(vocab_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    model = ImageCaptionModel(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        dropout=cfg["dropout"],
        max_len=cfg.get("max_len", 80),  
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab


def caption_image(image_path: str, model, vocab, device, max_len: int = 30,
                   beam_width: int = 5) -> str:
    image = Image.open(image_path).convert("RGB")
    image_tensor = eval_transform(image).unsqueeze(0)
    if beam_width and beam_width > 1:
        caption = model.beam_search(image_tensor, vocab, beam_width=beam_width,
                                     max_len=max_len, device=device)
    else:
        caption = model.generate(image_tensor, vocab, max_len=max_len, device=device)
    return caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--vocab", type=str, default="checkpoints/vocab.pkl")
    parser.add_argument("--beam_width", type=int, default=5,
                         help="Beam search width. Use 1 for greedy decoding.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab = load_model(args.checkpoint, args.vocab, device)
    caption = caption_image(args.image, model, vocab, device, beam_width=args.beam_width)
    print(f"Caption: {caption}")


if __name__ == "__main__":
    main()
