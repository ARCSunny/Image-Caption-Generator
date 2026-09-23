import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

from data.dataset import FlickrDataset, CapCollate, train_transform, eval_transform
from model.caption_model import ImageCaptionModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", type=str, default="data/images")
    p.add_argument("--captions_csv", type=str, default="data/captions.txt")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--max_len", type=int, default=80,
                    help="Max caption length (in tokens) the model can handle, including "
                         "<sos>/<eos>. Must be >= your longest training caption + 2.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--freq_threshold", type=int, default=5)
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--fine_tune_cnn", action="store_true",
                    help="Fine-tune the ResNet backbone from the start")
    p.add_argument("--fine_tune_start_epoch", type=int, default=0,
                    help="Unfreeze the CNN backbone starting at this epoch "
                         "(0 = never auto-unfreeze; ignored if --fine_tune_cnn is set). "
                         "Uses --fine_tune_lr for the backbone once unfrozen.")
    p.add_argument("--fine_tune_lr", type=float, default=1e-5,
                    help="Learning rate for the CNN backbone once unfrozen.")
    p.add_argument("--early_stopping_patience", type=int, default=4,
                    help="Stop if val_loss hasn't improved for this many epochs (0 disables).")
    p.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint (e.g. checkpoints/last.pt) to resume training "
                         "from, picking up model/optimizer state and epoch count. Useful after "
                         "a Colab disconnect.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    full_dataset = FlickrDataset(
        image_dir=args.image_dir,
        captions_csv=args.captions_csv,
        transform=train_transform,
        freq_threshold=args.freq_threshold,
    )

    vocab_path = os.path.join(args.checkpoint_dir, "vocab.pkl")
    if args.resume and os.path.exists(vocab_path):
        
        from data.dataset import Vocabulary
        vocab = Vocabulary.load(vocab_path)
        full_dataset.vocab = vocab
        print(f"Resuming: loaded existing vocabulary ({len(vocab)} words) from {vocab_path}")
    else:
        vocab = full_dataset.vocab
        vocab.save(vocab_path)
    print(f"Vocabulary size: {len(vocab)}")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    
    longest_caption_tokens = max(
        len(full_dataset.vocab.numericalize(c)) for c in full_dataset.captions
    ) + 2  
    effective_max_len = max(args.max_len, longest_caption_tokens)
    if effective_max_len > args.max_len:
        print(f"Longest caption needs {longest_caption_tokens} tokens; "
              f"raising max_len from {args.max_len} to {effective_max_len}.")

    pad_idx = vocab.stoi["<pad>"]
    collate_fn = CapCollate(pad_idx=pad_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, collate_fn=collate_fn)

    
    model = ImageCaptionModel(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_len=effective_max_len,
        fine_tune_cnn=args.fine_tune_cnn,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=args.label_smoothing)

    def build_optimizer():
        """
        Rebuilt whenever the set of trainable parameters changes (i.e. when
        the CNN backbone gets unfrozen partway through training), so the
        newly-unfrozen backbone params get their own, lower learning rate
        while the rest of the model keeps using --lr.
        """
        decoder_params = list(model.decoder.parameters()) + list(model.encoder.projection.parameters())
        backbone_params = [p for p in model.encoder.backbone.parameters() if p.requires_grad]
        param_groups = [{"params": decoder_params, "lr": args.lr}]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": args.fine_tune_lr})
        return torch.optim.Adam(param_groups, weight_decay=args.weight_decay)

    optimizer = build_optimizer()

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    backbone_unfrozen = args.fine_tune_cnn

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        backbone_unfrozen = ckpt.get("backbone_unfrozen", args.fine_tune_cnn)
        if backbone_unfrozen:
            model.encoder.fine_tune(True)
            optimizer = build_optimizer()
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"Could not restore optimizer state ({e}); continuing with a fresh optimizer.")
        print(f"Resuming at epoch {start_epoch}, best_val_loss so far = {best_val_loss:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        
        if (not backbone_unfrozen and args.fine_tune_start_epoch > 0
                and epoch == args.fine_tune_start_epoch):
            print(f"Unfreezing CNN backbone at epoch {epoch} "
                  f"(backbone lr={args.fine_tune_lr})")
            model.encoder.fine_tune(True)
            optimizer = build_optimizer()
            backbone_unfrozen = True

        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for images, captions in loop:
            images, captions = images.to(device), captions.to(device)

            
            input_captions = captions[:, :-1]
            target_captions = captions[:, 1:]

            logits = model(images, input_captions, pad_idx=pad_idx)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), target_captions.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, captions in val_loader:
                images, captions = images.to(device), captions.to(device)
                input_captions = captions[:, :-1]
                target_captions = captions[:, 1:]
                logits = model(images, input_captions, pad_idx=pad_idx)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), target_captions.reshape(-1)
                )
                val_loss += loss.item()
        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

        improved = avg_val_loss < best_val_loss
        if improved:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "backbone_unfrozen": backbone_unfrozen,
            "config": {
                "embed_dim": args.embed_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "ff_dim": args.ff_dim,
                "dropout": args.dropout,
                "vocab_size": len(vocab),
                "max_len": effective_max_len,
            },
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, "last.pt"))
        if improved:
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
            print("  -> saved new best checkpoint")
        elif (args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience):
            print(f"No val_loss improvement for {epochs_without_improvement} epochs. "
                  f"Stopping early at epoch {epoch} (best val_loss={best_val_loss:.4f}).")
            break

    print("Training complete.")


if __name__ == "__main__":
    main()
