import os
import re
import pickle
from collections import Counter

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Vocabulary:
    

    def __init__(self, freq_threshold: int = 5):
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text: str):
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9 ]", "", text)
        return text.split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4  

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text: str):
        tokens = self.tokenize(text)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_captions(captions_csv: str):
    
    df = pd.read_csv(captions_csv, on_bad_lines="skip", engine="python")
    if df.shape[1] == 1:
        for sep in ["|", "\t", ";"]:
            try:
                candidate = pd.read_csv(captions_csv, sep=sep, on_bad_lines="skip", engine="python")
                if candidate.shape[1] > 1:
                    df = candidate
                    break
            except Exception:
                continue

    df.columns = [c.strip().lower() for c in df.columns]

    image_aliases = ["image", "image_name", "filename", "file_name", "image_id", "img"]
    caption_aliases = ["caption", "comment", "text", "caption_text", "captions", "description"]

    image_col = next((c for c in image_aliases if c in df.columns), None)
    caption_col = next((c for c in caption_aliases if c in df.columns), None)

    if image_col is None or caption_col is None:
        raise ValueError(
            f"Could not find recognizable image/caption columns in {captions_csv}. "
            f"Found columns: {list(df.columns)}. "
            "Open the file and check its header, then add the real column "
            "names to `image_aliases` / `caption_aliases` in data/dataset.py."
        )

    df = df.rename(columns={image_col: "image", caption_col: "caption"})
    df["caption"] = df["caption"].astype(str).str.strip()
    df["image"] = df["image"].astype(str).str.strip()

    return df[["image", "caption"]]


class FlickrDataset(Dataset):
    def __init__(self, image_dir: str, captions_csv: str, vocab: Vocabulary = None,
                 transform=None, freq_threshold: int = 5):
        self.image_dir = image_dir
        self.df = load_captions(captions_csv)
        self.transform = transform or eval_transform

        self.images = self.df["image"].tolist()
        self.captions = self.df["caption"].tolist()

        if vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocabulary(self.captions)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        numeric_caption = [self.vocab.stoi["<sos>"]]
        numeric_caption += self.vocab.numericalize(caption)
        numeric_caption.append(self.vocab.stoi["<eos>"])

        return image, torch.tensor(numeric_caption)


class CapCollate:
    

    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)

        captions = [item[1] for item in batch]
        captions = torch.nn.utils.rnn.pad_sequence(
            captions, batch_first=True, padding_value=self.pad_idx
        )
        return images, captions
