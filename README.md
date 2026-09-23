# 🖼️ Image Caption Generator
<img width="774" height="531" alt="Capture" src="https://github.com/user-attachments/assets/4f01b9d3-f33c-4d51-b4aa-cd87c9d3aecf" />

An end-to-end **Image Captioning** application that generates a natural-language description for an uploaded image.

The project combines a **pretrained ResNet-50 CNN encoder** with a **Transformer decoder**. The CNN extracts visual features from the image, and the Transformer generates a caption token by token. During inference, the application uses **beam search** to improve caption generation compared with simple greedy decoding.

The project includes:

- A PyTorch image-captioning model
- ResNet-50 as the visual feature extractor
- Transformer decoder for natural-language generation
- Vocabulary construction and serialization
- Flickr30k-based training pipeline
- Training/validation split
- Checkpoint saving and training resume support
- Greedy decoding and beam-search decoding
- Flask REST API
- Browser-based drag-and-drop frontend
- CPU/GPU automatic device selection
- Saved trained model checkpoints

## 📌 Project Overview

### What does this project do?

The application follows this pipeline:

```
User uploads an image
        │
        ▼
Image preprocessing
(Resize + normalization)
        │
        ▼
ResNet-50 CNN Encoder
        │
        ▼
Visual feature map
        │
        ▼
Linear projection
        │
        ▼
Transformer Decoder
        │
        ▼
Beam Search
        │
        ▼
Generated caption
        │
        ▼
Displayed in the web interface
```

For example, an input image might produce a caption similar to:

> "a dog running through a field"

The exact caption depends on the training data, vocabulary, learned weights, and image content.

## ✨ Features

### 1. Image Upload

The web interface supports:

- Drag-and-drop image upload
- File selection through a browser
- Image preview
- Automatic caption generation
- Regeneration of the caption
- Uploading another image
- Error messages when the server or image is unavailable

### 2. Deep Learning Architecture

The model uses:

### Encoder
**ResNet-50**

- Pretrained on ImageNet
- The final classification layers are removed
- The resulting spatial feature map is converted into a sequence
- A linear projection maps the 2048-dimensional CNN features to the Transformer embedding dimension

### Decoder
**Transformer Decoder**

Default configuration:

| Parameter | Value |
|---|---:|
| Embedding dimension | 512 |
| Attention heads | 8 |
| Transformer layers | 4 |
| Feed-forward dimension | 1024 |
| Dropout | 0.2 |
| Maximum sequence length | 80 |
| Vocabulary threshold | 5 |

The training script can override these values through command-line arguments.

### 3. Beam Search

Inference uses:

```
Beam width = 5
Maximum generated length = 30 tokens
```

Beam search keeps multiple candidate captions while generating the sentence instead of committing to only the locally highest-probability token at every step.

The implementation also applies a length penalty when comparing candidate sequences.

### 4. Checkpointing

Training saves:

```
checkpoints/
├── best.pt
├── last.pt
└── vocab.pkl
```

### `best.pt`

The checkpoint with the best validation loss observed during training.

### `last.pt`

The latest checkpoint, including model and optimizer state, which can be used to resume training.

### `vocab.pkl`

The serialized vocabulary required to convert generated token IDs back into words.

**The model checkpoint and vocabulary must come from the same training run/configuration.**

---

## 🧠 Model Architecture

### Encoder: ResNet-50

The project initializes:

```
models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
```

The final two ResNet stages are removed from the classification network so that the model produces spatial visual features rather than ImageNet class probabilities.

The resulting feature tensor has the general form:

```
Batch × Channels × Height × Width
```

The code then changes it to:

```
Batch × (Height × Width) × Channels
```

and projects:

```
2048 → 512
```

This produces a sequence of visual embeddings that can be consumed by the Transformer decoder.

### Decoder: Transformer

The Transformer receives:

1. The previously generated caption tokens
2. The CNN visual features

The decoder uses:

- Token embeddings
- Sinusoidal positional encoding
- Causal self-attention
- Cross-attention over image features
- Feed-forward layers
- Padding masks
- A final linear layer over the vocabulary

During training, a causal mask prevents a token from attending to future caption tokens.

Conceptually:

```
Image
  │
  ▼
ResNet-50
  │
  ▼
Image features ──────────────┐
                             │
Caption tokens               │
  │                          │
  ▼                          ▼
Token Embedding        Transformer Decoder
                             │
                             ▼
                       Vocabulary logits
                             │
                             ▼
                      Next-token prediction
```

## 📚 Dataset

The training pipeline is designed around the **Flickr30k** image-caption dataset.

The provided training notebook uses the Flickr30k dataset and points to:

```
flickr30k_images/
└── flickr30k_images/

results.csv
```

The dataset contains images paired with natural-language captions.

The project does not hard-code a single CSV column name. `dataset.py` searches for common alternatives such as:

### Image columns

```
image
image_name
filename
file_name
image_id
img
```

### Caption columns

```
caption
comment
text
caption_text
captions
description
```

This makes the data loader more tolerant of different caption-file formats.

## 📝 Vocabulary

The `Vocabulary` class creates a word-to-index and index-to-word mapping.

Four special tokens are always present:

| Token | Purpose |
|---|---|
| `<pad>` | Padding shorter sequences |
| `<sos>` | Start of sentence |
| `<eos>` | End of sentence |
| `<unk>` | Unknown word |

The default frequency threshold is:

```
5
```

Only words occurring at least five times in the training captions are added to the vocabulary.

Words outside the vocabulary are represented using:

```
<unk>
```

The vocabulary is saved as:

```
checkpoints/vocab.pkl
```

## 🖼️ Image Preprocessing

### Training Transform

Training images use:

```
Resize → 256 × 256
Random Crop → 224 × 224
Random Horizontal Flip
ToTensor
ImageNet normalization
```

ImageNet normalization:

```
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

### Evaluation Transform

Inference images use:

```
Resize → 224 × 224
ToTensor
ImageNet normalization
```

The same evaluation transform is used when generating captions from individual images.

## 🏋️ Training

The main training script is:

```
train.py
```

It automatically selects:

```
CUDA GPU → if available
CPU     → otherwise
```

### Default training configuration

| Argument | Default |
|---|---:|
| Image directory | `data/images` |
| Captions file | `data/captions.txt` |
| Checkpoint directory | `checkpoints` |
| Embedding dimension | `512` |
| Attention heads | `8` |
| Transformer layers | `4` |
| Feed-forward dimension | `1024` |
| Dropout | `0.2` |
| Maximum length | `80` |
| Batch size | `32` |
| Epochs | `20` |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Label smoothing | `0.1` |
| Vocabulary frequency threshold | `5` |
| Validation split | `5%` |
| Early stopping patience | `4` |
| Fine-tune learning rate | `1e-5` |

The actual maximum sequence length is automatically increased if the training dataset contains a caption longer than the requested maximum.

## 🔧 Training Strategy

The CNN backbone can initially remain frozen while the Transformer and feature projection layers learn the captioning task.

The provided notebook uses:

```
--fine_tune_start_epoch 8
```

This means the ResNet backbone is automatically unfrozen starting at epoch 8.

The lower learning rate for the CNN backbone is:

```
1e-5
```

while the rest of the model uses:

```
3e-4
```

This allows the pretrained visual representation to be fine-tuned more conservatively.

## 💾 Resume Training

The training script supports resuming from a saved checkpoint.

This is particularly useful when training is interrupted by:

- Colab/Kaggle runtime disconnections
- Computer shutdown
- GPU session limits
- Network/session problems
- Manual interruption

Example:

```
python train.py \
    --image_dir data/images \
    --captions_csv data/captions.txt \
    --checkpoint_dir checkpoints \
    --resume checkpoints/last.pt
```

The checkpoint stores:

- Model weights
- Optimizer state
- Epoch number
- Best validation loss
- CNN fine-tuning state
- Model configuration

## 🧪 Training Notebook

The repository contains:

```
training.ipynb
```

The notebook was prepared for a Kaggle environment and performs the following operations:

1. Inspect available Kaggle datasets
2. Copy the project into the Kaggle working directory
3. Install required lightweight dependencies
4. Configure Flickr30k paths
5. Train the captioning model
6. Generate a caption for a sample image
7. Zip the resulting checkpoints

The notebook uses the Flickr30k dataset and trains for up to 20 epochs with:

```
--batch_size 32
--fine_tune_start_epoch 8
--early_stopping_patience 4
```

## 🗂️ Recommended GitHub Project Structure

For the current Python imports and Flask application to work cleanly, the repository should use a package structure similar to:

```
image-caption-generator/
│
├── app.py
├── generate_caption.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   ├── __init__.py
│   └── caption_model.py
│
├── data/
│   ├── __init__.py
│   └── dataset.py
│
├── templates/
│   └── index.html
│
├── checkpoints/
│   ├── best.pt
│   ├── last.pt
│   └── vocab.pkl
│
├── data/
│   ├── images/
│   └── captions.txt
│
└── training.ipynb
```
If the repository is already organized this way, no change is necessary.

## 🔌 API

The application exposes:

```
POST /api/caption
```

The request must contain an image in a multipart form field named:

```
image
```

## Example using cURL

```
curl -X POST \
  -F "image=@path/to/image.jpg" \
  http://localhost:5000/api/caption
```

Successful response:

```
{
  "caption": "generated caption goes here"
}
```

If no image is uploaded:

```
{
  "error": "No image uploaded"
}
```

If the uploaded file cannot be read:

```
{
  "error": "Could not read image file"
}
```

If the trained checkpoint or vocabulary is unavailable, the API returns HTTP status `503`.

## 🖥️ Frontend

The browser interface is a lightweight HTML/CSS/JavaScript application.

The frontend:

1. Accepts an image
2. Shows a preview
3. Sends the image to `/api/caption`
4. Receives the JSON response
5. Displays the generated caption with a typing animation
6. Allows the user to regenerate the caption
7. Allows another image to be selected

The frontend communicates with Flask using:

```
fetch('/api/caption', {
    method: 'POST',
    body: formData
});
```

No separate frontend framework is required.

## 🧪 Command-Line Caption Generation

Captions can also be generated without starting Flask.

Use:

```
python generate_caption.py \
    --image path/to/image.jpg \
    --checkpoint checkpoints/best.pt \
    --vocab checkpoints/vocab.pkl
```

Example:

```
python generate_caption.py \
    --image test.jpg \
    --checkpoint checkpoints/best.pt \
    --vocab checkpoints/vocab.pkl
```

The default beam width is:

```
5
```

## 🎯 Greedy Decoding vs Beam Search

The project supports two decoding methods.

### Greedy decoding

Set:

```
--beam_width 1
```

The model selects the highest-probability next token at every step.

### Beam search

Use:

```
--beam_width 5
```

Beam search keeps several candidate sequences during generation.

The larger the beam width, the more candidate sequences are explored, but inference can become slower.

## 📦 Checkpoints

The trained checkpoints are important deployment artifacts.

Expected files:

```
checkpoints/
├── best.pt
├── last.pt
└── vocab.pkl
```

The provided trained files include:

- `best.pt` — best validation-loss model
- `last.pt` — latest training checkpoint
- `vocab.pkl` — vocabulary

### Do not rename the files without updating the application configuration.

By default, Flask expects:

```
checkpoints/best.pt
checkpoints/vocab.pkl
```

The application also supports environment variables:

```
CHECKPOINT_PATH
VOCAB_PATH
```

For example:

### Windows PowerShell

```
$env:CHECKPOINT_PATH="checkpoints/best.pt"
$env:VOCAB_PATH="checkpoints/vocab.pkl"
python app.py
```

### Linux/macOS

```
export CHECKPOINT_PATH="checkpoints/best.pt"
export VOCAB_PATH="checkpoints/vocab.pkl"
python app.py
```

## 🌐 Online Deployment

This project is designed to expose a Flask web application, so it can be deployed to a cloud platform that supports Python/Flask applications.

The deployment environment should provide:

- Python
- PyTorch
- Torchvision
- Flask
- Pillow
- The trained checkpoint
- The vocabulary file
- Sufficient RAM for model inference
- A publicly accessible HTTP port

### Production server

The current `app.py` uses Flask's built-in development server:

```
app.run(debug=True, host="0.0.0.0", port=5000)
```

This is suitable for local development, but a production deployment should normally use a production WSGI server such as Gunicorn where supported.

For example:

```bash
gunicorn app:app
```

If using Gunicorn, add it to `requirements.txt`:

```text
gunicorn
```

The exact start command depends on the hosting provider.

## 📈 Training Details

The training loop performs standard autoregressive language-model training.

For each caption:

```
<sos> a dog runs <eos>
```

the input and target sequences are shifted:

```
Input:
<sos> a dog runs

Target:
a dog runs <eos>
```

The model predicts the next token at every position.

The loss is calculated using:

```
nn.CrossEntropyLoss(
    ignore_index=pad_idx,
    label_smoothing=label_smoothing
)
```

Padding tokens do not contribute to the loss.

Gradient clipping is also applied:

```
max_norm = 5.0
```

## 🛑 Early Stopping

The training process monitors validation loss.

When validation loss improves:

```
best.pt
```

is updated.

If validation loss does not improve for the configured number of epochs, training stops.

Default:

```
early_stopping_patience = 4
```

This reduces unnecessary training once validation performance stops improving.

## 🔄 Fine-Tuning the CNN

There are two ways to fine-tune the ResNet backbone.

### Fine-tune from the beginning

```
python train.py --fine_tune_cnn
```

### Unfreeze at a later epoch

Example:

```
python train.py --fine_tune_start_epoch 8
```

The second approach keeps the pretrained CNN frozen initially and begins fine-tuning later using the smaller:

```
fine_tune_lr = 1e-5
```

## 🧩 Main Files

| File | Purpose |
|---|---|
| `app.py` | Flask web server and REST API |
| `generate_caption.py` | Standalone inference script |
| `train.py` | Model training pipeline |
| `caption_model.py` | CNN encoder + Transformer decoder |
| `dataset.py` | Dataset, vocabulary, transforms and batching |
| `index.html` | Browser user interface |
| `training.ipynb` | Kaggle training workflow |
| `requirements.txt` | Python dependencies |
| `checkpoints/best.pt` | Best trained model |
| `checkpoints/last.pt` | Latest training checkpoint |
| `checkpoints/vocab.pkl` | Saved vocabulary |


## 🧱 Core Classes

### `CNNEncoder`

Located in:

```
model/caption_model.py
```

Responsibilities:

- Load ResNet-50
- Remove the final classification layers
- Extract spatial image features
- Project 2048-dimensional features into the Transformer embedding space
- Optionally fine-tune the CNN backbone

### `TransformerDecoder`

Responsibilities:

- Token embedding
- Positional encoding
- Causal masking
- Transformer decoding
- Padding masking
- Vocabulary prediction

### `ImageCaptionModel`

Combines:

```
CNNEncoder + TransformerDecoder
```

and provides:

- Forward pass
- Greedy caption generation
- Beam-search caption generation

### `Vocabulary`

Located in:

```
data/dataset.py
```

Responsibilities:

- Tokenization
- Vocabulary construction
- Numericalization
- Saving/loading vocabulary mappings

### `FlickrDataset`

Responsibilities:

- Read image/caption pairs
- Load images
- Apply transformations
- Convert captions into token IDs

### `CapCollate`

Responsibilities:

- Combine image tensors into batches
- Pad captions to a common sequence length

## 🧰 Troubleshooting

### `FileNotFoundError: No trained checkpoint found`

Make sure:

```
checkpoints/best.pt
checkpoints/vocab.pkl
```

exist.

Or configure:

```
CHECKPOINT_PATH
VOCAB_PATH
```

with the correct paths.

### `ModuleNotFoundError: No module named 'data'`

The current Python imports expect:

```
data/
└── dataset.py
```

Make sure `dataset.py` is inside the `data` package and that the package contains:

```
data/__init__.py
```

### `ModuleNotFoundError: No module named 'model'`

The current code expects:

```
model/
└── caption_model.py
```

Create:

```
model/__init__.py
```

and place `caption_model.py` inside the `model` directory.

### `TemplateNotFound: index.html`

Flask's:

```
render_template("index.html")
```

expects:

```
templates/index.html
```

Therefore the recommended structure is:

```
templates/
└── index.html
```

### `CUDA out of memory`

Try a smaller batch size during training:

```
python train.py --batch_size 16
```

or:

```
python train.py --batch_size 8
```

For inference, CPU mode is automatically used when CUDA is unavailable.

### Training is interrupted

Resume from:

```
checkpoints/last.pt
```

Example:

```
python train.py \
    --resume checkpoints/last.pt
```

Use the same dataset and compatible training configuration.

### The model generates poor captions

Caption quality depends on several factors, including:

- Dataset quality
- Number of training examples
- Caption diversity
- Vocabulary threshold
- Training duration
- CNN fine-tuning
- Transformer configuration
- Checkpoint quality
- Beam-search configuration

A lower validation loss does not automatically mean every individual generated caption will be linguistically or semantically perfect.

## 📊 Current Model Limitations

Known limitations include:

- Vocabulary is built from the training captions.
- Unknown words are mapped to `<unk>`.
- The tokenizer is intentionally simple and English-oriented.
- The model is trained for a fixed maximum sequence length.
- CPU inference can be relatively slow.
- Beam search increases inference computation.
- The application does not currently implement authentication.
- The API does not currently enforce an explicit upload-size limit.
- The Flask development server should not be used as the production server.
- Caption quality is dependent on the training dataset and learned checkpoint.
- The current frontend is a simple single-page interface rather than a full user-management system.

## 🔮 Possible Future Improvements

Potential improvements include:

- Add COCO or a larger caption dataset
- Train for more epochs where appropriate
- Improve tokenizer/subword handling
- Use BPE/SentencePiece tokenization
- Add attention visualization
- Add CIDEr, BLEU, METEOR and ROUGE evaluation
- Add caption confidence/log-probability information
- Add multiple caption generation
- Add multilingual captioning
- Add image preprocessing validation
- Add maximum upload size
- Add API authentication
- Add request rate limiting
- Add asynchronous inference
- Add model quantization
- Add ONNX/TorchScript optimization where appropriate
- Add automated tests
- Add CI/CD
- Add production monitoring
- Add a dedicated frontend framework if the UI grows
- Add Docker support
- Add a health-check endpoint such as `/health`
- Add structured logging

## ⭐ Project Summary

This project demonstrates a complete deep-learning image-captioning workflow:

```
Flickr30k Dataset
       │
       ▼
Vocabulary Creation
       │
       ▼
Image Preprocessing
       │
       ▼
ResNet-50 Encoder
       │
       ▼
Visual Embeddings
       │
       ▼
Transformer Decoder
       │
       ▼
Training + Validation
       │
       ▼
best.pt / last.pt
       │
       ▼
Flask API
       │
       ▼
Web Interface
       │
       ▼
Generated Image Caption
```
