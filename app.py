import os
import io
import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image

from generate_caption import load_model, caption_image

app = Flask(__name__)

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/best.pt")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "checkpoints/vocab.pkl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vocab = None, None


def get_model():
    
    global model, vocab
    if model is None:
        if not (os.path.exists(CHECKPOINT_PATH) and os.path.exists(VOCAB_PATH)):
            raise FileNotFoundError(
                f"No trained checkpoint found at '{CHECKPOINT_PATH}'. "
                "Run train.py first, or point CHECKPOINT_PATH/VOCAB_PATH at an existing model."
            )
        model, vocab = load_model(CHECKPOINT_PATH, VOCAB_PATH, DEVICE)
    return model, vocab


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/caption", methods=["POST"])
def api_caption():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read image file"}), 400

    try:
        m, v = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    from data.dataset import eval_transform
    image_tensor = eval_transform(image).unsqueeze(0)
    caption = m.beam_search(image_tensor, v, beam_width=5, max_len=30, device=DEVICE)

    return jsonify({"caption": caption})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
