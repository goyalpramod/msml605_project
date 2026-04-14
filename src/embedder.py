"""FaceNet embedding module for face verification.

Uses InceptionResnetV1 (facenet-pytorch, pretrained on VGGFace2) to extract
512-dimensional L2-normalized face embeddings.

Preprocessing assumptions:
- Input images are face crops (no detection is run here).
- Resized to 160x160 pixels — FaceNet's expected input size.
- Pixel values normalized to [-1, 1]: (pixel - 127.5) / 128.0
- Deterministic: model is always in eval() mode; no dropout or augmentation.

Embedding properties:
- Dimensionality: 512
- Normalization: L2 unit sphere (cosine similarity == dot product for normalized vectors)
- Model: InceptionResnetV1, pretrained='vggface2' via facenet-pytorch
- Weights are downloaded once to ~/.cache/torch/hub/ and reused on subsequent calls.
"""

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image

__all__ = ["load_model", "preprocess_image", "get_embedding"]

_TARGET_SIZE = 160


def load_model() -> InceptionResnetV1:
    """Load the pretrained FaceNet model (InceptionResnetV1, VGGFace2).

    Always returns the model in eval() mode on CPU. Weights are downloaded
    from the facenet-pytorch cache on first call and reused thereafter.

    Returns:
        InceptionResnetV1 in eval mode, gradients disabled externally via
        torch.no_grad() at call sites.
    """
    model = InceptionResnetV1(pretrained="vggface2").eval()
    return model


def preprocess_image(img: np.ndarray) -> torch.Tensor:
    """Preprocess a face image into a FaceNet-ready tensor.

    Steps:
      1. Convert to uint8 if needed (clips values to [0, 255]).
      2. Convert to RGB PIL image (handles both grayscale and RGB input).
      3. Resize to 160x160 using bilinear interpolation (deterministic).
      4. Cast to float32 tensor of shape (1, 3, 160, 160).
      5. Normalize: (pixel - 127.5) / 128.0  → values in [-1, 1].

    Args:
        img: numpy array of shape (H, W) grayscale or (H, W, 3) RGB.
             Any numeric dtype; values assumed to be in [0, 255] range.

    Returns:
        torch.Tensor of shape (1, 3, 160, 160) with values in [-1, 1].
    """
    if img.dtype != np.uint8:
        # Images stored in [0, 1] float range must be scaled to [0, 255] first.
        # Images already in [0, 255] float range are clipped directly.
        if img.max() <= 1.0 and img.min() >= 0.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    pil_img = pil_img.resize((_TARGET_SIZE, _TARGET_SIZE), Image.BILINEAR)

    # (H, W, C) → float32 → (C, H, W) → (1, C, H, W)
    arr = np.array(pil_img, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    # Standard FaceNet normalization: [0, 255] → [-1, 1]
    tensor = (tensor - 127.5) / 128.0
    return tensor


def get_embedding(img: np.ndarray, model: InceptionResnetV1) -> np.ndarray:
    """Extract a 512-dimensional L2-normalized FaceNet embedding.

    Runs preprocess_image → model forward pass → explicit L2 normalization.
    The model is assumed to be in eval() mode (see load_model()).

    Args:
        img: numpy array of shape (H, W) or (H, W, 3). See preprocess_image.
        model: InceptionResnetV1 returned by load_model().

    Returns:
        np.ndarray of shape (512,), L2-normalized (unit vector).
        Cosine similarity between two such vectors equals their dot product.
    """
    tensor = preprocess_image(img)
    with torch.no_grad():
        embedding = model(tensor)  # shape: (1, 512)

    emb = embedding.squeeze(0).cpu().numpy()  # shape: (512,)

    # Explicit L2 normalization — defensive in case model weights change.
    norm = float(np.linalg.norm(emb))
    if norm > 0:
        emb = emb / norm
    return emb
