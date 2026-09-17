"""Turn an arbitrary uploaded image into a silhouette framed like the training data.

Training silhouettes: white object on black, anti-aliased edges, object roughly centred and filling a
typical share of the frame. Uploads can be anything: a black shape on white paper, a transparent PNG,
a drawing, a photo on a plain background. Steps:

  1. decode safely (size limits, formats)          -> InputError on anything suspicious
  2. build a mask: alpha channel if the image has transparency, otherwise grey levels with
     Otsu's threshold, inverted when the border is bright (dark object on light background)
  3. optionally fill closed outlines (for drawings)
  4. crop to the object, centre it, scale it to `fill` of the frame, resize with anti-aliasing
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP"}


class InputError(ValueError):
    """The upload can't be used. `status` is the HTTP status the API returns."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


@dataclass
class Prepared:
    silhouette: np.ndarray                   # uint8 [size, size], what the model sees
    info: dict = field(default_factory=dict)


def decode_image(data: bytes, max_bytes: int, max_pixels: int) -> Image.Image:
    if not data:
        raise InputError("empty upload", 400)
    if len(data) > max_bytes:
        raise InputError(f"file is larger than {max_bytes / 1e6:.1f} MB", 413)
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            w, h = probe.size
            probe.verify()                                    # structural check without decoding pixels
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise InputError("not a readable image", 415) from e
    if fmt not in ALLOWED_FORMATS:
        raise InputError(f"unsupported image format {fmt}; use PNG, JPEG, WEBP or BMP", 415)
    if w * h > max_pixels:
        raise InputError(f"image is {w}x{h}; at most {max_pixels:,} pixels allowed", 413)
    if min(w, h) < 16:
        raise InputError("image is too small (under 16 px)", 422)
    img = Image.open(io.BytesIO(data))
    img.load()
    return ImageOps.exif_transpose(img)                       # respect phone camera rotation


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    sigma_b = np.nan_to_num(sigma_b)
    return int(np.argmax(sigma_b))


def object_mask(img: Image.Image) -> tuple[np.ndarray, dict]:
    info = {}
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        alpha = np.asarray(img.convert("RGBA").getchannel("A"))
        if alpha.min() < 250:                                 # real transparency, not just an opaque alpha channel
            info["source"] = "alpha"
            return alpha > 127, info
    gray = np.asarray(img.convert("L"))
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    inverted = bool(np.median(border) > 127)
    if inverted:
        gray = 255 - gray
    t = otsu_threshold(gray)
    if gray.max() - gray.min() < 30:                          # flat image: Otsu would split noise
        raise InputError("the image has no visible object (almost uniform)", 422)
    info.update(source="threshold", inverted=inverted, threshold=t)
    return gray > t, info


def fill_closed_outlines(mask: np.ndarray) -> np.ndarray:
    """Pixels not reachable from the border become object (turns a drawn outline into a solid shape)."""
    # .copy(): an image created from a NumPy array shares its memory, and flood fill silently skips writes then
    padded = Image.fromarray(np.pad(mask, 1).astype(np.uint8) * 255).copy()
    ImageDraw.floodfill(padded, (0, 0), 128)
    return (np.asarray(padded) != 128)[1:-1, 1:-1]


def frame(mask: np.ndarray, size: int, fill: float) -> tuple[np.ndarray, dict]:
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = mask[y0:y1, x0:x1]
    side = int(np.ceil(max(crop.shape) / fill))
    canvas = np.zeros((side, side), np.uint8)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop * 255
    out = Image.fromarray(canvas).resize((size, size), Image.LANCZOS)     # area-like downsampling = soft edges
    return np.asarray(out, dtype=np.uint8), {"bbox": [int(x0), int(y0), int(x1), int(y1)]}


def prepare(data_or_image, size: int, fill: float, *, fill_outlines: bool = False,
            max_bytes: int = 5_000_000, max_pixels: int = 4096 * 4096) -> Prepared:
    if isinstance(data_or_image, Image.Image):               # already decoded (Gradio UI)
        img = data_or_image
        if img.width * img.height > max_pixels:
            raise InputError(f"image is {img.width}x{img.height}; at most {max_pixels:,} pixels allowed", 413)
    else:
        img = decode_image(data_or_image, max_bytes, max_pixels)
    mask, info = object_mask(img)
    if fill_outlines:
        mask = fill_closed_outlines(mask)
    share = float(mask.mean())
    info.update(width=img.width, height=img.height, object_share=round(share, 4), filled_outlines=fill_outlines)
    if share < 0.002:
        raise InputError("no object found: draw or upload a single dark shape on a light background "
                         "(or a light shape on dark)", 422)
    if share > 0.9:
        raise InputError("the object fills almost the whole image; leave some background around it", 422)
    sil, finfo = frame(mask, size, fill)
    info.update(finfo)
    return Prepared(sil, info)
