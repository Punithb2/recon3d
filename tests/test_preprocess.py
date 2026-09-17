import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from recon3d.framing import bbox_stats
from recon3d.serve.preprocess import InputError, fill_closed_outlines, otsu_threshold, prepare


def png(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


def dark_rect_on_white(w=300, h=200, box=(20, 30, 80, 150)):
    img = Image.new("L", (w, h), 255)
    ImageDraw.Draw(img).rectangle(box, fill=0)
    return img


def test_dark_object_on_white_is_inverted_centred_and_scaled():
    p = prepare(png(dark_rect_on_white()), size=64, fill=0.6)
    s = p.silhouette
    assert s.shape == (64, 64) and s.dtype == np.uint8
    assert p.info["source"] == "threshold" and p.info["inverted"] is True
    st = bbox_stats(s)
    assert st["fill"] == pytest.approx(0.6, abs=0.05)             # longest side fills 60% of the frame
    assert abs(st["cx"]) < 0.05 and abs(st["cy"]) < 0.05           # object moved from the corner to the centre
    assert s[0, 0] == 0 and s[32, 32] == 255                       # white object on black, like training data


def test_light_object_on_black_is_kept():
    img = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(img).ellipse((30, 30, 70, 70), fill=255)
    p = prepare(png(img), size=64, fill=0.5)
    assert p.info["inverted"] is False and p.silhouette[32, 32] == 255


def test_transparent_png_uses_alpha():
    img = Image.new("RGBA", (120, 120), (255, 255, 255, 0))       # white but fully transparent background
    ImageDraw.Draw(img).rectangle((40, 20, 80, 100), fill=(200, 200, 200, 255))
    p = prepare(png(img), size=64, fill=0.6)
    assert p.info["source"] == "alpha" and p.silhouette[32, 32] == 255


def test_opaque_rgba_falls_back_to_threshold():
    img = dark_rect_on_white().convert("RGBA")                     # alpha channel present but all 255
    assert prepare(png(img), size=64, fill=0.6).info["source"] == "threshold"


def test_jpeg_and_webp_accepted():
    for fmt in ("JPEG", "WEBP"):
        assert prepare(png(dark_rect_on_white().convert("RGB"), fmt), 64, 0.6).silhouette.max() == 255


def test_edges_are_anti_aliased_like_training_images():
    s = prepare(png(dark_rect_on_white(box=(21, 33, 87, 151))), size=64, fill=0.6).silhouette
    assert ((s > 0) & (s < 255)).any()


@pytest.mark.parametrize("data,status", [
    (b"", 400),
    (b"not an image at all", 415),
])
def test_bad_bytes(data, status):
    with pytest.raises(InputError) as e:
        prepare(data, 64, 0.6)
    assert e.value.status == status


def test_gif_is_rejected():
    with pytest.raises(InputError) as e:
        prepare(png(dark_rect_on_white().convert("P"), "GIF"), 64, 0.6)
    assert e.value.status == 415


def test_size_limits():
    data = png(dark_rect_on_white())
    with pytest.raises(InputError) as e:
        prepare(data, 64, 0.6, max_bytes=100)
    assert e.value.status == 413
    with pytest.raises(InputError) as e:
        prepare(data, 64, 0.6, max_pixels=1000)
    assert e.value.status == 413
    with pytest.raises(InputError) as e:
        prepare(png(Image.new("L", (10, 10))), 64, 0.6)
    assert e.value.status == 422
    with pytest.raises(InputError) as e:                           # already-decoded images are checked too
        prepare(dark_rect_on_white(), 64, 0.6, max_pixels=1000)
    assert e.value.status == 413


def test_blank_and_full_images_are_rejected():
    for img in (Image.new("L", (100, 100), 255), Image.new("L", (100, 100), 0)):
        with pytest.raises(InputError, match="no visible object"):
            prepare(png(img), 64, 0.6)
    full = Image.new("RGBA", (100, 100), (0, 0, 0, 255))           # opaque object covering 98% ...
    ImageDraw.Draw(full).rectangle((0, 98, 99, 99), fill=(0, 0, 0, 0))   # ... with a thin transparent strip
    with pytest.raises(InputError, match="fills almost"):
        prepare(png(full), 64, 0.6)
    speck = Image.new("L", (400, 400), 255)
    speck.putpixel((10, 10), 0)
    with pytest.raises(InputError, match="no object found"):
        prepare(png(speck), 64, 0.6)


def test_closed_outline_is_filled_open_one_is_not():
    img = Image.new("L", (100, 100), 255)
    ImageDraw.Draw(img).rectangle((20, 20, 80, 80), outline=0, width=3)
    plain = prepare(png(img), 64, 0.6)
    filled = prepare(png(img), 64, 0.6, fill_outlines=True)
    assert plain.silhouette[32, 32] == 0 and filled.silhouette[32, 32] == 255
    assert filled.info["object_share"] > 3 * plain.info["object_share"]

    m = np.zeros((20, 20), bool)
    m[5:15, 5] = m[5, 5:15] = m[14, 5:15] = True                   # a "C": not closed
    assert fill_closed_outlines(m).sum() == m.sum()


def test_otsu_splits_two_levels():
    g = np.array([10] * 50 + [200] * 50, np.uint8)
    assert 10 <= otsu_threshold(g) < 200
