# =========================================================
# conure_features.py  -- ฟังก์ชันสกัดฟีเจอร์ที่ใช้ร่วมกัน
# (ทั้งสคริปต์เปรียบเทียบโมเดล และแอป Streamlit ต้องใช้ตัวเดียวกันเท่านั้น
#  ไม่งั้นผลทำนายจะเพี้ยน)
# =========================================================
import hashlib
import re

import numpy as np
from PIL import Image, ImageOps

BACKBONE_SIZE = {
    "MobileNetV2": 224,
    "EfficientNetV2B0": 224,
    "ConvNeXtTiny": 224,
    "ResNet50V2": 224,
}

_SRC_PATTERN = re.compile(r"_(?:jpg|jpeg|png)\.rf\.", re.IGNORECASE)


def source_id(path):
    """ชื่อแบบ Roboflow  Normal73_jpg.rf.<hash>.jpg -> 'Normal73' (ภาพต้นฉบับเดียวกัน)"""
    parts = _SRC_PATTERN.split(path.name, maxsplit=1)
    return parts[0] if len(parts) == 2 else path.stem


def paths_hash(paths):
    return hashlib.md5("\n".join(map(str, paths)).encode()).hexdigest()


# ---------------------------------------------------------
# โหลดภาพ
# ---------------------------------------------------------
def prepare(img, max_side=768):
    """แก้การหมุนจาก EXIF, แปลงเป็น RGB, ย่อภาพใหญ่ (กัน RAM เต็ม)"""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def load_rgb(path):
    with Image.open(path) as im:
        im.load()
        return prepare(im)


# ---------------------------------------------------------
# ฟีเจอร์จาก CNN (pretrained, ไม่ fine-tune)
# ---------------------------------------------------------
def load_backbone(name, weights="imagenet"):
    from tensorflow import keras  # import ตอนใช้งานเท่านั้น

    A = keras.applications
    table = {
        "MobileNetV2": (A.MobileNetV2, A.mobilenet_v2.preprocess_input),
        "EfficientNetV2B0": (A.EfficientNetV2B0, A.efficientnet_v2.preprocess_input),
        "ConvNeXtTiny": (A.ConvNeXtTiny, A.convnext.preprocess_input),
        "ResNet50V2": (A.ResNet50V2, A.resnet_v2.preprocess_input),
    }
    ctor, prep = table[name]
    size = BACKBONE_SIZE[name]
    model = ctor(include_top=False, weights=weights, pooling="avg",
                 input_shape=(size, size, 3))
    return {"name": name, "model": model, "prep": prep, "size": size}


def cnn_features(imgs, bb, batch=32):
    """imgs: list ของ PIL.Image (RGB) -> array (N, D)"""
    size = bb["size"]
    out = []
    for i in range(0, len(imgs), batch):
        arr = np.stack([np.asarray(im.resize((size, size)), dtype=np.float32)
                        for im in imgs[i:i + batch]])
        out.append(bb["model"].predict(bb["prep"](arr), verbose=0))
    return np.concatenate(out)


def cnn_features_from_paths(paths, bb, chunk=64):
    out = []
    for i in range(0, len(paths), chunk):
        imgs = [load_rgb(p) for p in paths[i:i + chunk]]
        out.append(cnn_features(imgs, bb))
        print(f"    {bb['name']}: {min(i + chunk, len(paths))}/{len(paths)}", end="\r")
    print()
    return np.concatenate(out)


# ---------------------------------------------------------
# ฟีเจอร์สีโดยตรง (HSV histogram ของบริเวณกลางภาพ)
# ---------------------------------------------------------
def color_features(img, crop=0.6, h_bins=18, s_bins=3, v_bins=3):
    w, h = img.size
    cw, ch = int(w * crop), int(h * crop)
    left, top = (w - cw) // 2, (h - ch) // 2
    c = img.crop((left, top, left + cw, top + ch)).resize((96, 96))
    hsv = np.asarray(c.convert("HSV"), dtype=np.float32) / 255.0
    pts = hsv.reshape(-1, 3)
    hist, _ = np.histogramdd(pts, bins=(h_bins, s_bins, v_bins), range=((0, 1),) * 3)
    hist = hist.ravel() / max(hist.sum(), 1.0)
    rgb = np.asarray(c, dtype=np.float32) / 255.0
    stats = np.concatenate([rgb.mean((0, 1)), rgb.std((0, 1))])
    return np.concatenate([np.sqrt(hist), stats]).astype(np.float32)


def color_features_batch(imgs):
    return np.stack([color_features(im) for im in imgs])


def color_features_from_paths(paths):
    return np.stack([color_features(load_rgb(p)) for p in paths])


def featurize(imgs, bb, use_color):
    """ใช้ในแอป: PIL list -> เวกเตอร์ฟีเจอร์ (รูปแบบเดียวกับตอนเทรน)"""
    X = cnn_features(imgs, bb)
    if use_color:
        X = np.concatenate([X, color_features_batch(imgs)], axis=1)
    return X


def featurize_with_cnn(imgs, bb, use_color):
    """เหมือน featurize แต่คืน (ฟีเจอร์ CNN ล้วน, ฟีเจอร์ที่ส่งเข้า classifier)"""
    cnn = cnn_features(imgs, bb)
    X = np.concatenate([cnn, color_features_batch(imgs)], axis=1) if use_color else cnn
    return cnn, X


# =========================================================
# ตัวกรอง "ไม่ใช่นกแก้วกรีนชีค" (open-set rejection)
#   Gate 1: ImageNet classifier -> คะแนนรวมของคลาส "นก" (ตัดสัตว์อื่น/สิ่งของ)
#   Gate 2: ความคล้ายกับภาพกรีนชีคในชุดเทรน (cosine kNN บน embedding)
# =========================================================
# ดัชนีคลาสนกใน ImageNet-1k: 7-24, 80-100, 127-146
BIRD_IDX = list(range(7, 25)) + list(range(80, 101)) + list(range(127, 147))


def load_imagenet_gate():
    from tensorflow import keras

    A = keras.applications
    model = A.MobileNetV2(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    return {"model": model, "prep": A.mobilenet_v2.preprocess_input}


def bird_scores(imgs, gate, batch=32):
    """imgs: list ของ PIL -> array (N,) ความน่าจะเป็นรวมของคลาสนกใน ImageNet"""
    out = []
    for i in range(0, len(imgs), batch):
        arr = np.stack([np.asarray(im.resize((224, 224)), dtype=np.float32)
                        for im in imgs[i:i + batch]])
        p = gate["model"].predict(gate["prep"](arr), verbose=0)
        out.append(p[:, BIRD_IDX].sum(axis=1))
    return np.concatenate(out)


def bird_scores_from_paths(paths, gate, chunk=64):
    out = []
    for i in range(0, len(paths), chunk):
        out.append(bird_scores([load_rgb(p) for p in paths[i:i + chunk]], gate))
        print(f"    bird-gate: {min(i + chunk, len(paths))}/{len(paths)}", end="\r")
    print()
    return np.concatenate(out)


def l2norm(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def knn_similarity(emb, ref, k=5):
    """emb (n,d), ref (m,d) ต้อง L2-normalize แล้ว -> ค่าเฉลี่ย cosine ของ k เพื่อนบ้านที่ใกล้สุด"""
    S = emb @ ref.T
    k = min(k, S.shape[1])
    return np.sort(S, axis=1)[:, -k:].mean(axis=1)
