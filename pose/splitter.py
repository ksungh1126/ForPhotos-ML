# 컷분할을 위한 파일입니다.
import cv2
import numpy as np
from PIL import Image, ImageOps

def _load_oriented(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def _estimate_border_color_lab(bgr, border_frac=0.03):
    H, W = bgr.shape[:2]
    t = max(1, int(min(H, W) * border_frac))
    mask = np.zeros((H, W), np.uint8)
    mask[:t,:] = 1; mask[-t:,:] = 1; mask[:, :t] = 1; mask[:, -t:] = 1
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    border = lab[mask == 1].reshape(-1, 3).astype(np.float32)
    mean_lab = border.mean(axis=0)
    return mean_lab, lab

def _delta_e_lab(lab_img, ref_lab):
    diff = lab_img.astype(np.float32) - ref_lab.reshape(1,1,3)
    dist = np.sqrt((diff**2).sum(axis=2))
    return dist

def _binarize_by_delta(dist, method="otsu", quantile=0.35):
    norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if method == "otsu":
        thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (norm >= thr).astype(np.uint8) * 255
    else:
        q = np.quantile(norm, quantile)
        mask = (norm >= q).astype(np.uint8) * 255
    return mask

def _postprocess_mask(mask, k_close=11, k_open=5):
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko, iterations=1)
    return m

def _find_panels_from_mask(mask, min_area_ratio=0.06, topk=4):
    H, W = mask.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= min_area_ratio * H * W:
            pad = int(0.01 * max(H, W))
            x0 = max(0, x - pad); y0 = max(0, y - pad)
            x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
            boxes.append((y0, y1, x0, x1, area))
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)[:topk]
    return [(y0, y1, x0, x1) for (y0, y1, x0, x1, _) in boxes]

def _sort_boxes_grid(boxes):
    centers = [((y0+y1)/2, (x0+x1)/2) for (y0,y1,x0,x1) in boxes]
    ys = sorted([c[0] for c in centers])
    split_y = (ys[1] + ys[2]) / 2
    top = sorted([(b,c) for b,c in zip(boxes, centers) if c[0] <= split_y], key=lambda bc: bc[1][1])
    bot = sorted([(b,c) for b,c in zip(boxes, centers) if c[0] >  split_y], key=lambda bc: bc[1][1])
    return [top[0][0], top[1][0], bot[0][0], bot[1][0]]

def _sort_boxes_vertical(boxes):
    return sorted(boxes, key=lambda b: (b[0]+b[1])/2)

def _sort_boxes_horizontal(boxes):
    return sorted(boxes, key=lambda b: (b[2]+b[3])/2)

def _uniform_boxes(H, W, rows, cols):
    ys = [int(i*H/rows) for i in range(rows+1)]
    xs = [int(j*W/cols) for j in range(cols+1)]
    return [(ys[r], ys[r+1], xs[c], xs[c+1]) for r in range(rows) for c in range(cols)]

def split_photobooth_strip(path, force_layout=None):
    img = _load_oriented(path)
    H, W = img.shape[:2]

    bg_lab, lab = _estimate_border_color_lab(img, border_frac=0.03)
    dist = _delta_e_lab(lab, bg_lab)
    mask = _binarize_by_delta(dist, method="otsu")
    mask = _postprocess_mask(mask, k_close=11, k_open=5)

    boxes = _find_panels_from_mask(mask, min_area_ratio=0.06, topk=4)

    if len(boxes) == 4:
        ar = H / max(1, W)
        layout = force_layout or ('1x4' if ar > 1.5 else '4x1' if ar < 0.7 else '2x2')
        if layout == '1x4':
            boxes = _sort_boxes_vertical(boxes)
        elif layout == '4x1':
            boxes = _sort_boxes_horizontal(boxes)
        else:
            boxes = _sort_boxes_grid(boxes)
    else:
        ar = H / max(1, W)
        layout = force_layout or ('1x4' if ar > 1.5 else '4x1' if ar < 0.7 else '2x2')
        if layout == '1x4':
            boxes = _uniform_boxes(H, W, 4, 1)
        elif layout == '4x1':
            boxes = _uniform_boxes(H, W, 1, 4)
        else:
            boxes = _uniform_boxes(H, W, 2, 2)

    crops = []
    for (y0,y1,x0,x1) in boxes:
        y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))
        x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))
        crop = img[y0:y1, x0:x1].copy()
        crops.append((crop, (y0,y1,x0,x1)))
    return crops