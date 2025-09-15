import os
import cv2
import numpy as np
import tempfile
import pandas as pd
import mediapipe as mp

from deepface import DeepFace
from splitter import split_photobooth_strip
from analysis import get_pose_type_from_array, get_num_people   # YOLO 경로 입력용
from metadata import create_dataframe

# ===== Debug/heuristics switches =====
DEBUG_GENDER = True         # set False after verification
FACE_MIN_CONF = 0.25        # MediaPipe detection threshold (0.35~0.6)
FACE_PAD = 0.45             # padding ratio around detected face box
WHOLE_BACKENDS = ('mtcnn', 'retinaface', 'opencv')  # try order for whole-crop DeepFace
MAX_SIDE_GENDER = 1024       # upscale small crops to help detectors
TRY_INSIGHTFACE = True      # if installed, use it for gender per-face

# Cached InsightFace app (lazy init)
INSIGHT_APP = None
def _get_insight_app():
    global INSIGHT_APP
    if not TRY_INSIGHTFACE:
        return None
    if INSIGHT_APP is not None:
        return INSIGHT_APP
    try:
        from insightface.app import FaceAnalysis  # type: ignore
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0, det_size=(1024, 1024))
        INSIGHT_APP = app
        if DEBUG_GENDER:
            print("[DBG] InsightFace app initialized")
        return INSIGHT_APP
    except Exception as e:
        if DEBUG_GENDER:
            print(f"[DBG] InsightFace init failed: {e}")
        return None

# ---------- Embedding clustering helpers ----------
def _l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)

def _extract_faces_with_embeddings(crop_bgr, frame_idx, photo_id):
    """
    Return list of dicts: [{'photo_id', 'frame_id', 'embedding', 'gender_scalar', 'bbox'}]
    Uses InsightFace only (fallback: empty list).
    """
    app = _get_insight_app()
    results = []
    if app is None:
        return results
    try:
        rgb = cv2.cvtColor(resize_for_deepface(crop_bgr, MAX_SIDE_GENDER), cv2.COLOR_BGR2RGB)
        faces = app.get(rgb) or []
        for f in faces:
            emb = getattr(f, 'embedding', None)
            gen = getattr(f, 'gender', None)
            bbox = getattr(f, 'bbox', None)
            if emb is None or gen is None:
                continue
            try:
                emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)[0]
            except Exception:
                continue
            results.append({
                'photo_id': photo_id,
                'frame_id': frame_idx,
                'embedding': emb,
                'gender_scalar': float(gen),  # ~0: female, ~1: male
                'bbox': bbox
            })
    except Exception as e:
        if DEBUG_GENDER:
            print(f"[DBG] embedding extract failed f{frame_idx}: {e}")
    return results

def _cluster_and_vote(faces, k=2):
    """
    faces: list of dicts from _extract_faces_with_embeddings
    Returns:
      - per_face_labels: list of cluster ids aligned with faces
      - cluster_gender: dict {cluster_id: 'male'/'female'}
    Fallbacks to simple threshold if sklearn unavailable or too few faces.
    """
    if len(faces) < 2:
        return None, None
    X = np.stack([f['embedding'] for f in faces], axis=0)
    X = _l2_normalize(X)

    # Try sklearn KMeans
    labels = None
    try:
        from sklearn.cluster import KMeans  # type: ignore
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
    except Exception as e:
        if DEBUG_GENDER:
            print(f"[DBG] sklearn KMeans unavailable: {e}. Falling back to cosine-split.")
        # simple fallback: split by cosine to first vector (2 clusters)
        c0 = X[0]
        sims = X @ c0
        thresh = float(np.median(sims))
        labels = (sims < thresh).astype(int)

    # vote genders per cluster using mean of gender_scalar
    cluster_gender = {}
    for cid in range(k):
        inds = [i for i, lb in enumerate(labels) if lb == cid]
        if not inds:
            cluster_gender[cid] = 'male'  # default, won't matter
            continue
        mean_g = float(np.mean([faces[i]['gender_scalar'] for i in inds]))
        cluster_gender[cid] = 'male' if mean_g >= 0.5 else 'female'
    return labels, cluster_gender

def _dbg_print_gender_result(tag, analysis):
    if not DEBUG_GENDER:
        return
    try:
        import json
        if isinstance(analysis, (list, tuple)):
            sample = analysis[0] if analysis else {}
        else:
            sample = analysis
        # keep only relevant tiny subset for readability
        subset = {}
        if isinstance(sample, dict):
            for k in ('dominant_gender', 'gender', 'region', 'face_confidence'):
                if k in sample:
                    subset[k] = sample[k]
        print(f"[DBG][{tag}] sample={json.dumps(subset, ensure_ascii=False)}")
    except Exception as e:
        print(f"[DBG][{tag}] print failed: {e}")

CSV_PATH = 'metadata.csv'
IMAGE_FOLDER = 'photos'
FORCE_LAYOUT = None  # '1x4' | '4x1' | '2x2' | None(자동추정)

def safe_load_or_create_df(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df = pd.read_csv(csv_path)
        print("[INFO] 기존 metadata.csv 로드")
    else:
        df = create_dataframe()
        if 'frame_id' not in df.columns:
            df['frame_id'] = []
        print("[INFO] 새 데이터프레임 생성")
    return df

def resize_for_deepface(bgr, max_side=MAX_SIDE_GENDER):
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale > 1.0:
        # upscale small faces to help detection
        return cv2.resize(bgr, (int(w * scale), int(h * scale)))
    elif scale < 1.0:
        return cv2.resize(bgr, (int(w * scale), int(h * scale)))
    return bgr

def infer_gender_counts_with_deepface(crop_bgr, backend='retinaface'):
    """
    Gender counting (InsightFace-first):
    1) Try InsightFace per-face (MediaPipe boxes, padded & upscaled) for robust gender.
    2) If still undecided, try InsightFace directly on the whole crop (it has its own detector).
    3) Fallback: DeepFace on the whole crop with backend cascade.
    Returns: (male_count, female_count)
    """
    def _count_if_gender_attr(faces):
        male = female = 0
        for f in faces or []:
            g = getattr(f, 'gender', None)
            if g is None:
                continue
            try:
                if int(round(float(g))) == 1:
                    male += 1
                else:
                    female += 1
            except Exception:
                pass
        return male, female

    # ---------- A) InsightFace per-face (using MediaPipe boxes) ----------
    app = _get_insight_app()
    try:
        h, w = crop_bgr.shape[:2]
        with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=FACE_MIN_CONF) as fd:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            res = fd.process(rgb)
            if app is not None and res.detections:
                male = female = 0
                for idx, det in enumerate(res.detections):
                    bbox = det.location_data.relative_bounding_box
                    x0 = max(0, int(bbox.xmin * w)); y0 = max(0, int(bbox.ymin * h))
                    x1 = min(w, int((bbox.xmin + bbox.width) * w)); y1 = min(h, int((bbox.ymin + bbox.height) * h))
                    if x1 - x0 <= 0 or y1 - y0 <= 0:
                        continue
                    # padding
                    cx = (x0 + x1) / 2.0; cy = (y0 + y1) / 2.0
                    bw = (x1 - x0);       bh = (y1 - y0)
                    nx0 = int(max(0, cx - bw * (1 + 2 * FACE_PAD) / 2))
                    nx1 = int(min(w, cx + bw * (1 + 2 * FACE_PAD) / 2))
                    ny0 = int(max(0, cy - bh * (1 + 2 * FACE_PAD) / 2))
                    ny1 = int(min(h, cy + bh * (1 + 2 * FACE_PAD) / 2))
                    face_crop = crop_bgr[ny0:ny1, nx0:nx1]
                    if face_crop.size == 0:
                        continue
                    # ensure decent size
                    fh, fw = face_crop.shape[:2]
                    long_side = max(fh, fw)
                    if long_side < 256:
                        scale = 256.0 / max(1, long_side)
                        face_crop = cv2.resize(face_crop, (int(fw*scale), int(fh*scale)))
                    # InsightFace on face crop (RGB)
                    try:
                        frgb = cv2.cvtColor(resize_for_deepface(face_crop, 320), cv2.COLOR_BGR2RGB)
                        faces = app.get(frgb)
                        # pick the largest face within the crop
                        if faces:
                            f = max(faces, key=lambda F: (F.bbox[2]-F.bbox[0])*(F.bbox[3]-F.bbox[1]))
                            m, fe = _count_if_gender_attr([f])
                            male += m; female += fe
                            if DEBUG_GENDER:
                                print(f"[DBG][IF] face#{idx} gender={(1 if m else 0)}")
                    except Exception as e:
                        if DEBUG_GENDER:
                            print(f"[DBG][IF] per-face failed: {e}")
                perface_m, perface_f = male, female
    except Exception as e:
        if DEBUG_GENDER:
            print(f"[DBG] FaceDetection path failed: {e}")

    # Keep best result across strategies
    perface_m = locals().get('perface_m', 0)
    perface_f = locals().get('perface_f', 0)

    # ---------- B) InsightFace on whole crop (its own detector) ----------
    try:
        if app is not None:
            frgb = cv2.cvtColor(resize_for_deepface(crop_bgr, MAX_SIDE_GENDER), cv2.COLOR_BGR2RGB)
            faces = app.get(frgb)
            whole_m, whole_f = _count_if_gender_attr(faces)
            if DEBUG_GENDER:
                print(f"[DBG][IF] whole-crop counted: M={whole_m}, F={whole_f}")
            if (whole_m + whole_f) >= (perface_m + perface_f):
                best_m, best_f = whole_m, whole_f
            else:
                best_m, best_f = perface_m, perface_f
            if (best_m + best_f) > 0:
                return best_m, best_f
    except Exception as e:
        if DEBUG_GENDER:
            print(f"[DBG][IF] whole-crop failed: {e}")

    # If whole-crop failed but per-face had some counts, use per-face
    if (perface_m + perface_f) > 0:
        return perface_m, perface_f

    # ---------- C) DeepFace fallback on whole crop ----------
    def _parse_deepface_result(analysis):
        male = female = 0
        itr = analysis if isinstance(analysis, list) else [analysis]
        for face in itr:
            if not isinstance(face, dict):
                continue
            dom = face.get('dominant_gender')
            if dom is None:
                gdict = face.get('gender', {})
                if isinstance(gdict, dict) and gdict:
                    dom = max(gdict, key=gdict.get)
            if isinstance(dom, str):
                d = dom.strip().lower()
                if d in ('man', 'male'): male += 1
                elif d in ('woman', 'female'): female += 1
        return male, female

    def _deepface_on_img(img_bgr, backend_name, tag):
        img_small = resize_for_deepface(img_bgr, max_side=MAX_SIDE_GENDER)
        try:
            analysis = DeepFace.analyze(
                img_path=img_small,
                actions=['gender'],
                enforce_detection=False,
                detector_backend=backend_name,
            )
            _dbg_print_gender_result(tag, analysis)
            return _parse_deepface_result(analysis)
        except Exception as e:
            if DEBUG_GENDER:
                print(f"[DBG][{tag}] DeepFace failed: {e}")
            if backend_name != 'opencv':
                try:
                    analysis = DeepFace.analyze(
                        img_path=img_small,
                        actions=['gender'],
                        enforce_detection=False,
                        detector_backend='opencv',
                    )
                    _dbg_print_gender_result(f"{tag}->opencv", analysis)
                    return _parse_deepface_result(analysis)
                except Exception as e2:
                    if DEBUG_GENDER:
                        print(f"[DBG][{tag}] opencv fallback failed: {e2}")
            return 0, 0

    for bk in WHOLE_BACKENDS:
        m, f = _deepface_on_img(crop_bgr, bk, tag=f'whole:{bk}')
        if (m + f) > 0:
            return m, f

    return 0, 0

def yolo_people_fallback_count(crop_bgr):
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    try:
        cv2.imwrite(tmp.name, crop_bgr)
        count = get_num_people(tmp.name)
    finally:
        try: os.unlink(tmp.name)
        except Exception: pass
    return count

def main():
    # Prepare append mode for CSV
    csv_exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
    if csv_exists:
        # read only photo_id column to compute next id
        try:
            existing = pd.read_csv(CSV_PATH, usecols=['photo_id'])
            next_photo_id = int(existing['photo_id'].max()) + 1 if len(existing) > 0 else 1
        except Exception:
            # fallback if schema changed
            existing = pd.read_csv(CSV_PATH)
            next_photo_id = int(existing['photo_id'].max()) + 1 if 'photo_id' in existing.columns and len(existing) > 0 else 1
    else:
        next_photo_id = 1

    # For append logic: if file does not exist or empty, we will write header once
    header_written = csv_exists

    strips = sorted([f for f in os.listdir(IMAGE_FOLDER)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not strips:
        print(f"[WARN] '{IMAGE_FOLDER}' 폴더에 이미지가 없습니다."); return


    for strip_name in strips:
        strip_path = os.path.join(IMAGE_FOLDER, strip_name)
        crops = split_photobooth_strip(strip_path, force_layout=FORCE_LAYOUT)
        print(f"[INFO] {strip_name} → {len(crops)} cuts")

        photo_id = next_photo_id; next_photo_id += 1

        # --- Strip-level embedding clustering & voting (InsightFace) ---
        # 1) extract embeddings from all crops in this strip
        all_faces = []
        for idx_for_embed, (crop_for_embed, _) in enumerate(crops, start=1):
            all_faces.extend(_extract_faces_with_embeddings(crop_for_embed, idx_for_embed, photo_id))

        # 2) cluster into two identities (typical photobooth case)
        labels = None; cluster_gender = None
        if len(all_faces) >= 2:
            labels, cluster_gender = _cluster_and_vote(all_faces, k=2)
            if DEBUG_GENDER and labels is not None:
                print(f"[DBG][CLUSTER] faces={len(all_faces)}, clusters={len(set(labels))}, cluster_gender={cluster_gender}")
        # 3) build a quick lookup: for each frame, male/female counts from clustering
        cluster_counts_by_frame = {}
        if labels is not None and cluster_gender is not None:
            for face, lb in zip(all_faces, labels):
                frame_id = face['frame_id']
                g = cluster_gender.get(lb, 'male')
                if frame_id not in cluster_counts_by_frame:
                    cluster_counts_by_frame[frame_id] = {'male': 0, 'female': 0}
                cluster_counts_by_frame[frame_id][g] += 1

        for frame_idx, (crop, _) in enumerate(crops, start=1):
            # Prefer cluster-voted counts if available for this frame
            if 'cluster_counts_by_frame' in locals() and frame_idx in cluster_counts_by_frame:
                counts = cluster_counts_by_frame[frame_idx]
                male, female = counts.get('male', 0), counts.get('female', 0)
                if DEBUG_GENDER:
                    print(f"[DBG][CLUSTER] frame#{frame_idx} vote M={male} F={female}")
            else:
                try:
                    male, female = infer_gender_counts_with_deepface(crop, backend='retinaface')
                except Exception as e:
                    print(f"[WARN] DeepFace analyze failed: {e}"); male, female = 0, 0

            try:
                yolo_count = yolo_people_fallback_count(crop)
            except Exception as e:
                print(f"[WARN] YOLO fallback failed: {e}"); yolo_count = 0
            num_people = max(male + female, yolo_count)

            genders = {'male': male, 'female': female}
            pose_type = get_pose_type_from_array(crop)

            row = {
                "photo_id": photo_id,
                "frame_id": frame_idx,
                "num_people": num_people,
                "gender_distribution": genders,
                "pose_type": pose_type
            }
            row_df = pd.DataFrame([row])
            if not header_written:
                row_df.to_csv(CSV_PATH, mode='w', header=True, index=False)
                header_written = True
            else:
                row_df.to_csv(CSV_PATH, mode='a', header=False, index=False)

    print(f"[INFO] append 완료: {CSV_PATH}")

if __name__ == "__main__":
    main()