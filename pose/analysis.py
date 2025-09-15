# 포즈 분류 파일 (상체 중심 5종)
# 분류 가능한 포즈: peace_sign, thumbs_up, cross_arm, hands_up, big_heart
# 기타: other_pose, no_person

from ultralytics import YOLO
from deepface import DeepFace
import cv2
import math
import mediapipe as mp

mp_pose = mp.solutions.pose

# ──공용 헬퍼 ─────────────────────────────────────────────
def _xy(lm):
    return (lm.x, lm.y)

def _dist(p, q):
    return math.hypot(p[0]-q[0], p[1]-q[1])

def _angle_deg(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    n1 = math.hypot(v1[0], v1[1]); n2 = math.hypot(v2[0], v2[1])
    if n1 == 0 or n2 == 0: return 180.0
    cosv = max(-1.0, min(1.0, dot/(n1*n2)))
    return math.degrees(math.acos(cosv))

def _vec(a, b):
    return (b[0]-a[0], b[1]-a[1])

def _safe_get(lms, idx):
    return lms[idx] if 0 <= idx < len(lms) else None

# ──분류에 쓸 임계값 ───────────────────────────
TH = {
    "NEAR_FACE": 0.55,
    "BIG_HEART_Y": 0.12,
    "HEART_WRISTS_GAP": 0.65,
    "CHEST_Y_BAND": 0.22,
    "THUMB_MIN_UP": 45.0,
    "THUMB_MAX_SIDE": 45.0,
    "INDEX_BELOW": -0.01,
    "PEACE_MIN": 25.0,
    "PEACE_MAX": 110.0,
}

# ──손 단위 제스처 분류 ───────
def _classify_hand_side(landmarks, side="LEFT"):
    WRIST  = mp_pose.PoseLandmark.LEFT_WRIST.value if side=="LEFT" else mp_pose.PoseLandmark.RIGHT_WRIST.value
    INDEX  = mp_pose.PoseLandmark.LEFT_INDEX.value if side=="LEFT" else mp_pose.PoseLandmark.RIGHT_INDEX.value
    THUMB  = mp_pose.PoseLandmark.LEFT_THUMB.value if side=="LEFT" else mp_pose.PoseLandmark.RIGHT_THUMB.value
    NOSE   = mp_pose.PoseLandmark.NOSE.value

    w = landmarks[WRIST]; i = landmarks[INDEX]; t = landmarks[THUMB]; nose = landmarks[NOSE]
    if any(x is None for x in [w,i,t,nose]):
        return None

    wxy, ixy, txy, nxy = _xy(w), _xy(i), _xy(t), _xy(nose)

    LSH = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    RSH = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    if LSH is None or RSH is None:
        return None
    shoulder_w = abs(RSH.x - LSH.x)
    if shoulder_w <= 1e-6:
        return None

    near_face = (_dist(wxy, nxy) <= TH["NEAR_FACE"] * shoulder_w)

    v_wi = _vec(wxy, ixy)  # 손목→검지
    v_wt = _vec(wxy, txy)  # 손목→엄지

    # peace
    ang_it = _angle_deg(v_wi, v_wt)
    if near_face and TH["PEACE_MIN"] <= ang_it <= TH["PEACE_MAX"]:
        return "peace_sign"

    # thumbs-up
    ang_thumb_up = _angle_deg(v_wt, (0.0, -1.0))
    index_below = (ixy[1] - wxy[1]) >= TH["INDEX_BELOW"]
    if (TH["THUMB_MIN_UP"] <= ang_thumb_up <= 180-TH["THUMB_MAX_SIDE"]) and index_below:
        return "thumbs_up"

    return None

# ──포즈 분류 ──────────────────────────────
def classify_pose(landmarks):
    if not landmarks:
        return "no_person"

    LSH = _safe_get(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    RSH = _safe_get(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    LWR = _safe_get(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value)
    RWR = _safe_get(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value)
    NOSE = _safe_get(landmarks, mp_pose.PoseLandmark.NOSE.value)

    if any(x is None for x in [LSH,RSH,LWR,RWR,NOSE]):
        return "other_pose"

    lsh, rsh, lwr, rwr, nose = map(_xy, [LSH,RSH,LWR,RWR,NOSE])
    shoulder_w = abs(rsh[0] - lsh[0])
    if shoulder_w <= 1e-6:
        return "other_pose"

    chest_y = (lsh[1] + rsh[1]) / 2
    mid_x = (lsh[0] + rsh[0]) / 2

    # 1) peace / thumbs
    left = _classify_hand_side(landmarks, "LEFT")
    if left in ("peace_sign", "thumbs_up"):
        return left
    right = _classify_hand_side(landmarks, "RIGHT")
    if right in ("peace_sign", "thumbs_up"):
        return right

    # 2) cross_arm
    wrists_cross_side = (lwr[0] > mid_x*0.98) and (rwr[0] < mid_x*1.02)
    y_mid = (lwr[1] + rwr[1]) / 2
    in_chest_band = abs(y_mid - chest_y) <= TH["CHEST_Y_BAND"]
    near_torso = (abs(lwr[0]-mid_x) <= 0.6*abs(rsh[0]-lsh[0])) and (abs(rwr[0]-mid_x) <= 0.6*abs(rsh[0]-lsh[0]))
    if wrists_cross_side and in_chest_band and near_torso:
        return "cross_arm"

    # 3) hands_up
    cond_one_above = (lwr[1] < lsh[1]) or (rwr[1] < rsh[1])
    cond_both_above_chest = (lwr[1] < chest_y - 0.02) and (rwr[1] < chest_y - 0.02)
    if cond_one_above or cond_both_above_chest:
        return "hands_up"

    # 4) big_heart
    wrists_gap = _dist(lwr, rwr)
    above_head = (lwr[1] < nose[1] - TH["BIG_HEART_Y"]) and (rwr[1] < nose[1] - TH["BIG_HEART_Y"]) 
    if above_head and (wrists_gap <= TH["HEART_WRISTS_GAP"] * shoulder_w):
        return "big_heart"

    return "other_pose"

# ──공개 API ──────────────────────────────
def get_pose_type(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "other_pose"
    with mp_pose.Pose() as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return "no_person"
        return classify_pose(results.pose_landmarks.landmark)

def get_num_people(image_path):
    model = YOLO('yolov8n.pt')
    results = model(image_path)
    return len([obj for obj in results[0].boxes.cls if int(obj) == 0])

def get_gender_distribution(image_path):
    analysis = DeepFace.analyze(img_path=image_path, actions=['gender'])
    genders = {'male': 0, 'female': 0}
    for face in analysis:
        if face['gender'] == 'Man':
            genders['male'] += 1
        else:
            genders['female'] += 1
    return genders

def get_pose_type_from_array(bgr_image):
    with mp.solutions.pose.Pose() as pose:
        results = pose.process(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return "no_person"
        return classify_pose(results.pose_landmarks.landmark)