from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import hashlib
import os
import json
import time
import glob
import random
import numpy as np
import cv2
import dlib
from scipy.spatial import Delaunay
from datetime import datetime
import traceback
import shutil
import re
import threading
import uuid
import multiprocessing
import pickle
import tempfile
from zipfile import BadZipFile
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# 允许跨域和特殊请求头 (ngrok-skip-browser-warning 是为了穿透 Ngrok 的拦截页)
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers=["Content-Type", "Authorization", "ngrok-skip-browser-warning", "X-Merge-Session"])

# --- 全局配置变量 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_local_environment(env_filepath):
    """读取不会提交到 Git 的本地 .env；已有系统环境变量拥有更高优先级。"""
    if not os.path.isfile(env_filepath):
        return
    with open(env_filepath, 'r', encoding='utf-8') as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ.setdefault(key, value)


load_local_environment(os.path.join(BASE_DIR, '.env'))

PREDICTOR_PATH = os.path.join(BASE_DIR, 'shape_predictor_68_face_landmarks.dat')
FACE_ATTRIBUTE_MODEL_PATH = os.path.join(
    BASE_DIR,
    'models',
    'face_attrib_net.onnx'
)
DATABASE_PATH = os.path.join(BASE_DIR, 'database')
PREPROCESSED_DATABASE_PATH = os.path.join(BASE_DIR, 'database_preprocessed')
DATABASE_PREPROCESS_VERSION = 1  # 修改数据库裁剪/关键点规则时同步提升版本并重建。
PARTICIPANT_FACES_PATH = os.path.join(BASE_DIR, 'participant_faces')
GENERATED_FACES_PATH = os.path.join(BASE_DIR, 'generated_faces')
DATA_SAVE_PATH = os.path.join(BASE_DIR, 'data')
# 图片处理尺寸配置
PROCESS_WIDTH = 400
PROCESS_HEIGHT = 533
OUTPUT_WIDTH = 400
OUTPUT_HEIGHT = 533
GLOBAL_MORPH_WORKERS = 4
MAX_ACTIVE_PARTICIPANTS = 1
MERGE_JOB_TTL_SECONDS = 30 * 60
MERGE_JOB_LEASE_SECONDS = 25
MERGE_POLL_INTERVAL_SECONDS = 0.5
MERGE_WORK_PATH = os.path.join(BASE_DIR, '.merge_work')
DATA_SAVE_LOCK = threading.Lock()
FACE_ATTRIBUTE_LOCK = threading.Lock()
MERGE_JOB_LOCK = threading.Lock()
MERGE_PROCESS_START_LOCK = threading.Lock()
SONA_ACTIVE_ATTEMPTS = set()
SONA_CREDIT_LOCKS = {}
MERGE_JOBS = {}
MERGE_JOB_BY_REQUEST = {}
MERGE_SESSION_CANCELLATIONS = {}
# 仅一名被试进入处理；该被试的六组图片由最多四个可终止的子进程计算。
PARTICIPANT_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_ACTIVE_PARTICIPANTS)
# SONA 网络请求不占用 Flask 请求线程，也不长期持有全体被试共用的保存锁。
SONA_EXECUTOR = ThreadPoolExecutor(max_workers=2)
SONA_COMPLETION_API_URL = os.environ.get(
    'SONA_COMPLETION_API_URL',
    'https://bristolpsych.sona-systems.com/services/SonaAPI.svc/WebstudyCredit'
)
SONA_EXPERIMENT_ID = os.environ.get('SONA_EXPERIMENT_ID', '2514')
# 安全起见，credit token 只能由后端环境变量提供，不能提交到公开仓库。
SONA_CREDIT_TOKEN = os.environ.get('SONA_CREDIT_TOKEN', '').strip()
SONA_REQUEST_TIMEOUT_SECONDS = 15

# FaceAttribNet 输出顺序：左眼睁开、右眼睁开、普通眼镜、口罩、太阳镜。
# 本实验只使用普通眼镜和太阳镜两个输出；模型仅对两张上传照片各执行一次。
FACE_ATTRIBUTE_INPUT_SIZE = 128
FACE_ATTRIBUTE_LABELS = (
    'left_eye_open',
    'right_eye_open',
    'eyeglasses',
    'mask',
    'sunglasses'
)
FACE_OCCLUSION_THRESHOLDS = {
    'eyeglasses': 0.70,
    'sunglasses': 0.70
}

FACE_CANVAS_VALUE = 255
FACE_ALIGN_LEFT_EYE = (0.34, 0.40)
FACE_ALIGN_RIGHT_EYE = (0.66, 0.40)
FACE_ALIGN_MOUTH = (0.50, 0.69)
# AFA 默认使用眉、眼、鼻和外唇进行广义 Procrustes 对齐，排除较不稳定的
# 下颌线与内唇。实现依据 Gaspar & Garrod (2025), DOI: 10.5334/jors.542，
# 并针对本项目的内存图像管线重写，不依赖 AFA 的文件夹式批处理接口。
AFA_ALIGNMENT_LANDMARK_INDICES = np.arange(17, 60, dtype=np.int32)
AFA_GPA_MAX_ITERATIONS = 64
AFA_GPA_TOLERANCE = 1e-7
# 外轮廓点只作为Delaunay控制点，不作为最终合成遮罩。固定角度与固定
# 颈肩高度确保两张不同人像之间始终存在一一对应的轮廓点。
OUTER_HEAD_ANGLE_DEGREES = np.linspace(170.0, 370.0, 29, dtype=np.float32)
OUTER_SHOULDER_LEVEL_FRACTIONS = (0.16, 0.34, 0.54, 0.74)
OUTER_CONTOUR_POINT_COUNT = (
    len(OUTER_HEAD_ANGLE_DEGREES) +
    2 * len(OUTER_SHOULDER_LEVEL_FRACTIONS)
)
OUTER_GEOMETRY_POINT_COUNT = 68 + OUTER_CONTOUR_POINT_COUNT + 8
OUTER_CONTOUR_GRABCUT_ITERATIONS = 3
OUTER_CONTOUR_MIN_AREA_RATIO = 0.035
OUTER_CONTOUR_MAX_AREA_RATIO = 0.88

# 外层已经并行处理不同刺激图，限制 OpenCV 内部线程以避免笔记本过度抢占。
cv2.setNumThreads(1)

# 确保数据及图片文件夹存在
os.makedirs(DATA_SAVE_PATH, exist_ok=True)
os.makedirs(PARTICIPANT_FACES_PATH, exist_ok=True)
os.makedirs(GENERATED_FACES_PATH, exist_ok=True)
os.makedirs(MERGE_WORK_PATH, exist_ok=True)
os.makedirs(os.path.join(DATABASE_PATH, 'male'), exist_ok=True)
os.makedirs(os.path.join(DATABASE_PATH, 'female'), exist_ok=True)

# --- 初始化 Dlib 模型 ---
detector = None
predictor = None
face_attribute_net = None
FACE_ATTRIBUTE_MODEL_ERROR = ''
# 批量融合进程只使用已检测出的特征点，避免每个进程重复加载模型。
if os.environ.get('_EXPERIMENT_MORPH_BATCH_WORKER') != '1':
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
        print("[OK] Dlib models loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Dlib model could not be loaded from '{PREDICTOR_PATH}'.")
        print(e)

    try:
        if not os.path.isfile(FACE_ATTRIBUTE_MODEL_PATH):
            raise FileNotFoundError(
                f"Face attribute model not found: {FACE_ATTRIBUTE_MODEL_PATH}"
            )
        face_attribute_net = cv2.dnn.readNetFromONNX(FACE_ATTRIBUTE_MODEL_PATH)
        print("[OK] Face occlusion model loaded successfully.")
    except Exception as e:
        FACE_ATTRIBUTE_MODEL_ERROR = str(e)
        print(f"[ERROR] Face occlusion model could not be loaded: {e}")

# --- 核心图像处理函数 ---
# 这些函数负责特征点检测、规范对齐、三角剖分、仿射变换和人脸融合。

def get_points(image):
    try:
        if image is None: return np.array([])
        if detector is None or predictor is None: return np.array([])
        
        dets = detector(image, 1)
        if len(dets) == 0:
            h, w = image.shape[:2]
            return np.array([
                [0,0], [w//2,0], [w-1,0],
                [w-1, h//2], [w-1, h-1], [w//2, h-1],
                [0, h-1], [0, h//2]
            ])

        detected_face = dets[0]
        pose_landmarks = predictor(image, detected_face)
        points = []
        for p in pose_landmarks.parts():
            points.append([p.x, p.y])
        
        h, w = image.shape[:2]
        x = w - 1
        y = h - 1
        boundary_points = [
            [0, 0], [x // 2, 0], [x, 0],
            [x, y // 2], [x, y], [x // 2, y],
            [0, y], [0, y // 2]
        ]
        points.extend(boundary_points)
        return np.array(points)
    except Exception as e:
        print(f"Error in get_points: {e}")
        return np.array([])


def _prepare_face_attribute_input(image, points):
    """从68点区域构造FaceAttribNet所需的128×128 RGB letterbox张量。"""
    face_points = np.asarray(points[:68], dtype=np.float32)
    jaw = face_points[0:17]
    brows = face_points[17:27]
    jaw_left = float(np.min(jaw[:, 0]))
    jaw_right = float(np.max(jaw[:, 0]))
    brow_top = float(np.min(brows[:, 1]))
    chin_y = float(np.max(jaw[:, 1]))
    face_width = max(1.0, jaw_right - jaw_left)
    face_height = max(1.0, chin_y - brow_top)

    height, width = image.shape[:2]
    x1 = max(0, int(round(jaw_left - 0.18 * face_width)))
    x2 = min(width, int(round(jaw_right + 0.18 * face_width)))
    y1 = max(0, int(round(brow_top - 0.52 * face_height)))
    y2 = min(height, int(round(chin_y + 0.12 * face_height)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("The detected face crop is invalid.")

    face_crop = image[y1:y2, x1:x2]
    crop_height, crop_width = face_crop.shape[:2]
    scale = min(
        FACE_ATTRIBUTE_INPUT_SIZE / float(crop_width),
        FACE_ATTRIBUTE_INPUT_SIZE / float(crop_height)
    )
    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))
    resized = cv2.resize(
        face_crop,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    )

    letterbox = np.zeros(
        (FACE_ATTRIBUTE_INPUT_SIZE, FACE_ATTRIBUTE_INPUT_SIZE, 3),
        dtype=np.uint8
    )
    offset_x = (FACE_ATTRIBUTE_INPUT_SIZE - resized_width) // 2
    offset_y = (FACE_ATTRIBUTE_INPUT_SIZE - resized_height) // 2
    letterbox[
        offset_y:offset_y + resized_height,
        offset_x:offset_x + resized_width
    ] = resized

    rgb = cv2.cvtColor(letterbox, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb /= 255.0
    return np.transpose(rgb, (2, 0, 1))[None, ...]


def assess_face_occlusion(image, points):
    """检测普通眼镜和太阳镜，返回模型概率与问题码。"""
    if face_attribute_net is None:
        raise RuntimeError(
            "Face occlusion checking is unavailable: " +
            (FACE_ATTRIBUTE_MODEL_ERROR or "model was not initialized")
        )

    model_input = _prepare_face_attribute_input(image, points)
    with FACE_ATTRIBUTE_LOCK:
        face_attribute_net.setInput(model_input)
        raw_output = face_attribute_net.forward()

    values = np.asarray(raw_output, dtype=np.float32).reshape(-1)
    if len(values) < len(FACE_ATTRIBUTE_LABELS):
        raise RuntimeError("Face occlusion model returned an invalid output.")
    probabilities = {
        label: float(np.clip(values[index], 0.0, 1.0))
        for index, label in enumerate(FACE_ATTRIBUTE_LABELS)
    }

    issues = []
    # 太阳镜常会同时触发普通眼镜输出，只保留更明确的太阳镜原因。
    if probabilities['sunglasses'] >= FACE_OCCLUSION_THRESHOLDS['sunglasses']:
        issues.append('sunglasses_detected')
    elif probabilities['eyeglasses'] >= FACE_OCCLUSION_THRESHOLDS['eyeglasses']:
        issues.append('eyeglasses_detected')

    return probabilities, issues


def _normalize_landmark_shape(points, fit_indices):
    """按 AFA/GPA 约定平移到原点并统一形状尺度。"""
    shape = np.asarray(points, dtype=np.float64)
    fit_points = shape[fit_indices]
    center = np.mean(fit_points, axis=0)
    centered = shape - center
    shape_norm = float(np.linalg.norm(centered[fit_indices]))
    if not np.isfinite(shape_norm) or shape_norm <= 1e-8:
        raise ValueError("Face landmarks have an invalid Procrustes scale.")
    return centered / shape_norm


def _orthogonal_procrustes_rotation(source, target):
    """返回将行向量 source 最小二乘旋转到 target 的无反射矩阵。"""
    covariance = np.asarray(source, dtype=np.float64).T @ np.asarray(
        target,
        dtype=np.float64
    )
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    correction = np.eye(2, dtype=np.float64)
    if np.linalg.det(u_matrix @ vt_matrix) < 0:
        correction[-1, -1] = -1.0
    return u_matrix @ correction @ vt_matrix


def _estimate_similarity_transform(source_points, target_points):
    """以全部给定控制点估计无反射、各向同性的相似变换。"""
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("Source and target landmarks must have matching Nx2 shapes.")

    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    denominator = float(np.sum(source_centered * source_centered))
    if not np.isfinite(denominator) or denominator <= 1e-8:
        raise ValueError("Cannot estimate a transform from degenerate landmarks.")

    rotation = _orthogonal_procrustes_rotation(
        source_centered,
        target_centered
    )
    rotated_source = source_centered @ rotation
    scale = float(
        np.sum(rotated_source * target_centered) / denominator
    )
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Estimated face-alignment scale is invalid.")

    translation = target_center - scale * (source_center @ rotation)
    linear = (scale * rotation).T
    return np.column_stack([linear, translation]).astype(np.float32)


def _transform_landmarks(points, transform):
    """将 OpenCV 2x3 仿射矩阵应用到任意数量的二维关键点。"""
    landmarks = np.asarray(points, dtype=np.float32)
    return (
        landmarks @ transform[:, :2].T + transform[:, 2]
    ).astype(np.float32)


def build_afa_reference_shape(landmark_sets, output_width, output_height):
    """对同一刺激组执行 GPA，并把共识形状放入统一实验画布。"""
    shapes = [
        np.asarray(points[:68], dtype=np.float64)
        for points in landmark_sets
        if points is not None and len(points) >= 68
    ]
    if len(shapes) != len(landmark_sets) or len(shapes) < 2:
        raise ValueError("AFA alignment requires at least two valid 68-point faces.")

    fit_indices = AFA_ALIGNMENT_LANDMARK_INDICES
    normalized_shapes = [
        _normalize_landmark_shape(shape, fit_indices)
        for shape in shapes
    ]
    reference = normalized_shapes[0][fit_indices]
    mean_shape = normalized_shapes[0]

    for _ in range(AFA_GPA_MAX_ITERATIONS):
        aligned_shapes = []
        for shape in normalized_shapes:
            rotation = _orthogonal_procrustes_rotation(
                shape[fit_indices],
                reference
            )
            aligned_shapes.append(shape @ rotation)

        candidate = np.mean(aligned_shapes, axis=0)
        candidate = _normalize_landmark_shape(candidate, fit_indices)
        next_reference = candidate[fit_indices]
        change = float(np.linalg.norm(next_reference - reference))
        mean_shape = candidate
        reference = next_reference
        if change <= AFA_GPA_TOLERANCE:
            break

    source_anchors = np.float32([
        np.mean(mean_shape[36:42], axis=0),
        np.mean(mean_shape[42:48], axis=0),
        (mean_shape[48] + mean_shape[54]) / 2.0
    ])
    destination_anchors = np.float32([
        [FACE_ALIGN_LEFT_EYE[0] * output_width,
         FACE_ALIGN_LEFT_EYE[1] * output_height],
        [FACE_ALIGN_RIGHT_EYE[0] * output_width,
         FACE_ALIGN_RIGHT_EYE[1] * output_height],
        [FACE_ALIGN_MOUTH[0] * output_width,
         FACE_ALIGN_MOUTH[1] * output_height]
    ])
    canvas_transform = _estimate_similarity_transform(
        source_anchors,
        destination_anchors
    )
    return _transform_landmarks(mean_shape, canvas_transform)


def align_face_collection_afa(images, landmark_sets):
    """按 AFA 流程把一组人脸对齐到同一个 GPA 共识形状。"""
    if len(images) != len(landmark_sets):
        raise ValueError("Images and landmark sets must have the same length.")

    reference_shape = build_afa_reference_shape(
        landmark_sets,
        PROCESS_WIDTH,
        PROCESS_HEIGHT
    )
    fit_indices = AFA_ALIGNMENT_LANDMARK_INDICES
    aligned_faces = []
    for image, landmarks in zip(images, landmark_sets):
        face_points = np.asarray(landmarks[:68], dtype=np.float32)
        transform = _estimate_similarity_transform(
            face_points[fit_indices],
            reference_shape[fit_indices]
        )
        aligned_image = cv2.warpAffine(
            image,
            transform,
            (PROCESS_WIDTH, PROCESS_HEIGHT),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(FACE_CANVAS_VALUE,) * 3
        )
        aligned_points = _transform_landmarks(face_points, transform)
        aligned_faces.append((aligned_image, aligned_points))
    return aligned_faces


def _outer_contour_measurements(landmarks):
    """从68点得到外轮廓搜索所需的稳定人脸尺度和中心。"""
    face_points = np.asarray(landmarks[:68], dtype=np.float32)
    jaw = face_points[0:17]
    brows = face_points[17:27]
    jaw_left = float(np.min(jaw[:, 0]))
    jaw_right = float(np.max(jaw[:, 0]))
    brow_top = float(np.min(brows[:, 1]))
    chin_y = float(np.max(jaw[:, 1]))
    face_width = max(1.0, jaw_right - jaw_left)
    face_height = max(1.0, chin_y - brow_top)
    center_x = 0.5 * (jaw_left + jaw_right)
    # 射线中心放在上半脸，向左右搜索时落在耳侧而不是肩膀。
    center_y = brow_top + 0.42 * face_height
    return {
        "center": np.float32([center_x, center_y]),
        "jaw_left": jaw_left,
        "jaw_right": jaw_right,
        "brow_top": brow_top,
        "chin_y": chin_y,
        "face_width": face_width,
        "face_height": face_height
    }


def _component_containing_face(binary_mask, face_seed):
    """只保留与面部种子重叠最多的连通人物区域。"""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary_mask > 0).astype(np.uint8),
        connectivity=8
    )
    if count <= 1:
        return None

    seed_pixels = face_seed > 0
    best_label = 0
    best_overlap = 0
    for label in range(1, count):
        overlap = int(np.count_nonzero((labels == label) & seed_pixels))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label

    if best_label == 0:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = int(np.argmax(component_areas)) + 1

    component = np.where(labels == best_label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    solid = np.zeros_like(component)
    cv2.drawContours(
        solid,
        [max(contours, key=cv2.contourArea)],
        -1,
        255,
        thickness=-1
    )
    return solid


def estimate_person_outer_silhouette(image, landmarks):
    """用本地GrabCut估计人物外轮廓；结果仅用于放置几何控制点。"""
    if image is None or landmarks is None or len(landmarks) < 68:
        return None

    height, width = image.shape[:2]
    if height < 8 or width < 8:
        return None

    face_points = np.asarray(landmarks[:68], dtype=np.float32)
    metrics = _outer_contour_measurements(face_points)
    center_x = metrics["center"][0]
    face_width = metrics["face_width"]
    face_height = metrics["face_height"]
    chin_y = metrics["chin_y"]

    grabcut_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)

    # 头发、耳朵与颈部作为可能前景；下方梯形覆盖可能出现的肩膀。
    head_center = (
        int(round(center_x)),
        int(round(metrics["brow_top"] + 0.28 * face_height))
    )
    head_axes = (
        max(4, int(round(0.82 * face_width))),
        max(4, int(round(1.02 * face_height)))
    )
    cv2.ellipse(
        grabcut_mask,
        head_center,
        head_axes,
        0,
        0,
        360,
        cv2.GC_PR_FGD,
        thickness=-1
    )

    shoulder_top = int(np.clip(round(chin_y), 1, height - 2))
    shoulder_bottom = height - 2
    shoulder_top_half_width = 0.42 * face_width
    shoulder_bottom_half_width = max(0.95 * face_width, 0.42 * width)
    shoulder_polygon = np.float32([
        [center_x - shoulder_top_half_width, shoulder_top],
        [center_x + shoulder_top_half_width, shoulder_top],
        [center_x + shoulder_bottom_half_width, shoulder_bottom],
        [center_x - shoulder_bottom_half_width, shoulder_bottom]
    ])
    shoulder_polygon[:, 0] = np.clip(shoulder_polygon[:, 0], 1, width - 2)
    cv2.fillConvexPoly(
        grabcut_mask,
        np.int32(np.round(shoulder_polygon)),
        cv2.GC_PR_FGD
    )

    # 68点凸包是可靠的确定前景种子，但不会作为最终融合遮罩使用。
    face_seed = np.zeros((height, width), dtype=np.uint8)
    face_hull = cv2.convexHull(np.int32(np.round(face_points)))
    cv2.fillConvexPoly(face_seed, face_hull, 255)
    grabcut_mask[face_seed > 0] = cv2.GC_FGD

    neck_half_width = max(3, int(round(0.22 * face_width)))
    neck_top = int(np.clip(round(chin_y - 0.04 * face_height), 1, height - 2))
    neck_bottom = int(np.clip(round(chin_y + 0.24 * face_height), 1, height - 2))
    cv2.rectangle(
        grabcut_mask,
        (max(1, int(round(center_x)) - neck_half_width), neck_top),
        (min(width - 2, int(round(center_x)) + neck_half_width), neck_bottom),
        cv2.GC_FGD,
        thickness=-1
    )

    # 画布边界保持确定背景，防止前景区域退化成整张矩形。
    grabcut_mask[0:2, :] = cv2.GC_BGD
    grabcut_mask[-2:, :] = cv2.GC_BGD
    grabcut_mask[:, 0:2] = cv2.GC_BGD
    grabcut_mask[:, -2:] = cv2.GC_BGD

    try:
        background_model = np.zeros((1, 65), dtype=np.float64)
        foreground_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            image,
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            OUTER_CONTOUR_GRABCUT_ITERATIONS,
            cv2.GC_INIT_WITH_MASK
        )
    except Exception as error:
        print(f"Outer contour GrabCut fallback: {error}")
        return None

    foreground = np.where(
        (grabcut_mask == cv2.GC_FGD) |
        (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2
    )
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )
    foreground = _component_containing_face(foreground, face_seed)
    if foreground is None:
        return None

    area_ratio = float(np.count_nonzero(foreground)) / float(height * width)
    if (
        area_ratio < OUTER_CONTOUR_MIN_AREA_RATIO or
        area_ratio > OUTER_CONTOUR_MAX_AREA_RATIO
    ):
        return None
    return foreground


def _fallback_head_point(metrics, angle_radians, image_shape):
    """分割失败时使用与面部尺度绑定的保守头部椭圆。"""
    height, width = image_shape[:2]
    center = metrics["center"].astype(np.float64)
    direction = np.array([
        np.cos(angle_radians),
        np.sin(angle_radians)
    ], dtype=np.float64)
    radius_x = max(2.0, 0.70 * metrics["face_width"])
    radius_y = max(2.0, 0.92 * metrics["face_height"])
    denominator = np.sqrt(
        (direction[0] / radius_x) ** 2 +
        (direction[1] / radius_y) ** 2
    )
    radius = 1.0 / max(denominator, 1e-8)
    point = center + radius * direction
    point[0] = np.clip(point[0], 1, width - 2)
    point[1] = np.clip(point[1], 1, height - 2)
    return point.astype(np.float32)


def _sample_silhouette_ray(mask, center, angle_radians, minimum_radius):
    """沿固定方向取得与人物连通区域相交的最外像素。"""
    if mask is None:
        return None
    height, width = mask.shape[:2]
    direction = np.array([
        np.cos(angle_radians),
        np.sin(angle_radians)
    ], dtype=np.float64)
    corner_distances = [
        np.hypot(center[0], center[1]),
        np.hypot(width - 1 - center[0], center[1]),
        np.hypot(center[0], height - 1 - center[1]),
        np.hypot(width - 1 - center[0], height - 1 - center[1])
    ]
    maximum_radius = max(corner_distances)
    sample_count = max(2, int(np.ceil(maximum_radius - minimum_radius)) + 1)
    radii = np.linspace(minimum_radius, maximum_radius, sample_count)
    coordinates = center[None, :] + radii[:, None] * direction[None, :]
    x_values = np.rint(coordinates[:, 0]).astype(np.int32)
    y_values = np.rint(coordinates[:, 1]).astype(np.int32)
    valid = (
        (x_values >= 1) & (x_values < width - 1) &
        (y_values >= 1) & (y_values < height - 1)
    )
    if not np.any(valid):
        return None
    x_values = x_values[valid]
    y_values = y_values[valid]
    foreground_indices = np.flatnonzero(mask[y_values, x_values] > 0)
    if foreground_indices.size == 0:
        return None
    last_index = int(foreground_indices[-1])
    return np.float32([x_values[last_index], y_values[last_index]])


def _sample_silhouette_row(mask, y_value, center_x):
    """在固定高度获取人物轮廓左右端点。"""
    if mask is None:
        return None
    height, width = mask.shape[:2]
    y_center = int(np.clip(round(y_value), 1, height - 2))
    y1 = max(1, y_center - 2)
    y2 = min(height - 1, y_center + 3)
    row_coverage = np.mean(mask[y1:y2] > 0, axis=0)
    foreground_x = np.flatnonzero(row_coverage >= 0.40)
    if foreground_x.size < 2:
        return None

    # 选择包含身体中心的连续区段，避免偶然背景区域成为肩部端点。
    split_positions = np.flatnonzero(np.diff(foreground_x) > 1) + 1
    segments = np.split(foreground_x, split_positions)
    center_column = int(np.clip(round(center_x), 0, width - 1))
    containing = [
        segment for segment in segments
        if segment.size > 0 and segment[0] <= center_column <= segment[-1]
    ]
    segment = max(
        containing if containing else segments,
        key=lambda values: values.size
    )
    return np.float32([
        [max(1, int(segment[0])), y_center],
        [min(width - 2, int(segment[-1])), y_center]
    ])


def estimate_outer_contour_points(image, landmarks):
    """生成固定数量、固定语义顺序的头发/耳侧与颈肩控制点。"""
    face_points = np.asarray(landmarks[:68], dtype=np.float32)
    metrics = _outer_contour_measurements(face_points)
    center = metrics["center"].astype(np.float64)
    silhouette = estimate_person_outer_silhouette(image, face_points)
    image_shape = image.shape

    outer_points = []
    minimum_radius = 0.38 * metrics["face_width"]
    for angle_degrees in OUTER_HEAD_ANGLE_DEGREES:
        angle_radians = np.deg2rad(float(angle_degrees))
        point = _sample_silhouette_ray(
            silhouette,
            center,
            angle_radians,
            minimum_radius
        )
        if point is None:
            point = _fallback_head_point(
                metrics,
                angle_radians,
                image_shape
            )
        outer_points.append(point)

    height, width = image_shape[:2]
    chin_y = metrics["chin_y"]
    available_height = max(1.0, (height - 2) - chin_y)
    for level_fraction in OUTER_SHOULDER_LEVEL_FRACTIONS:
        y_value = chin_y + level_fraction * available_height
        row_points = _sample_silhouette_row(
            silhouette,
            y_value,
            metrics["center"][0]
        )
        if row_points is None:
            half_width = metrics["face_width"] * (
                0.42 + 0.72 * level_fraction
            )
            row_points = np.float32([
                [np.clip(metrics["center"][0] - half_width, 1, width - 2),
                 np.clip(y_value, 1, height - 2)],
                [np.clip(metrics["center"][0] + half_width, 1, width - 2),
                 np.clip(y_value, 1, height - 2)]
            ])
        outer_points.extend(row_points)

    return np.asarray(outer_points, dtype=np.float32)


def build_outer_contour_geometry(image, landmarks):
    """组合68点、外轮廓点和8个画布边界点供整图Delaunay使用。"""
    face_points = np.asarray(landmarks[:68], dtype=np.float32)
    if image is None or len(face_points) < 68:
        return np.array([])
    height, width = image.shape[:2]
    outer_points = estimate_outer_contour_points(image, face_points)
    if len(outer_points) != OUTER_CONTOUR_POINT_COUNT:
        return np.array([])

    x_max = width - 1
    y_max = height - 1
    boundary_points = np.float32([
        [0, 0], [x_max // 2, 0], [x_max, 0],
        [x_max, y_max // 2], [x_max, y_max], [x_max // 2, y_max],
        [0, y_max], [0, y_max // 2]
    ])
    return np.vstack([face_points, outer_points, boundary_points])

def get_triangles(points):
    try:
        return Delaunay(points).simplices
    except:
        return []

def affine_transform(input_image, input_triangle, output_triangle, size):
    try:
        warp_matrix = cv2.getAffineTransform(
            np.float32(input_triangle), np.float32(output_triangle))
        output_image = cv2.warpAffine(input_image, warp_matrix, (size[0], size[1]), None,
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return output_image
    except:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

def morph_triangle(img_src, t_src, t_dst):
    try:
        rect_src = cv2.boundingRect(np.float32([t_src]))
        rect_dst = cv2.boundingRect(np.float32([t_dst]))

        t_src_rect = []
        t_dst_rect = []

        for i in range(0, 3):
            t_src_rect.append(((t_src[i][0] - rect_src[0]), (t_src[i][1] - rect_src[1])))
            t_dst_rect.append(((t_dst[i][0] - rect_dst[0]), (t_dst[i][1] - rect_dst[1])))

        x, y, w, h = rect_src
        pad = 20
        needs_padding = x < 0 or y < 0 or x + w > img_src.shape[1] or y + h > img_src.shape[0]
        
        if needs_padding:
            img_padded = cv2.copyMakeBorder(img_src, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
            x_pad, y_pad = x + pad, y + pad
            img_src_crop = img_padded[y_pad:y_pad+h, x_pad:x_pad+w]
        else:
            img_src_crop = img_src[y:y+h, x:x+w]

        if img_src_crop.size == 0 or img_src_crop.shape[:2] != (h, w):
            return None

        size = (rect_dst[2], rect_dst[3])
        img_warped = affine_transform(img_src_crop, t_src_rect, t_dst_rect, size)
        return img_warped, rect_dst
    except:
        return None

def warp_image(img, src_points, dst_points, triangles):
    warped_img = np.zeros(img.shape, dtype=img.dtype)
    for i in triangles:
        x, y, z = i[0], i[1], i[2]
        t_src = [src_points[x], src_points[y], src_points[z]]
        t_dst = [dst_points[x], dst_points[y], dst_points[z]]
        res = morph_triangle(img, t_src, t_dst)
        if res is None: continue
        warped_tri, rect_dst = res
        x, y, w, h = rect_dst
        mask = np.zeros((h, w, 3), dtype=np.float32)
        t_dst_rect = [(p[0] - x, p[1] - y) for p in t_dst]
        cv2.fillConvexPoly(mask, np.int32(t_dst_rect), (1.0, 1.0, 1.0), 16, 0)
        y1, y2 = max(0, y), min(warped_img.shape[0], y+h)
        x1, x2 = max(0, x), min(warped_img.shape[1], x+w)
        if y1 >= y2 or x1 >= x2: continue
        mask_slice = mask[y1-y:y2-y, x1-x:x2-x]
        warp_slice = warped_tri[y1-y:y2-y, x1-x:x2-x]
        current_slice = warped_img[y1:y2, x1:x2]
        warped_img[y1:y2, x1:x2] = current_slice * (1 - mask_slice) + warp_slice * mask_slice
    return warped_img


def morph_faces_full(
    img1_arr,
    img2_arr,
    alpha=0.5,
    points1=None,
    points2=None
):
    """在AFA对齐结果上执行外轮廓增强的整图Delaunay与线性融合。"""
    try:
        img1_arr = cv2.resize(img1_arr, (PROCESS_WIDTH, PROCESS_HEIGHT))
        img2_arr = cv2.resize(img2_arr, (PROCESS_WIDTH, PROCESS_HEIGHT))
        img1 = np.copy(img1_arr)
        img2 = np.copy(img2_arr)

        if points1 is None:
            points1 = get_points(img1)
        if points2 is None:
            points2 = get_points(img2)

        points1 = np.asarray(points1, dtype=np.float32)
        points2 = np.asarray(points2, dtype=np.float32)

        if len(points1) == 0 or len(points2) == 0 or len(points1) != len(points2):
            return cv2.addWeighted(img1, 1-alpha, img2, alpha, 0), None

        # 68个人脸点负责五官，固定顺序的外轮廓点负责头发、耳侧与颈肩，
        # 最后8个画布点保证三角网格覆盖整张图像。
        points_avg = (1 - alpha) * points1 + alpha * points2
        triangles = get_triangles(points_avg)
        if len(triangles) == 0:
            return cv2.addWeighted(img1, 1-alpha, img2, alpha, 0), None
        warp1 = warp_image(img1, points1, points_avg, triangles)
        warp2 = warp_image(img2, points2, points_avg, triangles)
        final_img = cv2.addWeighted(warp1, 1-alpha, warp2, alpha, 0)
        return final_img, points_avg
    except Exception as e:
        print(f"Morphing error: {e}")
        s1 = cv2.resize(img1_arr, (PROCESS_WIDTH, PROCESS_HEIGHT))
        s2 = cv2.resize(img2_arr, (PROCESS_WIDTH, PROCESS_HEIGHT))
        return cv2.addWeighted(s1, 1-alpha, s2, alpha, 0), None





# --- 固定比例矩形头像裁剪回退 ---
def crop_face_tight(image, landmarks):
    if image is None: return None
    if landmarks is None or len(landmarks) < 68:
        return cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    h_img, w_img = image.shape[:2]
    face_pts = landmarks[0:68]
    min_x, max_x = np.min(face_pts[:, 0]), np.max(face_pts[:, 0])
    min_y, max_y = np.min(face_pts[:, 1]), np.max(face_pts[:, 1])
    face_w = max_x - min_x
    face_h = max_y - min_y
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    target_aspect = 3.0 / 4.0
    if face_h == 0: face_h = 1
    face_aspect = face_w / face_h
    if face_aspect > target_aspect:
        crop_w = face_w
        crop_h = crop_w / target_aspect
    else:
        crop_h = face_h
        crop_w = crop_h * target_aspect
    crop_w = int(crop_w)
    crop_h = int(crop_h)
    x1 = int(center_x - crop_w // 2)
    y1 = int(center_y - crop_h // 2)
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w_img)
    pad_bottom = max(0, y2 - h_img)
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        try:
            img_padded = cv2.copyMakeBorder(
                image,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REFLECT_101
            )
            crop = img_padded[
                y1 + pad_top:y2 + pad_top,
                x1 + pad_left:x2 + pad_left
            ]
        except:
            return cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    else:
        crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    return cv2.resize(
        crop,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        interpolation=cv2.INTER_LANCZOS4
    )

# --- 矩形头像标准化（用于上传图和数据库图） ---
def crop_portrait_wide(image):
    if image is None: return None
    h, w = image.shape[:2]
    target_aspect = 3.0 / 4.0
    try:
        dets = detector(image, 1)
        if len(dets) == 0:
            return cv2.resize(image, (PROCESS_WIDTH, PROCESS_HEIGHT))
        face = dets[0]
        cx = (face.left() + face.right()) // 2
        cy = (face.top() + face.bottom()) // 2
        fw = face.right() - face.left()
        crop_w = int(fw * 1.8)
        crop_h = int(crop_w / target_aspect)
        y1 = cy - int(crop_h * 0.45)
        x1 = cx - crop_w // 2
        y2 = y1 + crop_h
        x2 = x1 + crop_w
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            img_padded = cv2.copyMakeBorder(
                image,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REFLECT_101
            )
            crop = img_padded[
                y1 + pad_top:y2 + pad_top,
                x1 + pad_left:x2 + pad_left
            ]
        else:
            crop = image[y1:y2, x1:x2]
        return cv2.resize(crop, (PROCESS_WIDTH, PROCESS_HEIGHT))
    except:
        return cv2.resize(image, (PROCESS_WIDTH, PROCESS_HEIGHT))

# --- 文件保存辅助函数 ---
def save_uploaded_image(image, photo_batch_id, role):
    """将一组被试照片保存到独立批次文件夹，不与合成素材混放。"""
    save_dir = os.path.join(PARTICIPANT_FACES_PATH, photo_batch_id)
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{role}.jpg"
    filepath = os.path.join(save_dir, filename)
    if not cv2.imwrite(filepath, image):
        raise IOError(f"Failed to save participant photograph: {role}")
    return f"{photo_batch_id}/{filename}"

class PreprocessedDatabaseError(RuntimeError):
    code = 'DATABASE_PREPROCESSING_ERROR'
    http_status = 503


def database_source_files(gender):
    """只使用原实验规则指定的 male/female jpg、png，不包含 face_select。"""
    folder = 'male' if gender == 'male' else 'female'
    path = os.path.join(DATABASE_PATH, folder)
    return sorted(
        glob.glob(os.path.join(path, '*.jpg'))
        + glob.glob(os.path.join(path, '*.png'))
    )


def database_source_key(filepath):
    return os.path.relpath(filepath, DATABASE_PATH).replace(os.sep, '/')


def database_file_sha256(filepath):
    digest = hashlib.sha256()
    with open(filepath, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_preprocessed_database_manifest():
    """只验证离线结果与源图一致；实验运行时绝不重算数据库关键点。"""
    rebuild_message = (
        'Database face preprocessing is missing or outdated. Stop the server '
        'and run: python preprocess_database.py'
    )
    manifest_path = os.path.join(PREPROCESSED_DATABASE_PATH, 'manifest.json')
    try:
        with open(manifest_path, encoding='utf-8') as source:
            manifest = json.load(source)
        predictor_stat = os.stat(PREDICTOR_PATH)
        if (
            manifest.get('version') != DATABASE_PREPROCESS_VERSION
            or manifest.get('width') != PROCESS_WIDTH
            or manifest.get('height') != PROCESS_HEIGHT
            or manifest.get('predictor_size') != predictor_stat.st_size
            or manifest.get('predictor_mtime_ns') != predictor_stat.st_mtime_ns
        ):
            raise ValueError('Preprocessing settings or landmark model changed.')
        entries = manifest['faces']
        if not isinstance(entries, dict):
            raise ValueError('Invalid face manifest.')
        current_files = database_source_files('male') + database_source_files('female')
        if set(entries) != {database_source_key(path) for path in current_files}:
            raise ValueError('Database face files changed.')
        for filepath in current_files:
            entry = entries[database_source_key(filepath)]
            stat = os.stat(filepath)
            if (
                not isinstance(entry, dict)
                or entry.get('source_size') != stat.st_size
                or entry.get('source_mtime_ns') != stat.st_mtime_ns
            ):
                raise ValueError(f'Database face changed: {os.path.basename(filepath)}')
        return entries
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PreprocessedDatabaseError(f'{rebuild_message} ({error})') from error


def load_prepared_db_image(filepath, entries):
    """从脚本生成的持久化文件读取规范头像及68点，不再检测数据库人脸。"""
    key = database_source_key(filepath)
    entry = entries.get(key)
    if not isinstance(entry, dict) or not entry.get('valid'):
        raise PreprocessedDatabaseError(f'Database face is not preprocessed: {key}')
    rebuild_message = f'Invalid preprocessed database face {key}; run python preprocess_database.py'
    try:
        if database_file_sha256(filepath) != entry['source_sha256']:
            raise ValueError('Source image content changed.')
        root = os.path.abspath(PREPROCESSED_DATABASE_PATH)
        artifact = entry['artifact']
        if not isinstance(artifact, str) or not artifact.endswith('.npz'):
            raise ValueError('Invalid preprocessed artifact name.')
        artifact_path = os.path.abspath(os.path.join(root, artifact))
        if os.path.commonpath([root, artifact_path]) != root:
            raise ValueError('Invalid preprocessed artifact path.')
        if database_file_sha256(artifact_path) != entry['artifact_sha256']:
            raise ValueError('Preprocessed artifact content changed.')
        with np.load(artifact_path, allow_pickle=False) as prepared:
            image = prepared['image'].copy()
            points = prepared['points'].copy()
        if (
            image.shape != (PROCESS_HEIGHT, PROCESS_WIDTH, 3)
            or image.dtype != np.uint8
            or points.shape != (68, 2)
            or not np.isfinite(points).all()
        ):
            raise ValueError('Invalid preprocessed image or landmark geometry.')
        return image, points
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as error:
        raise PreprocessedDatabaseError(f'{rebuild_message} ({error})') from error


def get_random_original_db_images(gender, count=3, excluded_filenames=None, preprocessed_entries=None):
    """按原规则随机选同性数据库身份，直接加载脚本生成的有效预处理文件。"""
    entries = preprocessed_entries if preprocessed_entries is not None else load_preprocessed_database_manifest()
    all_files = database_source_files(gender)
    original_files = [
        filepath
        for filepath in all_files
        if "_uploaded" not in os.path.basename(filepath)
    ]
    if not original_files:
        if not all_files:
            return []
        original_files = all_files

    excluded_filenames = set(excluded_filenames or ())
    candidates = [
        filepath
        for filepath in original_files
        if os.path.basename(filepath) not in excluded_filenames
        and entries[database_source_key(filepath)].get('valid') is True
    ]
    random.shuffle(candidates)
    selected_images = []
    for selected_file in candidates[:count]:
        img_resized, detected_points = load_prepared_db_image(selected_file, entries)
        selected_images.append((
            img_resized,
            os.path.basename(selected_file),
            detected_points
        ))
    return selected_images


def align_afa_stimulus_group(
    subject_image,
    subject_detected_points,
    database_faces
):
    """以被试和其3张样本脸共同建立AFA/GPA参考形状并完成对齐。"""
    images = [subject_image] + [entry[0] for entry in database_faces]
    landmark_sets = [
        np.asarray(subject_detected_points[:68], dtype=np.float32)
    ] + [
        np.asarray(entry[2][:68], dtype=np.float32)
        for entry in database_faces
    ]
    aligned_records = align_face_collection_afa(images, landmark_sets)

    aligned_subject_image, aligned_subject_detected = aligned_records[0]
    subject_points = build_outer_contour_geometry(
        aligned_subject_image,
        aligned_subject_detected
    )
    if len(subject_points) != OUTER_GEOMETRY_POINT_COUNT:
        raise ValueError("AFA could not prepare the aligned participant face.")

    aligned_database_faces = []
    for aligned_record, database_entry in zip(
        aligned_records[1:],
        database_faces
    ):
        aligned_image, aligned_detected = aligned_record
        database_filename = database_entry[1]
        database_points = build_outer_contour_geometry(
            aligned_image,
            aligned_detected
        )
        if len(database_points) != OUTER_GEOMETRY_POINT_COUNT:
            raise ValueError(
                f"AFA could not prepare database face: {database_filename}"
            )
        aligned_database_faces.append((
            aligned_image,
            database_filename,
            database_points
        ))

    return (
        aligned_subject_image,
        subject_points
    ), aligned_database_faces

def base64_to_cv2(base64_string):
    try:
        if not base64_string or "base64," not in base64_string: return None
        base64_string = base64_string.split("base64,")[1]
        img_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except:
        return None

def cv2_to_base64(img_arr):
    try:
        _, buffer = cv2.imencode('.jpg', img_arr)
        img_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
    except:
        return ""


def save_generated_face(image, participant_id, trial_id, stimulus_type, ratio):
    """按被试保存一张最终呈现的融合刺激图，并返回相对路径。"""
    participant_dir = os.path.abspath(
        os.path.join(GENERATED_FACES_PATH, participant_id)
    )
    generated_root = os.path.abspath(GENERATED_FACES_PATH)
    if os.path.commonpath([generated_root, participant_dir]) != generated_root:
        raise ValueError("Invalid participant output path")

    os.makedirs(participant_dir, exist_ok=True)
    ratio_percent = int(round(float(ratio) * 100))
    filename = (
        f"stim_{int(trial_id):02d}_{stimulus_type}_"
        f"ratio_{ratio_percent:03d}.jpg"
    )
    output_path = os.path.join(participant_dir, filename)
    saved = cv2.imwrite(
        output_path,
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    if not saved:
        raise OSError(f"Could not save generated face: {output_path}")

    return os.path.relpath(output_path, BASE_DIR).replace(os.sep, '/')


def build_morph_stimulus(job):
    """在AFA规范对齐后使用原整图算法生成单张融合刺激图。"""
    ratio = job["ratio"]
    alpha = 1.0 - ratio

    # 保持既定端点规则：0% 直接使用样本，100% 直接使用被试头像。
    if ratio <= 1e-6:
        morphed_full = job["database_image"]
        avg_points = job["database_points"]
    elif ratio >= 1.0 - 1e-6:
        morphed_full = job["subject_image"]
        avg_points = job["subject_points"]
    else:
        morphed_full, avg_points = morph_faces_full(
            job["subject_image"],
            job["database_image"],
            alpha,
            points1=job["subject_points"],
            points2=job["database_points"]
        )

    final_face = crop_face_tight(morphed_full, avg_points)
    generated_image = save_generated_face(
        final_face,
        job["participant_id"],
        job["trial_id"],
        job["stimulus_type"],
        ratio
    )

    result = {
        "id": f"stim_{job['trial_id']}",
        "url": cv2_to_base64(final_face),
        "type": job["stimulus_type"],
        "description": job["description"],
        "source_upload": job["source_upload"],
        "source_db": job["source_db"],
        "generated_image": generated_image
    }
    result[job["ratio_key"]] = ratio
    return result


def cleanup_merge_jobs_locked():
    """清理已结束任务的内存结果；本地生成图片不受影响。调用方必须持锁。"""
    cutoff = time.time() - MERGE_JOB_TTL_SECONDS
    expired_job_ids = [
        job_id
        for job_id, job in MERGE_JOBS.items()
        if job.get("status") in ("completed", "failed", "cancelled")
        and job.get("finished_at", 0) < cutoff
    ]
    for job_id in expired_job_ids:
        job = MERGE_JOBS.pop(job_id)
        request_key = (
            job["participant_id"],
            job["session_id"],
            job["request_fingerprint"]
        )
        if MERGE_JOB_BY_REQUEST.get(request_key) == job_id:
            del MERGE_JOB_BY_REQUEST[request_key]
    for session_id, cancelled_at in list(MERGE_SESSION_CANCELLATIONS.items()):
        if cancelled_at < cutoff:
            del MERGE_SESSION_CANCELLATIONS[session_id]


def get_merge_queue_position_locked(job_id):
    """按实际进入时间动态计算当前排队位置。调用方必须持锁。"""
    queued_jobs = sorted(
        (
            (queued_id, queued_job)
            for queued_id, queued_job in MERGE_JOBS.items()
            if queued_job.get("status") == "queued"
        ),
        key=lambda item: item[1]["created_at"]
    )
    for position, (queued_id, _) in enumerate(queued_jobs, start=1):
        if queued_id == job_id:
            return position
    return None


def prepare_merge_groups(payload):
    """验证两张照片、AFA 对齐，并构建六组互不重叠的融合任务。"""
    participant_id = payload["participant_id"]
    self_gender = payload["self_gender"]
    partner_gender = payload["partner_gender"]
    raw_self = base64_to_cv2(payload.get('self_image'))
    raw_partner = base64_to_cv2(payload.get('partner_image'))
    if raw_self is None or raw_partner is None:
        raise MergePreparationError('Invalid images', 'INVALID_IMAGES', 400)

    img_self_wide = crop_portrait_wide(raw_self)
    img_partner_wide = crop_portrait_wide(raw_partner)
    self_detected_points = get_points(img_self_wide)
    partner_detected_points = get_points(img_partner_wide)
    invalid_roles = []
    if len(self_detected_points) < 68:
        invalid_roles.append('your photograph')
    if len(partner_detected_points) < 68:
        invalid_roles.append("your partner's photograph")
    if invalid_roles:
        raise MergePreparationError(
            'A clear face could not be detected in ' + ' and '.join(invalid_roles)
            + '. Please use a front-facing photograph without glasses, '
              'sunglasses, or face coverings, then try again.',
            'FACE_QUALITY_ERROR', 422
        )

    try:
        occlusion_checks = {
            'self': assess_face_occlusion(img_self_wide, self_detected_points),
            'partner': assess_face_occlusion(img_partner_wide, partner_detected_points)
        }
    except Exception as error:
        raise MergePreparationError(
            'The automated photograph check is temporarily unavailable. '
            'Please ask the researcher to restart the experiment server.',
            'FACE_OCCLUSION_CHECK_UNAVAILABLE', 503
        ) from error
    occlusion_issues = {
        role: issues for role, (_, issues) in occlusion_checks.items() if issues
    }
    if occlusion_issues:
        affected_roles = [
            label for role, label in (
                ('self', 'your photograph'),
                ('partner', "your partner's photograph")
            ) if role in occlusion_issues
        ]
        raise MergePreparationError(
            'The photo requirements were not met for '
            + ' and '.join(affected_roles)
            + '. Please use a clear, well-lit, front-facing photograph '
              'with a clean, uncluttered background. Remove glasses, '
              'sunglasses, and face coverings, and keep a neutral '
              'expression without smiling or making faces.',
            'FACE_OCCLUSION_ERROR', 422, issues=occlusion_issues
        )

    photo_batch_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    self_filename = save_uploaded_image(img_self_wide, photo_batch_id, "self")
    partner_filename = save_uploaded_image(img_partner_wide, photo_batch_id, "partner")
    print(f"[OK] Images saved: {self_filename}, {partner_filename}")

    trial_id = 1
    ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    morph_jobs = []

    # 本人和伴侣各自固定使用 3 个同性数据库面孔；同一面孔覆盖全部 6 个比例。
    # 双方性别相同时，伴侣组排除本人组已经选中的数据库身份。
    preprocessed_entries = load_preprocessed_database_manifest()
    self_db_faces = get_random_original_db_images(
        self_gender, count=3, preprocessed_entries=preprocessed_entries
    )
    partner_excluded_filenames = (
        {db_filename for _, db_filename, _ in self_db_faces}
        if self_gender == partner_gender
        else set()
    )
    partner_db_faces = get_random_original_db_images(
        partner_gender,
        count=3,
        excluded_filenames=partner_excluded_filenames,
        preprocessed_entries=preprocessed_entries
    )
    if len(self_db_faces) < 3 or len(partner_db_faces) < 3:
        raise RuntimeError(
            "At least three valid same-gender database faces are required "
            "for each photograph."
        )

    # 本人与伴侣分别用“上传脸 + 3张数据库脸”建立一次AFA共识形状。
    (
        (img_self_aligned, self_points),
        self_db_faces
    ) = align_afa_stimulus_group(
        img_self_wide,
        self_detected_points,
        self_db_faces
    )
    (
        (img_partner_aligned, partner_points),
        partner_db_faces
    ) = align_afa_stimulus_group(
        img_partner_wide,
        partner_detected_points,
        partner_db_faces
    )

    for db_index, (db_img, db_filename, db_points) in enumerate(
        self_db_faces,
        start=1
    ):
        for ratio in ratios:
            morph_jobs.append({
                "trial_id": trial_id,
                "participant_id": participant_id,
                "subject_image": img_self_aligned,
                "subject_points": self_points,
                "database_image": db_img,
                "database_points": db_points,
                "ratio": ratio,
                "ratio_key": "ratio_self",
                "stimulus_type": "self_morph",
                "description": f"Self Morph {int(ratio*100)}% / Face {db_index}",
                "source_upload": self_filename,
                "source_db": db_filename
            })
            trial_id += 1

    for db_index, (db_img, db_filename, db_points) in enumerate(
        partner_db_faces,
        start=1
    ):
        for ratio in ratios:
            morph_jobs.append({
                "trial_id": trial_id,
                "participant_id": participant_id,
                "subject_image": img_partner_aligned,
                "subject_points": partner_points,
                "database_image": db_img,
                "database_points": db_points,
                "ratio": ratio,
                "ratio_key": "ratio_partner",
                "stimulus_type": "partner_morph",
                "description": f"Partner Morph {int(ratio*100)}% / Face {db_index}",
                "source_upload": partner_filename,
                "source_db": db_filename
            })
            trial_id += 1

    return photo_batch_id, [morph_jobs[i:i + 6] for i in range(0, 36, 6)]


class MergePreparationError(Exception):
    def __init__(self, message, code, http_status, issues=None):
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.issues = issues


def write_merge_worker_error(path, error):
    with open(path, 'w', encoding='utf-8') as output:
        json.dump({
            'error': str(error),
            'code': getattr(error, 'code', 'MERGE_PROCESSING_ERROR'),
            'http_status': getattr(error, 'http_status', 500),
            'issues': getattr(error, 'issues', None)
        }, output)


def prepare_merge_worker(payload, work_dir):
    """独立进程：校验、对齐和分组，主进程可直接终止整个阶段。"""
    try:
        photo_batch_id, groups = prepare_merge_groups(payload)
        for index, group in enumerate(groups):
            with open(os.path.join(work_dir, f'group_{index}.pkl'), 'wb') as output:
                pickle.dump(group, output, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(work_dir, 'prepared.json'), 'w', encoding='utf-8') as output:
            json.dump({'photo_batch_id': photo_batch_id, 'group_count': len(groups)}, output)
    except Exception as error:
        traceback.print_exc()
        write_merge_worker_error(os.path.join(work_dir, 'prepare_error.json'), error)


def morph_group_worker(work_dir, index):
    """独立 CPU 进程：一次只计算一个数据库身份的六个融合比例。"""
    try:
        with open(os.path.join(work_dir, f'group_{index}.pkl'), 'rb') as source:
            group = pickle.load(source)
        result = [build_morph_stimulus(job) for job in group]
        with open(os.path.join(work_dir, f'result_{index}.json'), 'w', encoding='utf-8') as output:
            json.dump(result, output)
    except Exception as error:
        traceback.print_exc()
        write_merge_worker_error(os.path.join(work_dir, f'error_{index}.json'), error)


def finalize_merge_result(participant_id, photo_batch_id, result_images):
    """全部子进程退出后才确认结果、释放计算名额。"""

    self_images = [image for image in result_images if image.get("type") == "self_morph"]
    partner_images = [image for image in result_images if image.get("type") == "partner_morph"]
    if (
        len(result_images) != 36
        or len(self_images) != 18
        or len(partner_images) != 18
        or any(
            not isinstance(image.get("url"), str)
            or not image["url"].startswith("data:image/jpeg;base64,")
            or len(image["url"]) <= len("data:image/jpeg;base64,")
            for image in result_images
        )
    ):
        raise RuntimeError("Face processing did not produce all 36 valid images.")

    random.shuffle(result_images)
    return {
        "status": "completed",
        "participant_id": participant_id,
        "photo_batch_id": photo_batch_id,
        "images": result_images
    }


class MergeJobCancelled(Exception):
    pass


def cancel_merge_job_locked(job_id):
    """排队任务直接取消；运行任务须待子进程终止后才释放名额。"""
    job = MERGE_JOBS.get(job_id)
    if not job or job['status'] not in ('queued', 'processing'):
        return
    job['cancel_requested'] = True
    if job['status'] == 'queued':
        job['status'] = 'cancelled'
        job['finished_at'] = time.time()
        future = job.get('future')
        if future is not None:
            future.cancel()
    request_key = (job['participant_id'], job['session_id'], job['request_fingerprint'])
    if MERGE_JOB_BY_REQUEST.get(request_key) == job_id:
        del MERGE_JOB_BY_REQUEST[request_key]


def expire_merge_leases_locked():
    now = time.time()
    for job_id, job in MERGE_JOBS.items():
        if job['status'] in ('queued', 'processing') and now - job['last_seen'] > MERGE_JOB_LEASE_SECONDS:
            cancel_merge_job_locked(job_id)


def merge_cancel_requested(job_id):
    with MERGE_JOB_LOCK:
        expire_merge_leases_locked()
        job = MERGE_JOBS.get(job_id)
        return job is None or job.get('cancel_requested', False)


def stop_merge_process(process):
    """子进程确认退出前不允许下一名被试开始。"""
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
    while process.is_alive():
        process.kill()
        process.join(timeout=1)
    process.join()
    process.close()


def start_merge_process(process, batch=False):
    """Windows spawn 在导入目标模块前尚未设置进程名，靠继承环境区分轻量批进程。"""
    with MERGE_PROCESS_START_LOCK:
        key = '_EXPERIMENT_MORPH_BATCH_WORKER'
        previous = os.environ.get(key)
        if batch:
            os.environ[key] = '1'
        else:
            os.environ.pop(key, None)
        try:
            process.start()
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def read_merge_worker_result(work_dir, success_filename, error_filename):
    error_path = os.path.join(work_dir, error_filename)
    if os.path.isfile(error_path):
        with open(error_path, encoding='utf-8') as source:
            details = json.load(source)
        raise MergePreparationError(
            details.get('error', 'Face processing failed.'),
            details.get('code', 'MERGE_PROCESSING_ERROR'),
            details.get('http_status', 500),
            details.get('issues')
        )
    success_path = os.path.join(work_dir, success_filename)
    if not os.path.isfile(success_path):
        raise RuntimeError('Face processing worker exited without a result.')
    with open(success_path, encoding='utf-8') as source:
        return json.load(source)


def run_merge_job(job_id, payload):
    """单名被试独占名额；预处理及六组合成都运行于可终止的进程。"""
    with MERGE_JOB_LOCK:
        expire_merge_leases_locked()
        job = MERGE_JOBS.get(job_id)
        if job is None or job['status'] != 'queued' or job.get('cancel_requested'):
            return
        job['status'] = 'processing'
        job['started_at'] = time.time()

    processes = set()
    work_dir = None
    try:
        context = multiprocessing.get_context('spawn')
        # Windows 取消时文件可能仍被工作进程占用；先杀停进程，再二次清理临时目录。
        with tempfile.TemporaryDirectory(
            prefix=f'job_{job_id}_', dir=MERGE_WORK_PATH, ignore_cleanup_errors=True
        ) as work_dir:
            if os.path.commonpath([os.path.abspath(MERGE_WORK_PATH), os.path.abspath(work_dir)]) != os.path.abspath(MERGE_WORK_PATH):
                raise RuntimeError('Unexpected merge work directory.')
            if merge_cancel_requested(job_id):
                raise MergeJobCancelled()

            preparation = context.Process(
                target=prepare_merge_worker, args=(payload, work_dir), name='MorphPrepare', daemon=True
            )
            start_merge_process(preparation)
            processes.add(preparation)
            while preparation.is_alive():
                if merge_cancel_requested(job_id):
                    raise MergeJobCancelled()
                preparation.join(timeout=MERGE_POLL_INTERVAL_SECONDS)
            preparation.join()
            processes.remove(preparation)
            exit_code = preparation.exitcode
            preparation.close()
            if merge_cancel_requested(job_id):
                raise MergeJobCancelled()
            if exit_code != 0 and not os.path.isfile(os.path.join(work_dir, 'prepare_error.json')):
                raise RuntimeError(f'Photo preparation process exited ({exit_code}).')
            prepared = read_merge_worker_result(work_dir, 'prepared.json', 'prepare_error.json')
            if prepared.get('group_count') != 6:
                raise RuntimeError('Photo preparation did not produce six face groups.')

            pending = iter(range(6))
            running = {}
            result_images = []
            while running or pending is not None:
                if merge_cancel_requested(job_id):
                    raise MergeJobCancelled()
                while pending is not None and len(running) < GLOBAL_MORPH_WORKERS:
                    index = next(pending, None)
                    if index is None:
                        pending = None
                        break
                    process = context.Process(
                        target=morph_group_worker,
                        args=(work_dir, index),
                        name=f'MorphBatch-{index}',
                        daemon=True
                    )
                    start_merge_process(process, batch=True)
                    processes.add(process)
                    running[index] = process
                if not running:
                    continue
                next(iter(running.values())).join(timeout=MERGE_POLL_INTERVAL_SECONDS)
                for index, process in list(running.items()):
                    if process.is_alive():
                        continue
                    process.join()
                    processes.remove(process)
                    exit_code = process.exitcode
                    process.close()
                    del running[index]
                    if merge_cancel_requested(job_id):
                        raise MergeJobCancelled()
                    if exit_code != 0 and not os.path.isfile(os.path.join(work_dir, f'error_{index}.json')):
                        raise RuntimeError(f'Face processing worker exited ({exit_code}).')
                    result_images.extend(read_merge_worker_result(
                        work_dir, f'result_{index}.json', f'error_{index}.json'
                    ))
            result = finalize_merge_result(payload['participant_id'], prepared['photo_batch_id'], result_images)

        with MERGE_JOB_LOCK:
            job = MERGE_JOBS.get(job_id)
            if job is not None:
                if job.get('cancel_requested') or time.time() - job['last_seen'] > MERGE_JOB_LEASE_SECONDS:
                    job['status'] = 'cancelled'
                else:
                    job['status'] = 'completed'
                    job['result'] = result
                job['finished_at'] = time.time()
    except MergeJobCancelled:
        with MERGE_JOB_LOCK:
            job = MERGE_JOBS.get(job_id)
            if job is not None:
                job['status'] = 'cancelled'
                job['finished_at'] = time.time()
    except Exception as error:
        traceback.print_exc()
        with MERGE_JOB_LOCK:
            job = MERGE_JOBS.get(job_id)
            if job is not None:
                job['status'] = 'failed'
                job['error'] = str(error)
                job['code'] = getattr(error, 'code', 'MERGE_PROCESSING_ERROR')
                job['http_status'] = getattr(error, 'http_status', 500)
                job['issues'] = getattr(error, 'issues', None)
                job['finished_at'] = time.time()
    finally:
        for process in processes:
            stop_merge_process(process)
        if work_dir and os.path.isdir(work_dir):
            work_root = os.path.abspath(MERGE_WORK_PATH)
            work_target = os.path.abspath(work_dir)
            if os.path.commonpath([work_root, work_target]) == work_root:
                shutil.rmtree(work_target, ignore_errors=True)


def merge_job_response(job_id):
    """创建轻量状态响应；完成时仍直接返回Base64图片数据。"""
    with MERGE_JOB_LOCK:
        cleanup_merge_jobs_locked()
        job = MERGE_JOBS.get(job_id)
        if job is None:
            return None
        status = job["status"]
        if status == "queued":
            return {
                "status": "queued",
                "job_id": job_id,
                "queue_position": get_merge_queue_position_locked(job_id)
            }
        if status == "processing":
            return {"status": "processing", "job_id": job_id}
        if status == "failed":
            return {
                "status": "failed",
                "job_id": job_id,
                "error": job.get("error", "Face processing failed."),
                "code": job.get('code', 'MERGE_PROCESSING_ERROR'),
                "http_status": job.get('http_status', 500),
                "issues": job.get('issues')
            }
        if status == 'cancelled':
            return {'status': 'cancelled', 'job_id': job_id, 'error': 'Face processing was cancelled.'}
        return {"job_id": job_id, **job["result"]}

# --- 核心路由: 图片融合处理 ---
@app.route('/merge_faces', methods=['POST'])
def process_images_experiment():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON request body"}), 400

        participant_id = str(data.get('participant_id') or '').strip()
        if not re.fullmatch(r'^P_[A-Za-z0-9_-]{3,96}$', participant_id):
            return jsonify({"error": "Invalid participant ID"}), 400

        session_id = str(data.get('merge_session_id') or '').strip()
        if not re.fullmatch(r'^[a-f0-9]{32}$', session_id):
            return jsonify({'error': 'Invalid merge session ID'}), 400
        if not isinstance(data.get('self_image'), str) or not isinstance(data.get('partner_image'), str):
            return jsonify({'error': 'Two photographs are required.'}), 400

        self_gender = data.get('self_gender', 'male')
        partner_gender = data.get('partner_gender', 'female')
        if self_gender not in ('male', 'female') or partner_gender not in ('male', 'female'):
            return jsonify({'error': 'Invalid photo gender selection.'}), 400
        request_fingerprint = hashlib.sha256(
            (
                f"{self_gender}|{partner_gender}|"
                f"{data.get('self_image') or ''}|"
                f"{data.get('partner_image') or ''}"
            ).encode('utf-8')
        ).hexdigest()
        payload = {
            "participant_id": participant_id,
            "self_gender": self_gender,
            "partner_gender": partner_gender,
            "self_image": data['self_image'],
            "partner_image": data['partner_image']
        }

        # 后端防重复：同一被试提交相同两张照片时复用原任务；重新上传
        # 不同照片则创建新任务，保证最终使用的是最后确认的照片。
        with MERGE_JOB_LOCK:
            cleanup_merge_jobs_locked()
            expire_merge_leases_locked()
            if session_id in MERGE_SESSION_CANCELLATIONS:
                return jsonify({'error': 'This page session was closed.'}), 410
            request_key = (participant_id, session_id, request_fingerprint)
            existing_job_id = MERGE_JOB_BY_REQUEST.get(request_key)
            existing_job = (
                MERGE_JOBS.get(existing_job_id)
                if existing_job_id
                else None
            )
            if (
                existing_job
                and existing_job.get("status") != "failed"
                and existing_job.get("request_fingerprint") == request_fingerprint
            ):
                job_id = existing_job_id
            else:
                job_id = uuid.uuid4().hex
                MERGE_JOBS[job_id] = {
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "request_fingerprint": request_fingerprint,
                    "status": "queued",
                    "created_at": time.time(),
                    "last_seen": time.time()
                }
                MERGE_JOB_BY_REQUEST[request_key] = job_id
                try:
                    MERGE_JOBS[job_id]['future'] = PARTICIPANT_EXECUTOR.submit(run_merge_job, job_id, payload)
                except Exception:
                    del MERGE_JOBS[job_id]
                    del MERGE_JOB_BY_REQUEST[request_key]
                    raise

        response_body = merge_job_response(job_id)
        if response_body is None:
            return jsonify({"error": "Merge job could not be created."}), 500
        return jsonify(response_body), 202

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/merge_status/<job_id>', methods=['GET'])
def get_merge_status(job_id):
    if not re.fullmatch(r'^[a-f0-9]{32}$', job_id):
        return jsonify({"error": "Invalid merge job ID"}), 400

    session_id = request.headers.get('X-Merge-Session', '')
    with MERGE_JOB_LOCK:
        job = MERGE_JOBS.get(job_id)
        if job is None or job.get('session_id') != session_id:
            return jsonify({'error': 'Merge job not found.'}), 404
        expire_merge_leases_locked()
        if job['status'] in ('queued', 'processing') and not job.get('cancel_requested'):
            job['last_seen'] = time.time()

    response_body = merge_job_response(job_id)
    if response_body is None:
        return jsonify({"error": "Merge job not found or expired"}), 404
    response = jsonify(response_body)
    response.headers['Cache-Control'] = 'no-store'
    if response_body["status"] == "failed":
        return response, response_body.get('http_status', 500)
    if response_body['status'] == 'cancelled':
        return response, 410
    return response


@app.route('/merge_cancel', methods=['POST'])
def cancel_merge_session():
    """关闭页面的尽力通知；失联租约是该通知未送达时的兜底。"""
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid cancellation request.'}), 400
    session_id = str(data.get('merge_session_id') or '').strip()
    if not re.fullmatch(r'^[a-f0-9]{32}$', session_id):
        return jsonify({'error': 'Invalid merge session ID.'}), 400
    with MERGE_JOB_LOCK:
        MERGE_SESSION_CANCELLATIONS[session_id] = time.time()
        for job_id, job in MERGE_JOBS.items():
            if job['session_id'] == session_id:
                cancel_merge_job_locked(job_id)
    return jsonify({'status': 'cancellation_requested'})

# --- 删除被试照片 ---
@app.route('/delete_participant_faces', methods=['POST'])
def delete_participant_faces():
    try:
        data = request.get_json(silent=True) or {}
        photo_batch_id = data.get('photo_batch_id', '')

        if not re.fullmatch(r'^\d{8}_\d{6}_\d{6}$', photo_batch_id):
            return jsonify({"error": "Invalid photo batch ID"}), 400

        photo_root = os.path.abspath(PARTICIPANT_FACES_PATH)
        target_folder = os.path.abspath(
            os.path.join(PARTICIPANT_FACES_PATH, photo_batch_id)
        )

        if os.path.commonpath([photo_root, target_folder]) != photo_root:
            return jsonify({"error": "Invalid deletion path"}), 400

        if os.path.isdir(target_folder):
            shutil.rmtree(target_folder)

        return jsonify({
            "status": "success",
            "message": "Participant photographs deleted"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _sona_xml_element(root, local_name):
    """忽略 XML 命名空间查找 SONA 响应字段。"""
    for element in root.iter():
        if element.tag.rsplit('}', 1)[-1] == local_name:
            return element
    return None


def _parse_sona_credit_response(response_text):
    """解析 WebstudyCredit XML，并将重复授予视为幂等成功。"""
    normalized_text = (response_text or '').strip()
    lower_text = normalized_text.lower()
    already_granted_phrases = (
        'already received credit',
        'already participated',
        'credit has already been granted'
    )
    if any(phrase in lower_text for phrase in already_granted_phrases):
        return {
            'success': True,
            'status': 'already_granted',
            'message': 'SONA credit had already been granted.'
        }

    try:
        root = ET.fromstring(normalized_text)
    except ET.ParseError:
        return {
            'success': False,
            'status': 'invalid_response',
            'message': 'SONA returned an invalid XML response.'
        }

    result_element = _sona_xml_element(root, 'Result')
    credit_status_element = _sona_xml_element(root, 'credit_status')
    credit_status = (
        (credit_status_element.text or '').strip()
        if credit_status_element is not None
        else ''
    )
    if result_element is not None and (credit_status == 'G' or not credit_status):
        return {
            'success': True,
            'status': 'granted',
            'credit_status': credit_status or 'G',
            'message': 'SONA credit was granted successfully.'
        }

    error_messages = []
    errors_element = _sona_xml_element(root, 'Errors')
    if errors_element is not None:
        for element in errors_element.iter():
            text = (element.text or '').strip()
            if text and text not in error_messages:
                error_messages.append(text)

    return {
        'success': False,
        'status': 'rejected',
        'message': '; '.join(error_messages) or 'SONA did not confirm that credit was granted.'
    }


def grant_sona_credit(survey_code):
    """通过 SONA Server-Side Completion URL 授予参与 credit。"""
    survey_code = str(survey_code or '').strip()
    if not survey_code:
        return {
            'success': False,
            'status': 'missing_survey_code',
            'message': 'The SONA survey code is missing from the experiment URL.'
        }
    if not re.fullmatch(r'^[A-Za-z0-9._-]{1,128}$', survey_code):
        return {
            'success': False,
            'status': 'invalid_survey_code',
            'message': 'The SONA survey code has an invalid format.'
        }
    if not SONA_CREDIT_TOKEN:
        return {
            'success': False,
            'status': 'configuration_error',
            'message': 'The backend SONA_CREDIT_TOKEN environment variable is not configured.'
        }

    query = urllib.parse.urlencode({
        'experiment_id': SONA_EXPERIMENT_ID,
        'credit_token': SONA_CREDIT_TOKEN,
        'survey_code': survey_code
    })
    completion_url = f"{SONA_COMPLETION_API_URL}?{query}"
    sona_request = urllib.request.Request(
        completion_url,
        method='GET',
        headers={
            'Accept': 'application/xml, text/xml',
            'User-Agent': 'Bristol-Face-Experiment/1.0'
        }
    )

    try:
        with urllib.request.urlopen(
            sona_request,
            timeout=SONA_REQUEST_TIMEOUT_SECONDS
        ) as response:
            response_text = response.read().decode('utf-8', errors='replace')
        return _parse_sona_credit_response(response_text)
    except urllib.error.HTTPError as error:
        response_text = error.read().decode('utf-8', errors='replace')
        parsed_response = _parse_sona_credit_response(response_text)
        if parsed_response.get('success'):
            return parsed_response
        parsed_response['message'] = (
            f"SONA returned HTTP {error.code}: {parsed_response.get('message')}"
        )
        return parsed_response
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {
            'success': False,
            'status': 'connection_error',
            'message': f"Could not contact SONA: {error}"
        }


def _write_json_atomically(json_filepath, data):
    """在同一目录写临时文件后原子替换正式 JSON。调用方负责加锁。"""
    json_filename = os.path.basename(json_filepath)
    json_temp_path = os.path.join(
        DATA_SAVE_PATH,
        f".{json_filename}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with open(json_temp_path, 'w', encoding='utf-8') as temp_file:
            json.dump(data, temp_file, ensure_ascii=False, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(json_temp_path, json_filepath)
    finally:
        if os.path.exists(json_temp_path):
            os.remove(json_temp_path)


def _build_experiment_json_file_info(data, is_complete):
    """按稳定被试编号为一次实验生成唯一文件名，自动/最终保存共用。"""
    raw_sona_id = str(data.get('sona_id') or 'sona').strip() or 'sona'
    safe_sona_id = re.sub(r'[^A-Za-z0-9._-]+', '_', raw_sona_id)
    safe_sona_id = (safe_sona_id.strip('._-')[:128] or 'sona')

    start_timestamp = str(data.get('file_start_timestamp') or '').strip()
    if not re.fullmatch(r'^\d{12}$', start_timestamp):
        start_timestamp = datetime.now().strftime('%Y%m%d%H%M')
    data['file_start_timestamp'] = start_timestamp

    end_timestamp = str(data.get('file_end_timestamp') or '').strip()
    if is_complete and not re.fullmatch(r'^\d{12}$', end_timestamp):
        end_timestamp = datetime.now().strftime('%Y%m%d%H%M')
    data['file_end_timestamp'] = end_timestamp if is_complete else ''

    return f"sona_{safe_sona_id}_{start_timestamp}_{data['participant_id']}.json"


def _read_json_record(json_filepath):
    if not os.path.isfile(json_filepath):
        return None
    try:
        with open(json_filepath, 'r', encoding='utf-8') as existing_file:
            return json.load(existing_file)
    except (OSError, ValueError):
        return None


def _sona_state_locked(record, json_filepath):
    """从落盘记录和本进程中的授权任务判断当前状态；调用方必须持锁。"""
    if not isinstance(record, dict):
        return 'not_found'
    if record.get('is_complete') is not True:
        return 'partial'
    completion = record.get('sona_completion')
    if isinstance(completion, dict) and completion.get('success') is True:
        return 'success'
    if isinstance(completion, dict) and completion.get('status') == 'pending':
        return 'pending' if json_filepath in SONA_ACTIVE_ATTEMPTS else 'unknown'
    return 'failed' if completion else 'unknown'


def _save_status_body_locked(record, json_filepath):
    """仅公开保存/授权状态，不返回问卷答案或 SONA 令牌。"""
    state = _sona_state_locked(record, json_filepath)
    body = {'status': state, 'json_saved': state != 'not_found'}
    if isinstance(record, dict):
        body['json_filename'] = os.path.basename(json_filepath)
        completion = record.get('sona_completion')
        if isinstance(completion, dict) and state in ('success', 'failed'):
            body['sona_completion'] = completion
    return body


def _grant_sona_credit_in_background(json_filepath, survey_code, attempt_id):
    """SONA 请求完成后，只更新已保存的同一份最终 JSON。"""
    try:
        try:
            # 同一 SONA 编号即使在两个标签页形成不同 respond_id，也不并发请求 SONA。
            credit_key = str(survey_code or '').strip() or json_filepath
            with DATA_SAVE_LOCK:
                credit_lock = SONA_CREDIT_LOCKS.setdefault(
                    credit_key, threading.Lock()
                )
            with credit_lock:
                completion = grant_sona_credit(survey_code)
        except Exception as error:
            traceback.print_exc()
            completion = {
                'success': False,
                'status': 'connection_error',
                'message': f'SONA confirmation failed: {error}'
            }
        completion['attempted_at'] = datetime.now().isoformat()

        with DATA_SAVE_LOCK:
            current = _read_json_record(json_filepath)
            pending = (
                current.get('sona_completion')
                if isinstance(current, dict)
                else None
            )
            if (
                isinstance(pending, dict)
                and pending.get('status') == 'pending'
                and pending.get('attempt_id') == attempt_id
            ):
                current['sona_completion'] = completion
                _write_json_atomically(json_filepath, current)
    except Exception:
        traceback.print_exc()
        # 如写回失败，磁盘中的 pending 会在下一次状态查询时显示为 unknown。
    finally:
        with DATA_SAVE_LOCK:
            SONA_ACTIVE_ATTEMPTS.discard(json_filepath)


@app.route('/save_status', methods=['POST'])
def get_save_status():
    identity = request.get_json(silent=True)
    if not isinstance(identity, dict):
        return jsonify({'error': 'Invalid JSON request body'}), 400

    participant_id = str(identity.get('participant_id') or '').strip()
    start_timestamp = str(identity.get('file_start_timestamp') or '').strip()
    if (
        not re.fullmatch(r'^P_[A-Za-z0-9_-]{3,96}$', participant_id)
        or not re.fullmatch(r'^\d{12}$', start_timestamp)
    ):
        return jsonify({'error': 'Invalid save identity'}), 400

    identity['participant_id'] = participant_id
    json_filename = _build_experiment_json_file_info(identity, False)
    json_filepath = os.path.join(DATA_SAVE_PATH, json_filename)
    with DATA_SAVE_LOCK:
        body = _save_status_body_locked(
            _read_json_record(json_filepath), json_filepath
        )

    response = jsonify(body)
    response.headers['Cache-Control'] = 'no-store'
    return response

# --- 核心路由: 数据保存 ---
@app.route('/save_data', methods=['POST'])
def save_data():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON request body"}), 400

        # 如果是第一次请求，生成新的 ID
        participant_id = data.get('participant_id')
        if not participant_id:
            timestamp = int(time.time())
            participant_id = f"P_{timestamp}_{random.randint(1000,9999)}"
        if not re.fullmatch(r'^P_[A-Za-z0-9_-]{3,96}$', str(participant_id)):
            return jsonify({"error": "Invalid participant ID"}), 400
        
        data['participant_id'] = participant_id
        try:
            save_revision = max(0, int(data.get('save_revision', 0)))
        except (TypeError, ValueError):
            save_revision = 0
        data['save_revision'] = save_revision
        is_complete = data.get('is_complete', False)
        
        # 一次实验只用一个JSON文件；最终保存先落盘，SONA 在后台确认。
        json_filename = _build_experiment_json_file_info(data, is_complete)
        json_filepath = os.path.join(DATA_SAVE_PATH, json_filename)
        with DATA_SAVE_LOCK:
            existing_data = _read_json_record(json_filepath)

            # 已经完成的记录不能被稍后到达的旧自动保存请求降级覆盖。
            try:
                existing_revision = int(
                    existing_data.get('save_revision', 0)
                    if isinstance(existing_data, dict)
                    else 0
                )
            except (TypeError, ValueError):
                existing_revision = 0

            if not is_complete:
                preserve_existing = (
                    isinstance(existing_data, dict)
                    and (
                        existing_data.get('is_complete') is True
                        or existing_revision > save_revision
                    )
                )
                if not preserve_existing:
                    _write_json_atomically(json_filepath, data)
                return jsonify({
                    'status': 'success',
                    'participant_id': participant_id,
                    'save_revision': save_revision,
                    'json_filename': json_filename,
                    'json_saved': True
                })

            if isinstance(existing_data, dict) and existing_data.get('is_complete') is True:
                # 重试不改写之前已保存的正式答案；成功授权直接复用落盘结果。
                if str(existing_data.get('sona_id', '')).strip() != str(data.get('sona_id', '')).strip():
                    return jsonify({'error': 'SONA identity changed for this experiment.'}), 409
                state = _sona_state_locked(existing_data, json_filepath)
                if state in ('success', 'pending'):
                    return jsonify(_save_status_body_locked(existing_data, json_filepath)), (
                        200 if state == 'success' else 202
                    )
                # 失败或后台进程中断后，只接受先查状态再明确发起的重试。
                if data.get('retry_sona_credit') is not True:
                    return jsonify(_save_status_body_locked(existing_data, json_filepath)), 200
                data = existing_data
            elif existing_revision > save_revision:
                return jsonify({'error': 'A newer experiment revision is already saved.'}), 409

            attempt_id = uuid.uuid4().hex
            data['sona_completion'] = {
                'success': False,
                'status': 'pending',
                'attempt_id': attempt_id,
                'message': 'SONA credit confirmation is in progress.'
            }
            if json_filepath not in SONA_ACTIVE_ATTEMPTS:
                _write_json_atomically(json_filepath, data)
                SONA_ACTIVE_ATTEMPTS.add(json_filepath)
                try:
                    SONA_EXECUTOR.submit(
                        _grant_sona_credit_in_background,
                        json_filepath,
                        data.get('sona_id'),
                        attempt_id
                    )
                except Exception:
                    SONA_ACTIVE_ATTEMPTS.discard(json_filepath)
                    raise

            return jsonify(_save_status_body_locked(data, json_filepath)), 202
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)
