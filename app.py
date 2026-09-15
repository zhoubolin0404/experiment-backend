from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import os
import sys
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
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import pandas as pd  # 数据处理库

app = Flask(__name__)

# 允许跨域和特殊请求头 (ngrok-skip-browser-warning 是为了穿透 Ngrok 的拦截页)
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers=["Content-Type", "Authorization", "ngrok-skip-browser-warning"])

# --- 全局配置变量 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(BASE_DIR, 'shape_predictor_68_face_landmarks.dat')
DATABASE_PATH = os.path.join(BASE_DIR, 'database')
PARTICIPANT_FACES_PATH = os.path.join(BASE_DIR, 'participant_faces')
DATA_SAVE_PATH = os.path.join(BASE_DIR, 'data')
# 图片处理尺寸配置
PROCESS_WIDTH = 400
PROCESS_HEIGHT = 533
OUTPUT_WIDTH = 400
OUTPUT_HEIGHT = 533
MAX_MORPH_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))
DATA_SAVE_LOCK = threading.Lock()
EXCEL_SAVE_LOCK = threading.Lock()

# 以 68 点中的眉毛、下颌轮廓为基准，供检测失败时的矩形裁剪回退使用。
FACE_CROP_SIDE_MARGIN = 0.04
FACE_CROP_FOREHEAD_MARGIN = 0.42
FACE_CROP_CHIN_MARGIN = 0.05
FACE_BLEND_FOREHEAD_MARGIN = 0.20
FACE_HAIRLINE_MAX_MARGIN = 0.60
FACE_HAIRLINE_MIN_MARGIN = 0.04
FACE_MASK_INNER_FEATHER_RATIO = 0.06
FACE_CANVAS_VALUE = 255
# 被试照片通常来自手机，锐度明显高于样本库证件照。只在二者差异足够
# 大时做轻度匹配，并限制最大模糊，避免损失眼鼻口的身份信息。
SUBJECT_BLUR_TRIGGER_RATIO = 1.12
SUBJECT_BLUR_MIN_SIGMA = 0.45
SUBJECT_BLUR_MAX_SIGMA = 1.55
SUBJECT_BLUR_START_RATIO = 0.35
SUBJECT_BLUR_FULL_RATIO = 0.60
FACE_FOREHEAD_POINT_START = 68
FACE_FOREHEAD_POINT_COUNT = 17
FACE_FOREHEAD_INTERIOR_ROWS = 2
FACE_FOREHEAD_MESH_POINT_COUNT = (
    FACE_FOREHEAD_POINT_COUNT * (1 + FACE_FOREHEAD_INTERIOR_ROWS)
)
FACE_BOUNDARY_POINT_START = (
    FACE_FOREHEAD_POINT_START + FACE_FOREHEAD_MESH_POINT_COUNT
)
FACE_ALIGN_LEFT_EYE = (0.34, 0.40)
FACE_ALIGN_RIGHT_EYE = (0.66, 0.40)
FACE_ALIGN_MOUTH = (0.50, 0.69)

# 外层已经并行处理不同刺激图，限制 OpenCV 内部线程以避免笔记本过度抢占。
cv2.setNumThreads(1)

# 确保数据及图片文件夹存在
os.makedirs(DATA_SAVE_PATH, exist_ok=True)
os.makedirs(PARTICIPANT_FACES_PATH, exist_ok=True)
os.makedirs(os.path.join(DATABASE_PATH, 'male'), exist_ok=True)
os.makedirs(os.path.join(DATABASE_PATH, 'female'), exist_ok=True)

# --- 初始化 Dlib 模型 ---
detector = None
predictor = None
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    print("[OK] Dlib models loaded successfully.")
except Exception as e:
    print(f"[ERROR] Dlib model could not be loaded from '{PREDICTOR_PATH}'.")
    print(e)

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

def morph_triangle(img_src, img_dst, t_src, t_dst):
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
        res = morph_triangle(img, warped_img, t_src, t_dst)
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


def warp_face_mask(mask, src_points, dst_points, triangles):
    """使用与照片完全相同的三角网格形变可见面部掩膜。"""
    if mask is None:
        return None
    mask_three_channel = np.repeat(mask[:, :, None], 3, axis=2)
    warped = warp_image(
        mask_three_channel,
        src_points,
        dst_points,
        triangles
    )
    return warped[:, :, 0]


def build_face_contour_mask(points, image_shape, forehead_points=None):
    """用下颌线和连续额头曲线构造无孔洞的可见面部区域。"""
    h_img, w_img = image_shape[:2]
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    if points is None or len(points) < 68:
        return mask

    face_pts = np.asarray(points[:68], dtype=np.float32)
    jaw = face_pts[0:17]
    brows = face_pts[17:27]
    jaw_left = float(np.min(jaw[:, 0]))
    jaw_right = float(np.max(jaw[:, 0]))
    brow_top = float(np.min(brows[:, 1]))
    chin_y = float(np.max(jaw[:, 1]))
    face_height = max(1.0, chin_y - brow_top)

    if forehead_points is None:
        # 仅作为检测失败时的保守回退；不会向上猜到通常的发际线位置。
        x_fractions = np.linspace(
            0.05,
            0.95,
            FACE_FOREHEAD_POINT_COUNT,
            dtype=np.float32
        )
        arc_offsets = (
            0.12 * np.abs(np.linspace(
                -1.0,
                1.0,
                FACE_FOREHEAD_POINT_COUNT,
                dtype=np.float32
            )) ** 1.7
        )
        forehead_top = brow_top - FACE_BLEND_FOREHEAD_MARGIN * face_height
        forehead_points = np.column_stack([
            jaw_left + x_fractions * face_width,
            forehead_top + arc_offsets * face_height
        ])
    else:
        forehead_points = np.asarray(forehead_points, dtype=np.float32)

    # 下颌点按左到右排列，额头点反向闭合；fillPoly 可保留刘海造成的凹形边界。
    contour = np.vstack([jaw, forehead_points[::-1]])
    contour[:, 0] = np.clip(contour[:, 0], 0, w_img - 1)
    contour[:, 1] = np.clip(contour[:, 1], 0, h_img - 1)
    cv2.fillPoly(
        mask,
        [np.int32(np.round(contour))],
        255,
        lineType=cv2.LINE_AA
    )
    return mask


def estimate_visible_forehead_points(image, points):
    """学习当前人脸肤色，并从眉上皮肤向上追踪实际可见发际线。"""
    if image is None or points is None or len(points) < 68:
        return None

    face_pts = np.asarray(points[:68], dtype=np.float32)
    jaw = face_pts[0:17]
    brows = face_pts[17:27]
    jaw_left = float(np.min(jaw[:, 0]))
    jaw_right = float(np.max(jaw[:, 0]))
    brow_top = float(np.min(brows[:, 1]))
    chin_y = float(np.max(jaw[:, 1]))
    face_width = max(1.0, jaw_right - jaw_left)
    face_height = max(1.0, chin_y - brow_top)
    h_img, w_img = image.shape[:2]

    x_fractions = np.linspace(
        0.05,
        0.95,
        FACE_FOREHEAD_POINT_COUNT,
        dtype=np.float32
    )
    sample_x = jaw_left + x_fractions * face_width
    min_y = max(0, int(round(
        brow_top - FACE_HAIRLINE_MAX_MARGIN * face_height
    )))
    max_y = min(h_img - 1, int(round(brow_top + 0.12 * face_height)))
    minimum_visible_y = brow_top - FACE_HAIRLINE_MIN_MARGIN * face_height
    fallback_top = brow_top - FACE_BLEND_FOREHEAD_MARGIN * face_height
    normalized_x = np.linspace(
        -1.0,
        1.0,
        FACE_FOREHEAD_POINT_COUNT,
        dtype=np.float32
    )
    fallback_offsets = 0.12 * np.abs(normalized_x) ** 1.7 * face_height
    detected_y = fallback_top + fallback_offsets

    try:
        # 样本全部来自当前照片的两颊和鼻梁；肤色深浅、人种与白平衡不会
        # 使用全局固定阈值。仅在此处判断上边界，不对面孔内部逐像素挖洞。
        sample_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        seed_radius = max(4, int(round(face_width * 0.035)))
        seed_centers = [
            0.58 * face_pts[31] + 0.42 * face_pts[3],
            0.58 * face_pts[35] + 0.42 * face_pts[13],
            face_pts[27] + np.array([0.0, 0.05 * face_height], dtype=np.float32),
            face_pts[29],
            0.55 * face_pts[41] + 0.45 * face_pts[31],
            0.55 * face_pts[46] + 0.45 * face_pts[35]
        ]
        for center in seed_centers:
            cv2.circle(
                sample_mask,
                tuple(np.int32(np.round(center))),
                seed_radius,
                255,
                -1,
                lineType=cv2.LINE_AA
            )

        smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)
        lab_image = cv2.cvtColor(
            smoothed_image,
            cv2.COLOR_BGR2LAB
        ).astype(np.float32)
        samples = lab_image[sample_mask > 0]
        if samples.size == 0:
            return np.column_stack([sample_x, detected_y]).astype(np.float32)

        skin_center = np.median(samples, axis=0)
        absolute_deviation = np.abs(samples - skin_center)
        skin_scale = np.percentile(absolute_deviation, 80, axis=0) * 1.8
        skin_scale = np.clip(
            skin_scale,
            np.array([12.0, 5.0, 5.0], dtype=np.float32),
            np.array([30.0, 14.0, 14.0], dtype=np.float32)
        )
        normalized_delta = (lab_image - skin_center) / skin_scale
        # 亮度参与但权重略低，避免额头受光与两颊阴影造成误切。
        skin_distance = np.sqrt(
            0.70 * normalized_delta[:, :, 0] ** 2 +
            normalized_delta[:, :, 1] ** 2 +
            normalized_delta[:, :, 2] ** 2
        )
        skin_region = np.where(skin_distance <= 2.65, 255, 0).astype(np.uint8)

        candidate_top = brow_top - FACE_HAIRLINE_MAX_MARGIN * face_height
        candidate_offsets = (
            0.18 * np.abs(normalized_x) ** 1.7 * face_height
        )
        candidate_forehead = np.column_stack([
            sample_x,
            candidate_top + candidate_offsets
        ])
        candidate_mask = build_face_contour_mask(
            points,
            image.shape,
            candidate_forehead
        )
        skin_region = cv2.bitwise_and(skin_region, candidate_mask)
        skin_region[sample_mask > 0] = 255
        skin_region = cv2.morphologyEx(
            skin_region,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=2
        )
        skin_region = cv2.morphologyEx(
            skin_region,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1
        )

        count, labels, _, _ = cv2.connectedComponentsWithStats(
            (skin_region > 0).astype(np.uint8),
            connectivity=8
        )
        best_label = 0
        best_overlap = 0
        for label in range(1, count):
            overlap = int(np.count_nonzero(
                (labels == label) & (sample_mask > 0)
            ))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        if best_label == 0:
            return np.column_stack([sample_x, detected_y]).astype(np.float32)
        visible_skin = labels == best_label

        window_radius = max(2, int(round(face_width * 0.018)))
        required_connection_y = int(round(
            brow_top - 0.02 * face_height
        ))
        for index, x_value in enumerate(sample_x):
            x_center = int(round(x_value))
            x1 = max(0, x_center - window_radius)
            x2 = min(w_img, x_center + window_radius + 1)
            if x1 >= x2 or min_y >= max_y:
                continue

            profile = np.mean(
                visible_skin[min_y:max_y + 1, x1:x2],
                axis=1
            ) >= 0.32
            profile = cv2.morphologyEx(
                (profile.astype(np.uint8) * 255)[:, None],
                cv2.MORPH_CLOSE,
                np.ones((13, 1), dtype=np.uint8)
            )[:, 0] > 0

            padded = np.pad(profile.astype(np.int8), (1, 1))
            changes = np.diff(padded)
            run_starts = np.flatnonzero(changes == 1)
            run_ends = np.flatnonzero(changes == -1) - 1
            connected_runs = [
                (start, end)
                for start, end in zip(run_starts, run_ends)
                if min_y + end >= required_connection_y
            ]
            if connected_runs:
                start, _ = max(connected_runs, key=lambda run: run[1])
                # 在分类边界内再收缩少量，避免抗锯齿发丝进入皮肤三角形。
                detected_y[index] = float(
                    min_y + start + max(2, int(round(0.012 * face_height)))
                )

        # 先做短范围平滑，再拟合低阶曲线。单根发丝、反光或局部阴影
        # 不应成为 Delaunay 控制点，否则会被放大成截图中的三角楔形。
        smoothing_kernel = np.array(
            [1.0, 2.0, 3.0, 2.0, 1.0],
            dtype=np.float32
        ) / 9.0
        smoothed_y = np.convolve(
            np.pad(detected_y, (2, 2), mode='edge'),
            smoothing_kernel,
            mode='valid'
        )
        central_mask = np.abs(normalized_x) <= 0.46
        fit_mask = central_mask.copy()
        fitted_y = smoothed_y.copy()
        for _ in range(3):
            if np.count_nonzero(fit_mask) < 5:
                break
            coefficients = np.polyfit(
                normalized_x[fit_mask],
                smoothed_y[fit_mask],
                deg=2
            )
            fitted_y = np.polyval(coefficients, normalized_x)
            residual = smoothed_y - fitted_y
            median_residual = float(np.median(residual[fit_mask]))
            mad = float(np.median(np.abs(
                residual[fit_mask] - median_residual
            )))
            tolerance = max(0.035 * face_height, 2.8 * mad)
            next_fit_mask = central_mask & (
                np.abs(residual - median_residual) <= tolerance
            )
            if np.array_equal(next_fit_mask, fit_mask):
                break
            fit_mask = next_fit_mask

        # 向脸内移动至大多数观测点下方；宁可少融几像素额头，也不混入头发。
        central_residual = (smoothed_y - fitted_y)[central_mask]
        inward_shift = max(
            0.0,
            float(np.quantile(central_residual, 0.85))
        ) + 0.012 * face_height
        fitted_y = fitted_y + inward_shift

        # 中央区域确定额头高度，外侧平滑下降到太阳穴，避免两侧短发
        # 把整个额头曲线拖到眉毛上方。
        temple_blend = np.clip(
            (np.abs(normalized_x) - 0.46) / 0.54,
            0.0,
            1.0
        )
        temple_blend = (
            temple_blend * temple_blend * (3.0 - 2.0 * temple_blend)
        )
        fitted_y = (
            fitted_y * (1.0 - temple_blend) +
            minimum_visible_y * temple_blend
        )
        detected_y = np.clip(
            fitted_y,
            min_y,
            minimum_visible_y
        )
    except Exception as e:
        print(f"Visible forehead estimation fallback: {e}")

    return np.column_stack([sample_x, detected_y]).astype(np.float32)


def create_visible_face_mask(image, points, forehead_points=None):
    """为单张照片生成实心可见面部掩膜，不把该照片的头发当成额头。"""
    if forehead_points is None:
        forehead_points = estimate_visible_forehead_points(image, points)
    mask = build_face_contour_mask(points, image.shape, forehead_points)
    # 向内收一小圈，边缘稍后用距离变换在脸内完成渐变。
    return cv2.erode(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )


def build_forehead_mesh_points(points, hairline_points):
    """在发际线与眉毛之间补两排控制点，避免大三角形造成额头色块。"""
    face_pts = np.asarray(points[:68], dtype=np.float32)
    hairline = np.asarray(hairline_points, dtype=np.float32)
    brows = face_pts[17:27]
    jaw = face_pts[0:17]
    brow_top = float(np.min(brows[:, 1]))
    chin_y = float(np.max(jaw[:, 1]))
    face_height = max(1.0, chin_y - brow_top)

    brow_order = np.argsort(brows[:, 0])
    brow_x = brows[brow_order, 0]
    brow_y = brows[brow_order, 1]
    lower_y = np.interp(
        hairline[:, 0],
        brow_x,
        brow_y,
        left=float(brow_y[0]),
        right=float(brow_y[-1])
    ).astype(np.float32) - 0.035 * face_height
    # 太阳穴处发际线可能接近眉毛，仍需保持最小网格高度以避免重复点。
    lower_y = np.maximum(
        lower_y,
        hairline[:, 1] + 0.055 * face_height
    )
    lower_forehead = np.column_stack([hairline[:, 0], lower_y])

    mesh_rows = [hairline]
    for row_index in range(1, FACE_FOREHEAD_INTERIOR_ROWS + 1):
        fraction = row_index / float(FACE_FOREHEAD_INTERIOR_ROWS + 1)
        mesh_rows.append(
            (1.0 - fraction) * hairline + fraction * lower_forehead
        )
    return np.vstack(mesh_rows).astype(np.float32)


def prepare_face_geometry(image, points):
    """将实际额头边界加入 68 点网格，并返回与其一致的可见面部掩膜。"""
    if image is None or points is None or len(points) < 68:
        return np.array([]), None

    forehead_points = estimate_visible_forehead_points(image, points)
    if (
        forehead_points is None or
        len(forehead_points) != FACE_FOREHEAD_POINT_COUNT
    ):
        return np.array([]), None

    h_img, w_img = image.shape[:2]
    x_max = w_img - 1
    y_max = h_img - 1
    boundary_points = np.float32([
        [0, 0], [x_max // 2, 0], [x_max, 0],
        [x_max, y_max // 2], [x_max, y_max], [x_max // 2, y_max],
        [0, y_max], [0, y_max // 2]
    ])
    forehead_mesh_points = build_forehead_mesh_points(
        points,
        forehead_points
    )
    geometry_points = np.vstack([
        np.asarray(points[:68], dtype=np.float32),
        forehead_mesh_points,
        boundary_points
    ])
    visible_mask = create_visible_face_mask(
        image,
        geometry_points,
        forehead_points=forehead_points
    )
    return geometry_points, visible_mask


def database_face_quality_issue(image, face_mask):
    """返回严重曝光/色偏问题；不以天然肤色深浅作为排除条件。"""
    if image is None or face_mask is None:
        return "missing image or face mask"
    pixels_mask = face_mask > 127
    if np.count_nonzero(pixels_mask) < 512:
        return "visible face area is too small"

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[:, :, 0][pixels_mask]
    chroma = np.sqrt(
        (lab[:, :, 1][pixels_mask] - 128.0) ** 2 +
        (lab[:, :, 2][pixels_mask] - 128.0) ** 2
    )
    low, median, high = np.percentile(lightness, [5, 50, 95])
    black_clip = float(np.mean(lightness <= 10.0))
    white_clip = float(np.mean(lightness >= 248.0))

    if median < 48.0 and black_clip > 0.10:
        return "severely underexposed face"
    if median > 232.0 and white_clip > 0.12:
        return "severely overexposed face"
    if high - low > 150.0 and low < 24.0:
        return "extreme uneven facial lighting"
    if np.median(chroma) > 62.0 and np.percentile(chroma, 90) > 78.0:
        return "excessive facial colour cast"
    return None


def blend_face_textures_lab(subject_image, database_image, subject_ratio):
    """在感知均匀的 LAB 空间混合纹理，降低跨肤色中间比例的灰脏感。"""
    ratio = float(np.clip(subject_ratio, 0.0, 1.0))
    subject_lab = cv2.cvtColor(subject_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    database_lab = cv2.cvtColor(database_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    blended_lab = (
        ratio * subject_lab +
        (1.0 - ratio) * database_lab
    )
    return cv2.cvtColor(
        np.clip(blended_lab, 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2BGR
    )


def measure_face_detail(image, face_mask):
    """测量面部高频细节，忽略背景和面缘强对比造成的虚假锐度。"""
    if image is None:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if face_mask is None or face_mask.shape[:2] != gray.shape:
        valid_mask = np.zeros(gray.shape, dtype=np.uint8)
        margin_y = max(1, int(round(gray.shape[0] * 0.20)))
        margin_x = max(1, int(round(gray.shape[1] * 0.20)))
        valid_mask[
            margin_y:gray.shape[0] - margin_y,
            margin_x:gray.shape[1] - margin_x
        ] = 255
    else:
        valid_mask = np.where(face_mask > 127, 255, 0).astype(np.uint8)
        valid_mask = cv2.erode(
            valid_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1
        )

    detail = np.abs(
        gray - cv2.GaussianBlur(
            gray,
            (0, 0),
            sigmaX=1.0,
            sigmaY=1.0,
            borderType=cv2.BORDER_REFLECT_101
        )
    )
    samples = detail[valid_mask > 0]
    if samples.size < 128:
        return 0.0

    # 裁掉极少量眼睑、鼻孔等强边缘离群值，使分数主要反映照片纹理。
    ceiling = float(np.percentile(samples, 95))
    if ceiling <= 1e-6:
        return 0.0
    clipped = np.minimum(samples, ceiling)
    return float(np.sqrt(np.mean(clipped * clipped)))


def estimate_subject_blur_sigma(
    subject_image,
    database_image,
    subject_face_mask,
    database_face_mask
):
    """根据当前样本照片的清晰度估计被试照片所需的轻度模糊。"""
    subject_detail = measure_face_detail(subject_image, subject_face_mask)
    database_detail = measure_face_detail(database_image, database_face_mask)
    if subject_detail <= 0.0 or database_detail <= 0.0:
        return 0.0

    detail_ratio = subject_detail / database_detail
    if detail_ratio <= SUBJECT_BLUR_TRIGGER_RATIO:
        return 0.0

    excess_stops = np.log2(detail_ratio / SUBJECT_BLUR_TRIGGER_RATIO)
    sigma = SUBJECT_BLUR_MIN_SIGMA + 0.55 * excess_stops
    return float(np.clip(
        sigma,
        SUBJECT_BLUR_MIN_SIGMA,
        SUBJECT_BLUR_MAX_SIGMA
    ))


def soften_subject_for_morph(image, blur_sigma, subject_ratio):
    """从40%开始渐进降低被试锐度，60–80%使用完整匹配强度。"""
    sigma = float(blur_sigma or 0.0)
    ratio = float(np.clip(subject_ratio, 0.0, 1.0))
    if sigma <= 0.0 or ratio <= SUBJECT_BLUR_START_RATIO:
        return image

    strength = np.clip(
        (ratio - SUBJECT_BLUR_START_RATIO) /
        (SUBJECT_BLUR_FULL_RATIO - SUBJECT_BLUR_START_RATIO),
        0.0,
        1.0
    )
    strength = strength * strength * (3.0 - 2.0 * strength)
    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101
    )
    return cv2.addWeighted(
        blurred,
        float(strength),
        image,
        float(1.0 - strength),
        0.0
    )


def create_shared_face_blend_mask(
    mask1,
    mask2,
    points,
    image_shape,
    base_face_mask=None
):
    """以数据库可见面部为覆盖范围，并生成严格位于皮肤内的羽化。"""
    h_img, w_img = image_shape[:2]
    if mask2 is None:
        hard_mask = build_face_contour_mask(points, image_shape)
    else:
        hard_mask = np.where(mask2 > 127, 255, 0).astype(np.uint8)
    if base_face_mask is not None:
        hard_mask = np.where(
            (hard_mask > 0) & (base_face_mask > 127),
            255,
            0
        ).astype(np.uint8)

    hard_mask = cv2.morphologyEx(
        hard_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )
    contours, _ = cv2.findContours(
        hard_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    solid_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(solid_mask, [largest], -1, 255, thickness=-1)

    # 距离变换只在面部内部增加权重；掩膜外始终为零，所以头发不会参与融合。
    distance = cv2.distanceTransform(solid_mask, cv2.DIST_L2, 5)
    feather_width = max(
        8.0,
        min(h_img, w_img) * FACE_MASK_INNER_FEATHER_RATIO
    )
    soft_mask = np.clip(distance / feather_width, 0.0, 1.0)
    return soft_mask.astype(np.float32)[:, :, None]


def create_source_texture_confidence(
    mask,
    image_shape,
    target_face_mask=None
):
    """让被试纹理从目标面孔边界均匀渐入，避免额头压缩产生尖峰。"""
    h_img, w_img = image_shape[:2]
    confidence_mask = target_face_mask if target_face_mask is not None else mask
    if confidence_mask is None:
        return np.zeros((h_img, w_img, 1), dtype=np.float32)

    # 被试的可见面孔掩膜经过“高额头 -> 短额头”的强压缩后，内缩边界会
    # 变成细长尖峰。用它计算距离会令肤色沿尖峰进入额头。目标照片的面孔
    # 边界才是输出空间中的真实发际线，因此渐入距离必须由目标边界决定。
    hard_mask = np.where(confidence_mask > 127, 255, 0).astype(np.uint8)
    hard_mask = cv2.morphologyEx(
        hard_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1
    )
    contours, _ = cv2.findContours(
        hard_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    solid_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    if contours:
        cv2.drawContours(
            solid_mask,
            [max(contours, key=cv2.contourArea)],
            -1,
            255,
            thickness=-1
        )

    distance = cv2.distanceTransform(solid_mask, cv2.DIST_L2, 5)
    safe_width = max(18.0, min(h_img, w_img) * 0.12)
    confidence = np.clip(distance / safe_width, 0.0, 1.0)
    confidence = confidence * confidence * (3.0 - 2.0 * confidence)
    return confidence.astype(np.float32)[:, :, None]


def build_tone_matched_database_texture(
    target_morph,
    database_image,
    source_confidence,
    face_mask
):
    """用数据库真实皮肤补全边缘，并匹配当前比例已经生成的目标肤色。"""
    confidence = source_confidence[:, :, 0]
    coverage = face_mask[:, :, 0]
    reliable_skin = (confidence >= 0.90) & (coverage >= 0.90)
    if np.count_nonzero(reliable_skin) < 128:
        return database_image

    target_lab = cv2.cvtColor(target_morph, cv2.COLOR_BGR2LAB).astype(np.float32)
    database_lab = cv2.cvtColor(
        database_image,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)
    skin_delta = np.median(
        target_lab[reliable_skin] - database_lab[reliable_skin],
        axis=0
    )
    skin_delta = np.clip(
        skin_delta,
        np.array([-72.0, -26.0, -26.0], dtype=np.float32),
        np.array([72.0, 26.0, 26.0], dtype=np.float32)
    )

    # 全局中位数负责跨肤色的大范围校正；低频差分只在远离发际线的
    # 可靠区域逐渐启用，使额头/面颊的光照连续，又不会把头发暗色扩散进来。
    blur_sigma = max(8.0, min(database_image.shape[:2]) * 0.045)
    target_low = cv2.GaussianBlur(
        target_lab,
        (0, 0),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma,
        borderType=cv2.BORDER_REFLECT_101
    )
    database_low = cv2.GaussianBlur(
        database_lab,
        (0, 0),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma,
        borderType=cv2.BORDER_REFLECT_101
    )
    local_delta = target_low - database_low
    local_allowance = np.array([18.0, 8.0, 8.0], dtype=np.float32)
    local_delta = np.clip(
        local_delta,
        skin_delta - local_allowance,
        skin_delta + local_allowance
    )
    local_weight = np.clip(
        (confidence - 0.18) / 0.67,
        0.0,
        1.0
    )
    local_weight = local_weight * local_weight * (3.0 - 2.0 * local_weight)
    applied_delta = (
        skin_delta[None, None, :] * (1.0 - local_weight[:, :, None]) +
        local_delta * local_weight[:, :, None]
    )
    matched_lab = database_lab + applied_delta
    return cv2.cvtColor(
        np.clip(matched_lab, 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2BGR
    )


def composite_face_inward(face_image, base_image, face_mask):
    """在数据库面部内部做平滑直接合成，禁止暗色跨发际线扩散。"""
    mask = np.clip(face_mask.astype(np.float32), 0.0, 1.0)
    # smoothstep 使皮肤边缘连续，但保持掩膜外严格为数据库原图。
    mask = mask * mask * (3.0 - 2.0 * mask)
    result = (
        face_image.astype(np.float32) * mask +
        base_image.astype(np.float32) * (1.0 - mask)
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def harmonize_face_boundary(face_image, base_image, face_mask):
    """按位置匹配面缘低频光照，中央五官纹理保持不变。"""
    mask = np.clip(face_mask[:, :, 0].astype(np.float32), 0.0, 1.0)
    boundary_ring = (mask >= 0.18) & (mask <= 0.82)
    if np.count_nonzero(boundary_ring) < 64:
        return face_image

    face_lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    base_lab = cv2.cvtColor(base_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    global_delta = np.median(
        base_lab[boundary_ring] - face_lab[boundary_ring],
        axis=0
    )
    global_delta = np.clip(
        global_delta,
        np.array([-48.0, -18.0, -18.0], dtype=np.float32),
        np.array([48.0, 18.0, 18.0], dtype=np.float32)
    )

    # 数据库照片左右光照、胡须和太阳穴颜色通常不同。分别估计每个位置的
    # 低频差异，比单一全局偏移更能消除“贴上一张椭圆脸”的边界。
    blur_sigma = max(8.0, min(face_image.shape[:2]) * 0.040)
    face_low = cv2.GaussianBlur(
        face_lab,
        (0, 0),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma,
        borderType=cv2.BORDER_REFLECT_101
    )
    base_low = cv2.GaussianBlur(
        base_lab,
        (0, 0),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma,
        borderType=cv2.BORDER_REFLECT_101
    )
    local_delta = base_low - face_low
    local_delta = np.clip(
        local_delta,
        np.array([-64.0, -22.0, -22.0], dtype=np.float32),
        np.array([64.0, 22.0, 22.0], dtype=np.float32)
    )
    local_delta = 0.85 * local_delta + 0.15 * global_delta[None, None, :]

    edge_weight = np.where(
        mask > 0.0,
        np.power(1.0 - mask, 0.72),
        0.0
    ).astype(np.float32)
    adjusted_lab = face_lab + edge_weight[:, :, None] * local_delta
    return cv2.cvtColor(
        np.clip(adjusted_lab, 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2BGR
    )


def blend_face_without_hair_ghosting(
    warp1,
    warp2,
    points_avg,
    alpha,
    database_base,
    database_base_mask=None,
    warped_mask1=None,
    warped_mask2=None
):
    """只把几何融合后的面部写入未形变数据库底图。"""
    face_mask = create_shared_face_blend_mask(
        warped_mask1,
        warped_mask2,
        points_avg,
        warp1.shape,
        base_face_mask=database_base_mask
    )
    source_confidence = create_source_texture_confidence(
        warped_mask1,
        warp1.shape,
        target_face_mask=database_base_mask
    )
    subject_ratio = 1.0 - alpha
    standard_morph = blend_face_textures_lab(
        warp1,
        warp2,
        subject_ratio
    )
    database_fill = build_tone_matched_database_texture(
        standard_morph,
        warp2,
        source_confidence,
        face_mask
    )
    # 可靠区使用正常双脸融合；靠近被试发际线时换成已匹配目标肤色的
    # 数据库真实皮肤纹理，杜绝被拉长的发丝进入额头。
    blended_face = (
        standard_morph.astype(np.float32) * source_confidence +
        database_fill.astype(np.float32) * (1.0 - source_confidence)
    ).astype(np.uint8)

    # 头发、耳朵和矩形头像背景均取自未形变数据库图；面孔之外不做
    # 颜色分类或几何变形，避免把侧光阴影误认为头发。
    matched_database_base = database_base
    harmonized_face = harmonize_face_boundary(
        blended_face,
        matched_database_base,
        face_mask
    )
    return composite_face_inward(
        harmonized_face,
        matched_database_base,
        face_mask
    )


def interpolate_face_geometry(points1, points2, alpha):
    """融合五官几何，并把发际线与下颌固定到样本头部外框。"""
    source_points = np.asarray(points1, dtype=np.float32)
    database_points = np.asarray(points2, dtype=np.float32)
    target_points = (
        (1.0 - alpha) * source_points +
        alpha * database_points
    )

    if len(target_points) >= FACE_BOUNDARY_POINT_START:
        # 中间比例最终写回未变形的样本头部。固定下颌和发际线可使面孔
        # 与样本耳朵、头发精确相接，避免不同脸宽造成椭圆贴图边。
        target_points[0:17] = database_points[0:17]
        forehead_end = (
            FACE_FOREHEAD_POINT_START + FACE_FOREHEAD_POINT_COUNT
        )
        target_points[
            FACE_FOREHEAD_POINT_START:forehead_end
        ] = database_points[
            FACE_FOREHEAD_POINT_START:forehead_end
        ]
        target_points[FACE_BOUNDARY_POINT_START:] = database_points[
            FACE_BOUNDARY_POINT_START:
        ]
    return target_points


def morph_faces_full(
    img1_arr,
    img2_arr,
    alpha=0.5,
    points1=None,
    points2=None,
    face_mask1=None,
    face_mask2=None,
    subject_blur_sigma=0.0
):
    """使用提供文件中的全图 Delaunay 三角形仿射与线性融合策略。"""
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

        # 当前样本筛选仍可保留扩展额头网格；实际融合严格取提供文件使用的
        # 68 个面部点和 8 个画布边界点。
        if len(points1) > 76:
            points1 = np.concatenate((points1[:68], points1[-8:]), axis=0)
        if len(points2) > 76:
            points2 = np.concatenate((points2[:68], points2[-8:]), axis=0)

        if len(points1) == 0 or len(points2) == 0 or len(points1) != len(points2):
            return cv2.addWeighted(img1, 1-alpha, img2, alpha, 0), None

        points_avg = (1 - alpha) * points1 + alpha * points2
        triangles = get_triangles(points_avg)
        warp1 = warp_image(img1, points1, points_avg, triangles)
        warp2 = warp_image(img2, points2, points_avg, triangles)
        final_img = cv2.addWeighted(warp1, 1-alpha, warp2, alpha, 0)
        return final_img, points_avg
    except Exception as e:
        print(f"Morphing error: {e}")
        s1 = cv2.resize(img1_arr, (PROCESS_WIDTH, PROCESS_HEIGHT))
        s2 = cv2.resize(img2_arr, (PROCESS_WIDTH, PROCESS_HEIGHT))
        return cv2.addWeighted(s1, 1-alpha, s2, alpha, 0), None

def crop_face_region(image, landmarks, output_width, output_height):
    """按面部关键点标准化裁剪，只保留头部并在下巴下方留下极小余量。"""
    if image is None:
        return None
    if landmarks is None or len(landmarks) < 68:
        return cv2.resize(image, (output_width, output_height))

    try:
        face_pts = np.asarray(landmarks[:68], dtype=np.float32)
        jaw = face_pts[0:17]
        brows = face_pts[17:27]

        jaw_left = float(np.min(jaw[:, 0]))
        jaw_right = float(np.max(jaw[:, 0]))
        brow_top = float(np.min(brows[:, 1]))
        chin_y = float(np.max(jaw[:, 1]))
        face_width = jaw_right - jaw_left
        face_height = chin_y - brow_top

        if face_width <= 1 or face_height <= 1:
            return cv2.resize(image, (output_width, output_height))

        target_aspect = output_width / float(output_height)
        desired_top = brow_top - FACE_CROP_FOREHEAD_MARGIN * face_height
        desired_bottom = chin_y + FACE_CROP_CHIN_MARGIN * face_height

        # 同时满足额头到下巴的纵向范围和下颌两侧的最小留白。
        crop_height = desired_bottom - desired_top
        min_crop_width = face_width * (1.0 + 2.0 * FACE_CROP_SIDE_MARGIN)
        crop_height = max(crop_height, min_crop_width / target_aspect)
        crop_width = crop_height * target_aspect

        center_x = (jaw_left + jaw_right) / 2.0
        # 固定下边界在下巴附近；为满足比例而增加的高度全部放到头顶方向。
        x1 = int(round(center_x - crop_width / 2.0))
        x2 = int(round(center_x + crop_width / 2.0))
        y2 = int(round(desired_bottom))
        y1 = int(round(y2 - crop_height))

        h_img, w_img = image.shape[:2]
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w_img)
        pad_bottom = max(0, y2 - h_img)

        if pad_left or pad_top or pad_right or pad_bottom:
            image = cv2.copyMakeBorder(
                image,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REFLECT_101
            )
            x1 += pad_left
            x2 += pad_left
            y1 += pad_top
            y2 += pad_top

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return cv2.resize(image, (output_width, output_height))
        return cv2.resize(
            crop,
            (output_width, output_height),
            interpolation=cv2.INTER_LANCZOS4
        )
    except Exception as e:
        print(f"Error in crop_face_region: {e}")
        return cv2.resize(image, (output_width, output_height))


def align_face_to_canvas(image, landmarks, output_width, output_height):
    """按双眼和嘴部中心执行相似变换，使两张脸进入同一规范坐标系。"""
    if image is None:
        return None
    if landmarks is None or len(landmarks) < 68:
        return cv2.resize(image, (output_width, output_height))

    try:
        face_pts = np.asarray(landmarks[:68], dtype=np.float32)
        left_eye = np.mean(face_pts[36:42], axis=0)
        right_eye = np.mean(face_pts[42:48], axis=0)
        mouth_center = (face_pts[48] + face_pts[54]) / 2.0
        source_anchors = np.float32([
            left_eye,
            right_eye,
            mouth_center
        ])
        destination_anchors = np.float32([
            [FACE_ALIGN_LEFT_EYE[0] * output_width,
             FACE_ALIGN_LEFT_EYE[1] * output_height],
            [FACE_ALIGN_RIGHT_EYE[0] * output_width,
             FACE_ALIGN_RIGHT_EYE[1] * output_height],
            [FACE_ALIGN_MOUTH[0] * output_width,
             FACE_ALIGN_MOUTH[1] * output_height]
        ])

        transform, _ = cv2.estimateAffinePartial2D(
            source_anchors,
            destination_anchors,
            method=cv2.LMEDS
        )
        if transform is None or not np.all(np.isfinite(transform)):
            return crop_face_region(
                image,
                landmarks,
                output_width,
                output_height
            )

        scale_area = abs(float(np.linalg.det(transform[:, :2])))
        if scale_area < 1e-6:
            return crop_face_region(
                image,
                landmarks,
                output_width,
                output_height
            )

        return cv2.warpAffine(
            image,
            transform,
            (output_width, output_height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(FACE_CANVAS_VALUE,) * 3
        )
    except Exception as e:
        print(f"Error in align_face_to_canvas: {e}")
        return crop_face_region(
            image,
            landmarks,
            output_width,
            output_height
        )


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

@lru_cache(maxsize=256)
def load_prepared_db_image(filepath):
    """缓存样本库的规范矩形头像、融合网格与可见面部掩膜。"""
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Cannot read database image: {filepath}")
    img_resized = crop_portrait_wide(img)
    if img_resized is None:
        img_resized = cv2.resize(img, (PROCESS_WIDTH, PROCESS_HEIGHT))
    detected_points = get_points(img_resized)
    points, face_mask = prepare_face_geometry(img_resized, detected_points)
    quality_issue = database_face_quality_issue(img_resized, face_mask)
    if quality_issue:
        raise ValueError(
            f"Database face quality rejected: {quality_issue}"
        )
    return img_resized, points, face_mask

def get_random_original_db_images(gender, count=3, excluded_filenames=None):
    """随机选取指定数量的不重复同性数据库面孔，并完成裁剪和关键点预处理。"""
    folder = 'male' if gender == 'male' else 'female'
    path = os.path.join(DATABASE_PATH, folder)
    all_files = glob.glob(os.path.join(path, "*.jpg")) + glob.glob(os.path.join(path, "*.png"))
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
    ]
    random.shuffle(candidates)
    selected_images = []
    for selected_file in candidates:
        try:
            img_resized, points, face_mask = load_prepared_db_image(selected_file)
            if (
                len(points) >= FACE_BOUNDARY_POINT_START and
                face_mask is not None
            ):
                selected_images.append(
                    (
                        img_resized,
                        os.path.basename(selected_file),
                        points,
                        face_mask
                    )
                )
                if len(selected_images) == count:
                    break
        except Exception as e:
            print(
                f"Skipping database face {os.path.basename(selected_file)}: {e}"
            )
            continue
    return selected_images

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

def build_morph_stimulus(job):
    """按提供文件的三角形仿射算法生成单张刺激图。"""
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

    result = {
        "id": f"stim_{job['trial_id']}",
        "url": cv2_to_base64(final_face),
        "type": job["stimulus_type"],
        "description": job["description"],
        "source_upload": job["source_upload"],
        "source_db": job["source_db"]
    }
    result[job["ratio_key"]] = ratio
    return result

# --- 核心路由: 图片融合处理 ---
@app.route('/merge_faces', methods=['POST'])
def process_images_experiment():
    try:
        data = request.json
        self_gender = data.get('self_gender', 'male')
        partner_gender = data.get('partner_gender', 'female')
        raw_self = base64_to_cv2(data.get('self_image'))
        raw_partner = base64_to_cv2(data.get('partner_image'))
        
        if raw_self is None or raw_partner is None:
            return jsonify({"error": "Invalid images"}), 400

        img_self_wide = crop_portrait_wide(raw_self)
        img_partner_wide = crop_portrait_wide(raw_partner)

        # 上传照片的关键点只计算一次，供全部融合比例复用。
        self_detected_points = get_points(img_self_wide)
        partner_detected_points = get_points(img_partner_wide)

        invalid_roles = []
        if len(self_detected_points) < 68:
            invalid_roles.append("your photograph")
        if len(partner_detected_points) < 68:
            invalid_roles.append("your partner's photograph")

        if invalid_roles:
            return jsonify({
                "error": (
                    "A clear face could not be detected in "
                    + " and ".join(invalid_roles)
                    + ". Please use a front-facing photograph without glasses, "
                      "sunglasses, or face coverings, then try again."
                ),
                "code": "FACE_QUALITY_ERROR"
            }), 422

        # 每张上传照片独立估计实际可见额头，并将曲线加入三角形变网格。
        self_points, self_face_mask = prepare_face_geometry(
            img_self_wide,
            self_detected_points
        )
        partner_points, partner_face_mask = prepare_face_geometry(
            img_partner_wide,
            partner_detected_points
        )
        if (
            len(self_points) < FACE_BOUNDARY_POINT_START or
            len(partner_points) < FACE_BOUNDARY_POINT_START
        ):
            return jsonify({
                "error": "The visible face boundary could not be estimated reliably. Please use an evenly lit, front-facing photograph.",
                "code": "FACE_BOUNDARY_ERROR"
            }), 422

        quality_errors = []
        self_quality_issue = database_face_quality_issue(
            img_self_wide,
            self_face_mask
        )
        partner_quality_issue = database_face_quality_issue(
            img_partner_wide,
            partner_face_mask
        )
        if self_quality_issue:
            quality_errors.append(f"your photograph: {self_quality_issue}")
        if partner_quality_issue:
            quality_errors.append(
                f"your partner's photograph: {partner_quality_issue}"
            )
        if quality_errors:
            return jsonify({
                "error": "Photo quality is unsuitable: " + "; ".join(quality_errors),
                "code": "FACE_EXPOSURE_ERROR"
            }), 422

        # 双眼与嘴部已经被规范到同一坐标系。直接保留矩形头像内的头发、
        # 耳朵和原背景，不再做人像抠图或下颌镂空。

        photo_batch_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self_filename = save_uploaded_image(img_self_wide, photo_batch_id, "self")
        partner_filename = save_uploaded_image(img_partner_wide, photo_batch_id, "partner")
        print(f"[OK] Images saved: {self_filename}, {partner_filename}")

        result_images = []
        trial_id = 1
        ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        morph_jobs = []

        # 本人和伴侣各自固定使用 3 个同性数据库面孔；同一面孔覆盖全部 6 个比例。
        # 双方性别相同时，伴侣组排除本人组已经选中的数据库身份。
        self_db_faces = get_random_original_db_images(self_gender, count=3)
        partner_excluded_filenames = (
            {db_filename for _, db_filename, _, _ in self_db_faces}
            if self_gender == partner_gender
            else set()
        )
        partner_db_faces = get_random_original_db_images(
            partner_gender,
            count=3,
            excluded_filenames=partner_excluded_filenames
        )
        if len(self_db_faces) < 3 or len(partner_db_faces) < 3:
            return jsonify({
                "error": "At least three valid same-gender database faces are required for each photograph.",
                "code": "INSUFFICIENT_DATABASE_FACES"
            }), 500

        # 1. Self Morphs: 3 database identities × 6 ratios = 18 images
        for db_index, (db_img, db_filename, db_points, db_mask) in enumerate(self_db_faces, start=1):
            subject_blur_sigma = estimate_subject_blur_sigma(
                img_self_wide,
                db_img,
                self_face_mask,
                db_mask
            )
            for ratio in ratios:
                morph_jobs.append({
                    "trial_id": trial_id,
                    "subject_image": img_self_wide,
                    "subject_points": self_points,
                    "subject_mask": self_face_mask,
                    "database_image": db_img,
                    "database_points": db_points,
                    "database_mask": db_mask,
                    "subject_blur_sigma": subject_blur_sigma,
                    "ratio": ratio,
                    "ratio_key": "ratio_self",
                    "stimulus_type": "self_morph",
                    "description": f"Self Morph {int(ratio*100)}% / Face {db_index}",
                    "source_upload": self_filename,
                    "source_db": db_filename
                })
                trial_id += 1

        # 2. Partner Morphs: 3 database identities × 6 ratios = 18 images
        for db_index, (db_img, db_filename, db_points, db_mask) in enumerate(partner_db_faces, start=1):
            subject_blur_sigma = estimate_subject_blur_sigma(
                img_partner_wide,
                db_img,
                partner_face_mask,
                db_mask
            )
            for ratio in ratios:
                morph_jobs.append({
                    "trial_id": trial_id,
                    "subject_image": img_partner_wide,
                    "subject_points": partner_points,
                    "subject_mask": partner_face_mask,
                    "database_image": db_img,
                    "database_points": db_points,
                    "database_mask": db_mask,
                    "subject_blur_sigma": subject_blur_sigma,
                    "ratio": ratio,
                    "ratio_key": "ratio_partner",
                    "stimulus_type": "partner_morph",
                    "description": f"Partner Morph {int(ratio*100)}% / Face {db_index}",
                    "source_upload": partner_filename,
                    "source_db": db_filename
                })
                trial_id += 1

        # 人脸检测已在主线程完成；这里只并行执行相互独立的几何变换。
        with ThreadPoolExecutor(max_workers=MAX_MORPH_WORKERS) as executor:
            result_images.extend(executor.map(build_morph_stimulus, morph_jobs))

        random.shuffle(result_images)
        return jsonify({
            "status": "success",
            "photo_batch_id": photo_batch_id,
            "images": result_images
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

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
        is_complete = data.get('is_complete', False)
        
        # 1. JSON 是最终数据的主记录。采用锁和原子替换，避免自动保存与最终保存并发写坏文件。
        json_filename = f"experiment_data_{participant_id}.json"
        json_filepath = os.path.join(DATA_SAVE_PATH, json_filename)
        json_temp_path = None
        with DATA_SAVE_LOCK:
            existing_data = None
            if os.path.isfile(json_filepath):
                try:
                    with open(json_filepath, 'r', encoding='utf-8') as existing_file:
                        existing_data = json.load(existing_file)
                except (OSError, ValueError):
                    existing_data = None

            # 已经完成的记录不能被稍后到达的旧自动保存请求降级覆盖。
            preserve_completed_record = (
                isinstance(existing_data, dict) and
                existing_data.get('is_complete') is True and
                not is_complete
            )

            if not preserve_completed_record:
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
                    if json_temp_path and os.path.exists(json_temp_path):
                        os.remove(json_temp_path)

        excel_saved = None
        excel_destination = None
        excel_warning = None
            
        # 2. 只有当实验标记为 完成 (is_complete = True) 时，才去读写 Excel
        if is_complete:
            # Windows PowerShell 常使用 GBK 控制台，日志必须避免 emoji，否则会在保存前抛出编码异常。
            print(f"[INFO] Experiment complete (ID: {participant_id}); syncing data to Excel...")
            excel_master_path = os.path.join(DATA_SAVE_PATH, 'all_experiment_data.xlsx')
            
            # --- 构建数据行 ---
            common_info = {
                'Participant_ID': participant_id,
                'SONA_ID': data.get('sona_id', ''),
                'Timestamp': data.get('timestamp', datetime.now().isoformat()),
                'Condition_Group': data.get('condition_group'),
                'Self_Gender': data.get('gender_info', {}).get('self'),
                'Partner_Gender': data.get('gender_info', {}).get('partner'),
                'User_Profile_Text': data.get('user_profile'),
                'Mode': data.get('mode'),
                'Is_Complete': is_complete
            }
            
            # 问卷数据
            q_answers = data.get('pre_questionnaire', {})
            for i in range(16):
                common_info[f'Q_{i+1}'] = q_answers.get(str(i), '')

            rows = []
            experiment_trials = data.get('experiment_data', [])
            
            if not experiment_trials:
                rows.append(common_info)
            else:
                for trial in experiment_trials:
                    row = common_info.copy()
                    row.update({
                        'Trial_Index': trial.get('trial_index'),
                        'Stimulus_ID': trial.get('stimulus_id'),
                        'Stimulus_Type': trial.get('stimulus_type'),
                        'Ratio_Level': trial.get('ratio_level'),
                        'Source_DB_Image': trial.get('source_db_image'),
                        'Source_Upload_Image': trial.get('source_upload_image'),
                        'Action': trial.get('action'),
                        'RT_ms': trial.get('reaction_time_ms'),
                        'Rating_Desirability': trial.get('rating_desirability'),
                        'Rating_Willingness': trial.get('rating_willingness')
                    })
                    rows.append(row)
            
            new_df = pd.DataFrame(rows)
            
            # --- 写入 Excel：串行处理并先写临时文件，避免多名被试同时完成时损坏主表。 ---
            excel_temp_path = None
            with EXCEL_SAVE_LOCK:
                try:
                    if os.path.exists(excel_master_path):
                        old_df = pd.read_excel(excel_master_path)
                        if 'Participant_ID' in old_df.columns:
                            old_df = old_df[old_df['Participant_ID'] != participant_id]
                        combined_df = pd.concat([old_df, new_df], ignore_index=True)
                    else:
                        combined_df = new_df

                    excel_temp_path = os.path.join(
                        DATA_SAVE_PATH,
                        f".all_experiment_data.{uuid.uuid4().hex}.tmp.xlsx"
                    )
                    combined_df.to_excel(excel_temp_path, index=False)
                    os.replace(excel_temp_path, excel_master_path)
                    excel_saved = True
                    excel_destination = 'master'
                    print(f"[OK] Excel updated: {excel_master_path}")
                except Exception as excel_err:
                    excel_warning = f"Master Excel update failed: {excel_err}"
                    print(f"[ERROR] {excel_warning}")
                    backup_path = os.path.join(
                        DATA_SAVE_PATH,
                        f"backup_{participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.xlsx"
                    )
                    try:
                        new_df.to_excel(backup_path, index=False)
                        excel_saved = True
                        excel_destination = 'backup'
                        print(f"[OK] Data saved to backup: {backup_path}")
                    except Exception as backup_err:
                        excel_saved = False
                        excel_warning = (
                            f"{excel_warning}; backup Excel write failed: {backup_err}. "
                            "The complete JSON record was saved successfully."
                        )
                        print(f"[ERROR] {excel_warning}")
                finally:
                    if excel_temp_path and os.path.exists(excel_temp_path):
                        os.remove(excel_temp_path)

        response_body = {
            "status": "success",
            "participant_id": participant_id,
            "json_saved": True,
            "excel_saved": excel_saved,
            "excel_destination": excel_destination
        }
        if excel_warning:
            response_body["warning"] = excel_warning
        return jsonify(response_body)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)
