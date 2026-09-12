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

# 以 68 点中的眉毛、下颌轮廓为基准，只保留头部和面孔。
# 下边界只越过下巴少量距离，避免肩部进入融合区域。
FACE_CROP_SIDE_MARGIN = 0.04
FACE_CROP_FOREHEAD_MARGIN = 0.42
FACE_CROP_CHIN_MARGIN = 0.05
FACE_BLEND_FOREHEAD_MARGIN = 0.38
FACE_BLEND_EXPAND_RATIO = 0.025
FACE_BLEND_FEATHER_RATIO = 0.055

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
    print("✅ Dlib 模型加载成功 (Dlib models loaded successfully).")
except Exception as e:
    print(f"❌ 警告: Dlib 模型加载失败。请确保 '{PREDICTOR_PATH}' 文件存在。")
    print(e)

# --- 核心图像处理函数 (保持不变) ---
# 这些函数负责特征点检测、三角剖分、仿射变换和人脸融合

def get_points(image):
    try:
        if image is None: return np.array([])
        if detector is None or predictor is None: return np.array([])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 先使用快速检测；失败时再放大检测一次，以兼顾速度和成功率。
        dets = detector(gray, 0)
        if len(dets) == 0:
            dets = detector(gray, 1)
        if len(dets) == 0:
            return np.array([])

        detected_face = dets[0]
        pose_landmarks = predictor(gray, detected_face)
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
        # 使用当前图像块填补失败区域，避免出现黑色或白色三角形。
        return cv2.resize(
            input_image,
            (size[0], size[1]),
            interpolation=cv2.INTER_LINEAR
        )

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
    # 以原图初始化，三角形取整产生的极小空隙会保留原始底图颜色。
    warped_img = np.copy(img)
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


def create_face_blend_mask(points, image_shape):
    """创建与肤色无关、内部连续且避开发际线的完整面部掩膜。"""
    h_img, w_img = image_shape[:2]
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    if points is None or len(points) < 68:
        return mask.astype(np.float32)[:, :, None]

    face_pts = np.asarray(points[:68], dtype=np.float32)
    jaw = face_pts[0:17]
    brows = face_pts[17:27]
    jaw_left = float(np.min(jaw[:, 0]))
    jaw_right = float(np.max(jaw[:, 0]))
    brow_left = float(np.min(brows[:, 0]))
    brow_right = float(np.max(brows[:, 0]))
    brow_top = float(np.min(brows[:, 1]))
    chin_y = float(np.max(jaw[:, 1]))
    face_width = max(1.0, jaw_right - jaw_left)
    face_height = max(1.0, chin_y - brow_top)
    center_x = (jaw_left + jaw_right) / 2.0

    # dlib 的 68 点不含发际线，因此用眉毛—下巴高度构造稳定的额头上缘。
    # 掩膜只由几何位置决定，不再按肤色阈值筛选，避免不同人种融合时出现孔洞。
    forehead_top = brow_top - FACE_BLEND_FOREHEAD_MARGIN * face_height
    forehead_points = np.array([
        [brow_left - 0.06 * face_width, brow_top],
        [center_x - 0.38 * face_width, forehead_top + 0.06 * face_height],
        [center_x, forehead_top],
        [center_x + 0.38 * face_width, forehead_top + 0.06 * face_height],
        [brow_right + 0.06 * face_width, brow_top]
    ], dtype=np.float32)
    full_face_hull = cv2.convexHull(
        np.int32(np.round(np.vstack([jaw, brows, forehead_points])))
    )
    cv2.fillConvexPoly(mask, full_face_hull, 255, lineType=cv2.LINE_AA)

    # 略微向外扩张，使下颌、脸颊和太阳穴完整覆盖；实心凸包保证内部无孔洞。
    expand_size = max(3, int(round(face_width * FACE_BLEND_EXPAND_RATIO)))
    if expand_size % 2 == 0:
        expand_size += 1
    expand_size = min(expand_size, 31)
    expand_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (expand_size, expand_size)
    )
    mask = cv2.dilate(mask, expand_kernel, iterations=1)

    # 宽渐变带只作用于轮廓边缘，内部仍保持完整融合；可柔化跨肤色边界。
    feather_size = max(
        9,
        int(round(min(h_img, w_img) * FACE_BLEND_FEATHER_RATIO))
    )
    if feather_size % 2 == 0:
        feather_size += 1
    feather_size = min(feather_size, 71)
    mask = cv2.GaussianBlur(mask, (feather_size, feather_size), 0)
    return mask.astype(np.float32)[:, :, None] / 255.0


def blend_face_without_hair_ghosting(warp1, warp2, points_avg, alpha):
    """数据库图作为底图，在连续完整面部掩膜内融合皮肤和五官。"""
    blended_face = cv2.addWeighted(warp1, 1-alpha, warp2, alpha, 0)

    # 第二张图是同一组比例共用的数据库模板。其头发、背景和外轮廓
    # 完整保留，只有掩膜内部的眼鼻口及面部区域随比例变化。
    hair_base = warp2
    face_mask = create_face_blend_mask(points_avg, blended_face.shape)
    result = (
        blended_face.astype(np.float32) * face_mask +
        hair_base.astype(np.float32) * (1.0 - face_mask)
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def morph_faces_full(img1_arr, img2_arr, alpha=0.5, points1=None, points2=None):
    try:
        if img1_arr.shape[:2] != (PROCESS_HEIGHT, PROCESS_WIDTH):
            img1_arr = cv2.resize(img1_arr, (PROCESS_WIDTH, PROCESS_HEIGHT))
        if img2_arr.shape[:2] != (PROCESS_HEIGHT, PROCESS_WIDTH):
            img2_arr = cv2.resize(img2_arr, (PROCESS_WIDTH, PROCESS_HEIGHT))
        img1 = np.copy(img1_arr)
        img2 = np.copy(img2_arr)

        if points1 is None:
            points1 = get_points(img1)
        if points2 is None:
            points2 = get_points(img2)

        if len(points1) < 68 or len(points2) < 68 or len(points1) != len(points2):
            # 无法可靠定位五官时保留数据库模板，禁止退化为整图透明叠加。
            return img2, None
        points_avg = (1 - alpha) * np.array(points1) + alpha * np.array(points2)
        triangles = get_triangles(points_avg)
        warp1 = warp_image(img1, points1, points_avg, triangles)
        warp2 = warp_image(img2, points2, points_avg, triangles)
        final_img = blend_face_without_hair_ghosting(
            warp1,
            warp2,
            points_avg,
            alpha
        )
        return final_img, points_avg
    except Exception as e:
        print(f"Morphing error: {e}")
        s2 = cv2.resize(img2_arr, (PROCESS_WIDTH, PROCESS_HEIGHT))
        return s2, None

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


# --- 融合结果紧裁剪：保留面孔和头部，不保留肩部 ---
def crop_face_tight(image, landmarks):
    return crop_face_region(image, landmarks, OUTPUT_WIDTH, OUTPUT_HEIGHT)

# --- 面孔紧裁剪（用于融合前标准化上传图和数据库图） ---
def crop_portrait_wide(image):
    if image is None: return None
    try:
        points = get_points(image)
        if len(points) < 68:
            return cv2.resize(image, (PROCESS_WIDTH, PROCESS_HEIGHT))
        return crop_face_region(
            image,
            points,
            PROCESS_WIDTH,
            PROCESS_HEIGHT
        )
    except Exception as e:
        print(f"Error in crop_portrait_wide: {e}")
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
    """缓存素材照片的裁剪结果和关键点，后续被试可直接复用。"""
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Cannot read database image: {filepath}")
    img_resized = crop_portrait_wide(img)
    if img_resized is None:
        img_resized = cv2.resize(img, (PROCESS_WIDTH, PROCESS_HEIGHT))
    points = get_points(img_resized)
    return img_resized, points

def get_random_original_db_images(gender, count=3):
    """随机选取指定数量的不重复同性数据库面孔，并完成裁剪和关键点预处理。"""
    folder = 'male' if gender == 'male' else 'female'
    path = os.path.join(DATABASE_PATH, folder)
    all_files = glob.glob(os.path.join(path, "*.jpg")) + glob.glob(os.path.join(path, "*.png"))
    original_files = [f for f in all_files if "_uploaded" not in os.path.basename(f)]
    if not original_files:
        if not all_files:
            return []
        original_files = all_files

    candidates = original_files.copy()
    random.shuffle(candidates)
    selected_images = []
    for selected_file in candidates:
        try:
            img_resized, points = load_prepared_db_image(selected_file)
            if len(points) >= 68:
                selected_images.append(
                    (img_resized, os.path.basename(selected_file), points)
                )
                if len(selected_images) == count:
                    break
        except Exception:
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
    """执行一个纯几何融合任务；关键点已预先计算，可安全并行。"""
    ratio = job["ratio"]
    alpha = 1.0 - ratio
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
        self_points = get_points(img_self_wide)
        partner_points = get_points(img_partner_wide)

        invalid_roles = []
        if len(self_points) < 68:
            invalid_roles.append("your photograph")
        if len(partner_points) < 68:
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

        photo_batch_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self_filename = save_uploaded_image(img_self_wide, photo_batch_id, "self")
        partner_filename = save_uploaded_image(img_partner_wide, photo_batch_id, "partner")
        print(f"✅ Images saved: {self_filename}, {partner_filename}")

        result_images = []
        trial_id = 1
        ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        morph_jobs = []

        # 本人和伴侣各自固定使用 3 个同性数据库面孔；同一面孔覆盖全部 6 个比例。
        self_db_faces = get_random_original_db_images(self_gender, count=3)
        partner_db_faces = get_random_original_db_images(partner_gender, count=3)
        if len(self_db_faces) < 3 or len(partner_db_faces) < 3:
            return jsonify({
                "error": "At least three valid same-gender database faces are required for each photograph.",
                "code": "INSUFFICIENT_DATABASE_FACES"
            }), 500

        # 1. Self Morphs: 3 database identities × 6 ratios = 18 images
        for db_index, (db_img, db_filename, db_points) in enumerate(self_db_faces, start=1):
            for ratio in ratios:
                morph_jobs.append({
                    "trial_id": trial_id,
                    "subject_image": img_self_wide,
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

        # 2. Partner Morphs: 3 database identities × 6 ratios = 18 images
        for db_index, (db_img, db_filename, db_points) in enumerate(partner_db_faces, start=1):
            for ratio in ratios:
                morph_jobs.append({
                    "trial_id": trial_id,
                    "subject_image": img_partner_wide,
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
        data = request.json
        # 如果是第一次请求，生成新的 ID
        participant_id = data.get('participant_id')
        if not participant_id:
            timestamp = int(time.time())
            participant_id = f"P_{timestamp}_{random.randint(1000,9999)}"
        
        data['participant_id'] = participant_id
        is_complete = data.get('is_complete', False)
        
        # 1. 始终保存 JSON 文件 (作为实时备份)
        json_filename = f"experiment_data_{participant_id}.json"
        json_filepath = os.path.join(DATA_SAVE_PATH, json_filename)
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        # 2. 只有当实验标记为 完成 (is_complete = True) 时，才去读写 Excel
        if is_complete:
            print(f"📊 实验完成 (ID: {participant_id})，正在同步数据到 Excel...")
            excel_master_path = os.path.join(DATA_SAVE_PATH, 'all_experiment_data.xlsx')
            
            # --- 构建数据行 ---
            common_info = {
                'Participant_ID': participant_id,
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
            
            # --- 写入 Excel (带文件锁保护) ---
            try:
                if os.path.exists(excel_master_path):
                    # 读取旧 Excel
                    old_df = pd.read_excel(excel_master_path)
                    # 先删除该 ID 可能存在的旧记录 (防止重复)
                    if 'Participant_ID' in old_df.columns:
                        old_df = old_df[old_df['Participant_ID'] != participant_id]
                    # 合并
                    combined_df = pd.concat([old_df, new_df], ignore_index=True)
                    combined_df.to_excel(excel_master_path, index=False)
                else:
                    new_df.to_excel(excel_master_path, index=False)
                print(f"✅ Excel 更新成功: {excel_master_path}")
            except Exception as excel_err:
                print(f"❌ Excel 写入失败 (文件可能被占用): {excel_err}")
                # 写入备份文件，防止数据丢失
                backup_path = os.path.join(DATA_SAVE_PATH, f"backup_{participant_id}.xlsx")
                new_df.to_excel(backup_path, index=False)
                print(f"✅ 数据已保存到备份文件: {backup_path}")

        return jsonify({"status": "success", "participant_id": participant_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)
