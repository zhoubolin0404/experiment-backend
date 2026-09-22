"""一次性建立数据库面孔的持久化预处理结果。

在实验开始前或修改 database/male、database/female 后运行：
    python preprocess_database.py

请在没有被试进行实验时执行；后端只读取已发布的结果，不自动重建。
"""

import json
import os
import tempfile
import uuid

import cv2
import numpy as np

from app import (
    DATABASE_PREPROCESS_VERSION,
    PREDICTOR_PATH,
    PREPROCESSED_DATABASE_PATH,
    PROCESS_HEIGHT,
    PROCESS_WIDTH,
    crop_portrait_wide,
    database_file_sha256,
    database_source_files,
    database_source_key,
    detector,
    get_points,
    predictor,
)


def build_database_preprocessing():
    if detector is None or predictor is None:
        raise RuntimeError('The dlib face detector or 68-point predictor is unavailable.')
    os.makedirs(PREPROCESSED_DATABASE_PATH, exist_ok=True)
    generation_name = f'generation_{uuid.uuid4().hex}'
    generation_path = os.path.join(PREPROCESSED_DATABASE_PATH, generation_name)
    os.mkdir(generation_path)

    predictor_stat = os.stat(PREDICTOR_PATH)
    entries = {}
    selected_counts = {}
    for gender in ('male', 'female'):
        files = database_source_files(gender)
        if not files:
            raise RuntimeError(f'No database photographs in database/{gender}.')
        original_files = [
            filepath for filepath in files
            if '_uploaded' not in os.path.basename(filepath)
        ]
        eligible_keys = {
            database_source_key(filepath)
            for filepath in (original_files or files)
        }

        for filepath in files:
            key = database_source_key(filepath)
            stat = os.stat(filepath)
            entry = {
                'source_size': stat.st_size,
                'source_mtime_ns': stat.st_mtime_ns,
                'source_sha256': database_file_sha256(filepath),
                'valid': False,
            }
            try:
                image = cv2.imread(filepath)
                if image is None:
                    raise ValueError('The image could not be read.')
                prepared_image = crop_portrait_wide(image)
                if prepared_image is None:
                    prepared_image = cv2.resize(image, (PROCESS_WIDTH, PROCESS_HEIGHT))
                points = get_points(prepared_image)
                if len(points) < 68:
                    raise ValueError('A clear face could not be detected.')
                if (
                    prepared_image.shape != (PROCESS_HEIGHT, PROCESS_WIDTH, 3)
                    or points[:68].shape != (68, 2)
                    or not np.isfinite(points[:68]).all()
                ):
                    raise ValueError('The prepared image or landmarks are invalid.')

                artifact_name = f'{uuid.uuid4().hex}.npz'
                artifact_path = os.path.join(generation_path, artifact_name)
                np.savez(
                    artifact_path,
                    image=np.asarray(prepared_image, dtype=np.uint8),
                    points=np.asarray(points[:68], dtype=np.int32),
                )
                entry.update({
                    'valid': True,
                    'artifact': f'{generation_name}/{artifact_name}',
                    'artifact_sha256': database_file_sha256(artifact_path),
                })
            except (OSError, ValueError, cv2.error) as error:
                entry['error'] = str(error)
                print(f'[SKIP] {key}: {error}')
            entries[key] = entry

        valid_count = sum(entries[key]['valid'] for key in eligible_keys)
        selected_counts[gender] = valid_count
        # 同性配对会为本人、伴侣分别抽取3个不同身份。
        if valid_count < 6:
            raise RuntimeError(
                f'database/{gender} has only {valid_count} eligible prepared faces; '
                'at least six are needed for same-gender pairs.'
            )

    manifest = {
        'version': DATABASE_PREPROCESS_VERSION,
        'width': PROCESS_WIDTH,
        'height': PROCESS_HEIGHT,
        'predictor_size': predictor_stat.st_size,
        'predictor_mtime_ns': predictor_stat.st_mtime_ns,
        'faces': entries,
    }
    temp_manifest = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', suffix='.json',
            prefix='manifest_', dir=PREPROCESSED_DATABASE_PATH,
            delete=False
        ) as output:
            temp_manifest = output.name
            json.dump(manifest, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_manifest, os.path.join(PREPROCESSED_DATABASE_PATH, 'manifest.json'))
    finally:
        if temp_manifest and os.path.exists(temp_manifest):
            os.remove(temp_manifest)

    print(
        '[OK] Database preprocessing published: '
        f"male {selected_counts['male']}, female {selected_counts['female']} "
        'eligible faces. Restart the backend before accepting participants.'
    )
    print(f'[OK] Manifest: {os.path.join(PREPROCESSED_DATABASE_PATH, "manifest.json")}')


if __name__ == '__main__':
    build_database_preprocessing()
