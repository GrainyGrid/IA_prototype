"""RetinaFace R34 and InsightFace ArcFace R50 ONNX inference.

Decoding follows yakhyo/retinaface-pytorch; recognition preprocessing follows
InsightFace's w600k_r50 model. No PyTorch or InsightFace package is required.
"""
import os
from pathlib import Path

import cv2
import numpy as np

MODEL_ID = 'arcface-w600k-r50-v1'
PROFILE_NAME = 'profile_arcface.npz'
EMBEDDING_SIZE = 512
ARCFACE_POINTS = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)


def session(path):
    try:
        import sys
        from pathlib import Path
        if sys.platform == "win32":
            nv_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
            if nv_root.exists():
                for bin_dir in nv_root.glob("*/bin"):
                    if bin_dir.is_dir():
                        os.add_dll_directory(str(bin_dir))
                        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
    except ImportError as error:
        raise RuntimeError('Install dependencies: python -m pip install -r requirements.txt') from error
    provider = os.environ.get('FACE_STUDIO_PROVIDER', 'CPUExecutionProvider').strip()
    if provider not in {'CPUExecutionProvider', 'CoreMLExecutionProvider', 'CUDAExecutionProvider', 'TensorrtExecutionProvider'} or provider not in ort.get_available_providers():
        raise RuntimeError(f'Unavailable inference provider: {provider}')
    options = ort.SessionOptions()
    cpu_count = os.cpu_count() or 4
    threads = int(os.environ.get('FACE_STUDIO_THREADS', min(cpu_count, 6)))
    options.intra_op_num_threads = max(1, threads)
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [provider] if provider == 'CPUExecutionProvider' else [provider, 'CPUExecutionProvider']
    return ort.InferenceSession(str(path), sess_options=options, providers=providers)


def alignment_matrix(points):
    """Least-squares similarity transform, without random RANSAC alignment."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise RuntimeError('Invalid facial landmarks.')
    target = ARCFACE_POINTS.astype(np.float64)
    source_center, target_center = points.mean(0), target.mean(0)
    source, dest = points - source_center, target - target_center
    variance = np.sum(source ** 2) / len(points)
    if variance < 1e-8:
        raise RuntimeError('Degenerate facial landmarks.')
    u, singular, vt = np.linalg.svd(dest.T @ source / len(points))
    signs = np.ones(2)
    if np.linalg.det(u @ vt) < 0:
        signs[-1] = -1
    rotation = u @ np.diag(signs) @ vt
    scale = np.dot(singular, signs) / variance
    matrix = np.empty((2, 3), dtype=np.float32)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = target_center - scale * rotation @ source_center
    return matrix


class RetinaFace:
    def __init__(self, path):
        self.session = session(path)
        self.input_name = self.session.get_inputs()[0].name
        self.size = 640
        self.priors = self.make_priors(self.size)
        self.threshold = .85

    @staticmethod
    def make_priors(size):
        priors = []
        for step, sizes in zip([8, 16, 32], [[16, 32], [64, 128], [256, 512]]):
            for y in range((size + step - 1) // step):
                for x in range((size + step - 1) // step):
                    for minimum in sizes:
                        priors.append([(x + .5) * step / size, (y + .5) * step / size,
                                       minimum / size, minimum / size])
        return np.asarray(priors, dtype=np.float32)

    def setInputSize(self, size):
        # Adapter for the existing service. Inference always uses a 640px square.
        pass

    def detect(self, frame):
        height, width = frame.shape[:2]
        scale = min(self.size / width, self.size / height)
        resized_width, resized_height = round(width * scale), round(height * scale)
        canvas = np.full((self.size, self.size, 3), (104, 117, 123), dtype=np.uint8)
        canvas[:resized_height, :resized_width] = cv2.resize(frame, (resized_width, resized_height))
        blob = (canvas.astype(np.float32) - np.array([104, 117, 123], np.float32)).transpose(2, 0, 1)[None]
        try:
            loc, conf, landmarks = self.session.run(None, {self.input_name: blob})
        except Exception as error:
            raise RuntimeError(f'RetinaFace inference failed: {error}') from error
        loc, conf, landmarks = loc[0], conf[0], landmarks[0]
        if loc.shape != (len(self.priors), 4) or landmarks.shape != (len(self.priors), 10):
            raise RuntimeError('Unsupported RetinaFace model outputs; use the bundled R34 checkpoint.')
        selected = np.flatnonzero(conf[:, 1] >= self.threshold)
        selected = selected[np.argsort(conf[selected, 1])[::-1][:5000]]
        if not len(selected):
            return None, None
        prior, offsets = self.priors[selected], loc[selected]
        centers = prior[:, :2] + offsets[:, :2] * .1 * prior[:, 2:]
        sizes = prior[:, 2:] * np.exp(offsets[:, 2:] * .2)
        # Use actual rounded resize ratios to map coordinates back to the frame.
        ratios = np.array([width / resized_width, height / resized_height], np.float32)
        boxes = np.concatenate([centers - sizes / 2, centers + sizes / 2], axis=1) * self.size
        boxes *= np.tile(ratios, 2)
        points = prior[:, None, :2] + landmarks[selected].reshape(-1, 5, 2) * .1 * prior[:, None, 2:]
        points *= self.size * ratios
        boxes[:, 2:] -= boxes[:, :2]
        scores = conf[selected, 1]
        keep = np.asarray(cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.threshold, .4)).reshape(-1)
        faces = np.concatenate([boxes[keep], points[keep].reshape(-1, 10), scores[keep, None]], axis=1)
        return None, faces.astype(np.float32)


class ArcFace:
    def __init__(self, path):
        self.session = session(path)
        self.input_name = self.session.get_inputs()[0].name

    def alignCrop(self, frame, face):
        matrix = alignment_matrix(face[4:14].reshape(5, 2))
        return cv2.warpAffine(frame, matrix, (112, 112), borderValue=0)

    def feature(self, aligned):
        blob = cv2.dnn.blobFromImage(aligned, 1 / 127.5, (112, 112),
                                     (127.5, 127.5, 127.5), swapRB=True)
        try:
            vector = self.session.run(None, {self.input_name: blob})[0].reshape(-1)
        except Exception as error:
            raise RuntimeError(f'ArcFace inference failed: {error}') from error
        length = np.linalg.norm(vector)
        if vector.shape != (EMBEDDING_SIZE,) or not np.isfinite(vector).all() or length < 1e-8:
            raise RuntimeError('Invalid ArcFace embedding; use the bundled w600k_r50 checkpoint.')
        return vector / length
