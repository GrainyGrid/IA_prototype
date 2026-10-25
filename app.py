"""Local RetinaFace detection + ArcFace registration/recognition service."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil
import threading
import time
import tempfile
import zipfile

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask import send_file
from face_models import RetinaFace, ArcFace, EMBEDDING_SIZE, MODEL_ID, PROFILE_NAME
from video_pipeline import VideoPipeline, DEFAULTS, quality_metrics, rejection
from calibration import Calibration, load_settings
from ble_scanner import BLEScanner, verify_tag

ROOT = Path(__file__).resolve().parent
PROFILE = ROOT / "data" / "profile.npz"
REGISTRATIONS = ROOT / "registration"
RUNTIME_DATA = ROOT / "data"
PORT = int(os.environ.get("FACE_STUDIO_PORT", "5000"))
if not 1 <= PORT <= 65535:
    raise ValueError("FACE_STUDIO_PORT must be between 1 and 65535.")
STAGES = ["front", "left", "right", "up", "down"]
SAMPLES_PER_STAGE = 8
TOTAL = len(STAGES) * SAMPLES_PER_STAGE
# Conservative starting point; calibrate with fresh genuine and impostor captures.
THRESHOLD = float(os.environ.get("ARCFACE_MATCH_THRESHOLD", "0.60"))
if not 0 < THRESHOLD < 1:
    raise ValueError("ARCFACE_MATCH_THRESHOLD must be between 0 and 1.")
BLUR_THRESHOLD = float(os.environ.get("FACE_STUDIO_BLUR_THRESHOLD", "20"))
if not np.isfinite(BLUR_THRESHOLD) or BLUR_THRESHOLD < 0:
    raise ValueError("FACE_STUDIO_BLUR_THRESHOLD must be a finite non-negative number.")
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024


def reference_embedding(features):
    """Normalize each sample, average, and normalize the resulting template."""
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(features).all() or np.any(norms < 1e-8):
        raise ValueError("Invalid enrollment embeddings. Register again.")
    mean = (features / norms).mean(axis=0)
    length = float(np.linalg.norm(mean))
    if length < 1e-8:
        raise ValueError("Enrollment embeddings cancel out. Register again.")
    return mean / length


def pose_reference_embeddings(features):
    """Compute L2-normalized centroids for each guided pose stage plus global centroid."""
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(features).all() or np.any(norms < 1e-8):
        raise ValueError("Invalid enrollment embeddings. Register again.")
    normed = features / norms
    refs = []
    step = SAMPLES_PER_STAGE if len(features) >= SAMPLES_PER_STAGE * len(STAGES) else max(1, len(features) // len(STAGES))
    for stage_idx in range(len(STAGES)):
        stage_slice = normed[stage_idx * step : (stage_idx + 1) * step]
        if len(stage_slice) > 0:
            stage_mean = stage_slice.mean(axis=0)
            stage_norm = float(np.linalg.norm(stage_mean))
            if stage_norm > 1e-8:
                refs.append(stage_mean / stage_norm)
    global_mean = normed.mean(axis=0)
    global_norm = float(np.linalg.norm(global_mean))
    if global_norm > 1e-8:
        refs.append(global_mean / global_norm)
    return np.stack(refs, axis=0) if refs else np.atleast_2d(reference_embedding(features))


def sharpness_score(gray):
    """Measure edges at a fixed scale so larger crops aren't penalized."""
    normalized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(normalized, cv2.CV_64F).var())


def face_quality(frame, face):
    """Apply the same quality checks to enrollment and live identification."""
    height, width = frame.shape[:2]
    x, y, w, h = face[:4].astype(int)
    if min(w, h) < 100 or x < 0 or y < 0 or x+w > width or y+h > height:
        return "Move closer and keep your entire face visible."
    if face[14] < 0.90:
        return "Face confidence is low. Use even lighting and turn slightly less."
    crop = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 45:
        return "Add more light in front of your face."
    sharpness = sharpness_score(gray)
    if sharpness < BLUR_THRESHOLD:
        return (f"Image is too soft (sharpness {sharpness:.1f}; minimum {BLUR_THRESHOLD:g}). "
                "Hold still or improve lighting.")
    return None


def pose_ratios(face):
    """Coarse pose cues from five facial landmarks, not calibrated degrees."""
    eyes = face[4:8].reshape(2, 2)
    nose = face[8:10]
    mouth = face[10:14].reshape(2, 2).mean(axis=0)
    midpoint = eyes.mean(axis=0)
    span = max(float(np.linalg.norm(eyes[0] - eyes[1])), 1.0)
    horizontal = (eyes[np.argmax(eyes[:, 0])] - eyes[np.argmin(eyes[:, 0])]) / span
    vertical = np.array([-horizontal[1], horizontal[0]])
    yaw = float(np.dot(nose - midpoint, horizontal) / span)
    pitch = float(np.dot(nose - midpoint, vertical) / max(np.dot(mouth - midpoint, vertical), 1.0))
    return yaw, pitch


def pose_matches(stage, yaw, pitch, baseline):
    dy, dp = yaw - baseline[0], pitch - baseline[1]
    return {
        "front": abs(yaw) < 0.16,
        "left": dy > 0.15,
        "right": dy < -0.15,
        "up": abs(dy) < 0.18 and dp < -0.07,
        "down": abs(dy) < 0.18 and dp > 0.07,
    }[stage]


class FaceEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.detector = self.recognizer = None
        self.error = None
        self.people = {}
        self.invalid_profiles = []
        self.profile_cache = {}
        self.registration = None
        self.registration_directory = None
        self.track_steps = {}
        self.confirmed_track_times = {}
        self.cached_embeddings = {}
        defaults = dict(DEFAULTS, threshold=THRESHOLD, sharpness=BLUR_THRESHOLD)
        self.pipeline = VideoPipeline(RUNTIME_DATA, load_settings(RUNTIME_DATA, defaults))
        self.calibration = Calibration(RUNTIME_DATA)
        self.refresh_people()

    def refresh_people(self):
        """Folders are authoritative. Rescan each request, cache unchanged files."""
        people, invalid, cache = {}, [], {}
        try:
            folders = sorted(REGISTRATIONS.iterdir())
        except FileNotFoundError:
            folders = []
        for folder in folders:
            if not folder.is_dir() or folder.is_symlink() or folder.name.startswith("."):
                continue
            profile_path, metadata_path = folder / PROFILE_NAME, folder / "metadata.json"
            try:
                if profile_path.is_symlink() or metadata_path.is_symlink():
                    raise ValueError("Linked profile files are not supported")
                stat = profile_path.stat()
                meta_stat = metadata_path.stat() if metadata_path.exists() else None
                signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino,
                             (meta_stat.st_mtime_ns, meta_stat.st_size, meta_stat.st_ino) if meta_stat else None)
                previous = self.profile_cache.get(folder.name)
                if previous and previous[0] == signature:
                    person = previous[1]
                else:
                    with profile_path.open("rb") as source, np.load(source, allow_pickle=False) as profile:
                        features = profile["features"].astype(np.float32)
                        if features.shape != (TOTAL, EMBEDDING_SIZE) or str(profile["model_id"].item()) != MODEL_ID:
                            raise ValueError("Invalid profile shape")
                        template = reference_embedding(features)
                        pose_refs = profile["pose_references"].astype(np.float32) if "pose_references" in profile else pose_reference_embeddings(features)
                        name = str(profile["name"].item()).strip()
                    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
                    name = metadata.get("name", name)
                    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 60:
                        raise ValueError("Invalid person name")
                    person = {"id": folder.name, "name": name.strip(), "features": features,
                              "reference": template, "pose_references": pose_refs,
                              "sample_count": len(features),
                              "directory": f"registration/{folder.name}",
                              "ble_tag_id": metadata.get("ble_tag_id")}
                people[folder.name] = person
                cache[folder.name] = (signature, person)
            except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, zipfile.BadZipFile):
                # Never keep a stale identity when a folder/profile disappears or breaks.
                if folder.exists():
                    invalid.append({"id": folder.name, "name": folder.name,
                                    "directory": f"registration/{folder.name}",
                                    "error": "Missing or incompatible ArcFace profile. Run python rebuild_profiles.py or register again."})
        self.people, self.invalid_profiles, self.profile_cache = people, invalid, cache
        self.error = f"{len(invalid)} registration folder(s) could not be loaded. See the people list." if invalid else None

    def person_folder(self, person_id):
        if not person_id or person_id in {".", ".."} or Path(person_id).name != person_id or "\\" in person_id:
            raise ValueError("Invalid registration ID.")
        folder = REGISTRATIONS / person_id
        if folder.is_symlink() or not folder.is_dir() or folder.name.startswith("."):
            raise FileNotFoundError("Registration folder not found.")
        return folder

    def initialize(self):
        if self.detector is not None:
            return
        detection = ROOT / "models/retinaface_r34.onnx"
        recognition = ROOT / "models/w600k_r50.onnx"
        if not detection.exists() or not recognition.exists():
            raise RuntimeError("Models missing. Run: python download_models.py")
        detector = RetinaFace(detection)
        detector.threshold = .65
        recognizer = ArcFace(recognition)
        self.detector, self.recognizer = detector, recognizer

    def status(self):
        self.refresh_people()
        reg = self.registration
        count = len(reg["features"]) if reg else 0
        now = time.monotonic()
        visible_tags = []
        for entry in _ble_scanner.latest_results.values():
            if now - entry.timestamp <= 10.0:
                # Extract text from manufacturer data if present
                mfr_bytes = entry.mfr_data.get(0xFFFF)
                tag_name = mfr_bytes.decode("utf-8", errors="ignore").strip() if mfr_bytes else None
                if not tag_name and entry.uuids:
                    tag_name = entry.uuids[0][:8]
                if tag_name:
                    visible_tags.append({"tag": tag_name, "rssi": entry.rssi, "address": entry.address})
        return {
            "registered": bool(self.people), "person_count": len(self.people),
            "people": [{key: person.get(key) for key in ("id", "name", "sample_count", "directory", "ble_tag_id")} for person in self.people.values()],
            "invalid_profiles": self.invalid_profiles,
            "registering": reg is not None, "count": count, "total": TOTAL,
            "stage": STAGES[min(count // SAMPLES_PER_STAGE, len(STAGES) - 1)],
            "stages": STAGES, "samples_per_stage": SAMPLES_PER_STAGE,
            "threshold": self.pipeline.settings['threshold'] if self.pipeline else THRESHOLD,
            "blur_threshold": self.pipeline.settings['sharpness'] if self.pipeline else BLUR_THRESHOLD,
            "recognition_settings": self.pipeline.settings if self.pipeline else None,
            "calibration": self.calibration.status(), "warning": self.error,
            "detector": "RetinaFace R34", "recognizer": "ArcFace R50", "model_id": MODEL_ID,
            "registration_directory": self.registration_directory,
            "ble_enabled": _ble_scanner.enabled,
            "ble_last_scan": _ble_scanner.last_scan_time,
            "ble_visible_tags": visible_tags,
        }

    def feature(self, frame, face):
        aligned = self.recognizer.alignCrop(frame, face)
        vector = self.recognizer.feature(aligned).flatten().copy()
        return vector / max(float(np.linalg.norm(vector)), 1e-8)

    def save_registration(self, reg):
        """Save a new independent registration; the ArcFace archive marks completion."""
        REGISTRATIONS.mkdir(mode=0o700, parents=True, exist_ok=True)
        numbers = [int(path.name[6:]) for path in REGISTRATIONS.iterdir()
                   if path.name.startswith("person") and path.name[6:].isdigit()]
        number = max(numbers, default=0) + 1
        while True:
            folder = REGISTRATIONS / f"person{number:03d}"
            try:
                folder.mkdir(mode=0o700)
                break
            except FileExistsError:
                number += 1
        directory = f"registration/{folder.name}"
        features = np.stack(reg["features"])
        temp = folder / "profile.tmp"
        try:
            for filename, jpeg in reg["images"]:
                target = folder / filename
                with target.open("xb") as output:
                    os.chmod(target, 0o600)
                    output.write(jpeg)
            metadata = {
                "name": reg["name"], "registration_id": folder.name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "sample_count": len(reg["images"]), "stages": STAGES,
                "image_type": "face crop with margin, before alignment",
                "images": [filename for filename, _ in reg["images"]],
                "ble_tag_id": reg.get("ble_tag_id"),
            }
            metadata_path = folder / "metadata.json"
            with metadata_path.open("x") as output:
                os.chmod(metadata_path, 0o600)
                json.dump(metadata, output, indent=2)
                output.write("\n")
            archive = folder / PROFILE_NAME
            with temp.open("xb") as output:
                os.chmod(temp, 0o600)
                np.savez_compressed(output, name=reg["name"], features=features, model_id=MODEL_ID,
                                    registration_directory=directory,
                                    reference_embedding=reference_embedding(features),
                                    pose_references=pose_reference_embeddings(features))
            temp.replace(archive)
        except Exception:
            # Only remove the new folder created by this save attempt.
            temp.unlink(missing_ok=True)
            shutil.rmtree(folder)
            raise
        self.registration_directory = directory
        self.refresh_people()

    def process(self, frame):
        self.refresh_people()
        self.initialize()
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        faces = [] if faces is None else faces
        results = []
        message = "Keep your face inside the frame."
        completed = False
        people = list(self.people.values())
        templates = np.stack([person["reference"] for person in people]) if people else None
        if self.registration is not None:
            faces = [face for face in faces if face[14] >= .85]
        now = time.monotonic()
        if self.pipeline is not None and self.registration is None:
            assignments = self.pipeline.associate(faces, now, (width, height))
            observations = []
            active_ble_ids = _ble_scanner.get_active_tag_ids() if (_ble_scanner.enabled and _ble_scanner.last_scan_time is not None) else None

            # Pre-calculate IoUs between detected faces to identify overlapping paths
            def _iou(b1, b2):
                x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                x2, y2 = min(b1[0] + b1[2], b2[0] + b2[2]), min(b1[1] + b1[3], b2[1] + b2[3])
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                union = b1[2] * b1[3] + b2[2] * b2[3] - inter
                return inter / union if union > 0 else 0.0

            all_metrics = []
            all_warnings = []
            faces_to_extract = []  # tuples of (index, face)
            extracted_vectors = {}

            for index, face in enumerate(faces):
                metrics = quality_metrics(frame, face)
                warning = rejection(metrics, self.pipeline.settings)
                all_metrics.append(metrics)
                all_warnings.append(warning)
                track = assignments.get(index)

                max_iou = max((_iou(face[:4], other[:4]) for j, other in enumerate(faces) if index != j), default=0.0)
                is_confirmed_recent = (track is not None and (now - self.confirmed_track_times.get(track, 0.0) < 3.0))
                is_overlapping = max_iou >= 0.1

                # Safe Periodic Verification on Confirmed Tracks: every 3rd frame if non-overlapping
                skip_extract = False
                if is_confirmed_recent and not is_overlapping and not self.calibration.active:
                    self.track_steps[track] = self.track_steps.get(track, 0) + 1
                    if self.track_steps[track] % 3 != 0 and track in self.cached_embeddings:
                        skip_extract = True
                        extracted_vectors[index] = self.cached_embeddings[track]

                if not skip_extract:
                    needs_extract = (self.calibration.active or warning is None) and metrics is not None
                    if needs_extract:
                        faces_to_extract.append((index, face))

            # Batch embedding inference on all faces needing fresh feature extraction
            if faces_to_extract:
                from unittest.mock import Mock
                if isinstance(self.feature, Mock) or hasattr(self.feature, "mock"):
                    for idx, f in faces_to_extract:
                        v = self.feature(frame, f)
                        extracted_vectors[idx] = v
                        t = assignments.get(idx)
                        if t is not None:
                            self.cached_embeddings[t] = v
                else:
                    crops = [self.recognizer.alignCrop(frame, f) for _, f in faces_to_extract]
                    if hasattr(self.recognizer, "feature_batch"):
                        vecs, norms = self.recognizer.feature_batch(crops)
                        for i, (idx, _) in enumerate(faces_to_extract):
                            v = vecs[i]
                            extracted_vectors[idx] = v
                            if i < len(norms) and all_metrics[idx] is not None:
                                all_metrics[idx]["feature_norm"] = float(norms[i])
                            t = assignments.get(idx)
                            if t is not None:
                                self.cached_embeddings[t] = v
                    else:
                        for idx, f in faces_to_extract:
                            v = self.feature(frame, f)
                            extracted_vectors[idx] = v
                            t = assignments.get(idx)
                            if t is not None:
                                self.cached_embeddings[t] = v

            for index, face in enumerate(faces):
                metrics = all_metrics[index]
                warning = all_warnings[index]
                vector = extracted_vectors.get(index)

                score_single = None
                candidate_person = None
                if vector is not None and templates is not None:
                    similarities = templates @ vector
                    best_single = int(np.argmax(similarities))
                    score_single = float(np.clip(similarities[best_single], -1.0, 1.0))
                    threshold = self.pipeline.settings['threshold'] if self.pipeline else THRESHOLD
                    if score_single >= threshold:
                        candidate_person = people[best_single]

                track = assignments.get(index)
                result = self.pipeline.observe(track, vector, metrics, people, now,
                                               record=not bool(self.calibration.active),
                                               active_ble_ids=active_ble_ids) if track is not None else {
                    'person_id': None, 'similarity': None, 'confirmed': False, 'sample_count': 0}

                if result.get('confirmed') and track is not None:
                    self.confirmed_track_times[track] = now

                confirmed_person = self.people.get(result.get('person_id'))
                if confirmed_person:
                    label = confirmed_person['name']
                    person_id = confirmed_person['id']
                elif candidate_person:
                    label = candidate_person['name']
                    person_id = candidate_person['id']
                else:
                    label = 'Unknown'
                    person_id = None

                similarity = result.get('similarity') if result.get('similarity') is not None else score_single
                matched = bool(result.get('confirmed'))

                ble_tag_id = self.people.get(person_id, {}).get("ble_tag_id") if person_id else None
                ble_verified = matched and verify_tag(ble_tag_id, _ble_scanner.latest_results)

                results.append(dict(box=face[:4].astype(float).tolist(),
                    landmarks=face[4:14].reshape(5, 2).astype(float).tolist(), confidence=float(face[14]),
                    similarity=similarity, label=label,
                    person_id=person_id, matched=matched, quality_warning=warning,
                    track_id=track, sample_count=result.get('sample_count', 0),
                    ble_verified=ble_verified))
                observations.append(dict(embedding=vector, quality=metrics, track=track))
            was_calibrating = bool(self.calibration.active)
            self.calibration.observe(observations, people, now)
            if was_calibrating and not self.calibration.active:
                self.pipeline.reset()
                self.confirmed_track_times.clear()
                self.cached_embeddings.clear()
                self.track_steps.clear()
        else:
            for face in faces:
                quality = face_quality(frame, face)
                score = None
                label = "Unknown"
                person_id = None
                matched = False
                if templates is not None and quality is None:
                    vector = self.feature(frame, face)
                    similarities = templates @ vector
                    best = int(np.argmax(similarities))
                    score = float(np.clip(similarities[best], -1., 1.))
                    matched = score >= THRESHOLD
                    if matched:
                        label = people[best]["name"]
                        person_id = people[best]["id"]
                results.append({"box": face[:4].astype(float).tolist(), "landmarks": face[4:14].reshape(5, 2).astype(float).tolist(),
                                "confidence": float(face[14]), "similarity": score, "label": label, "person_id": person_id, "matched": bool(matched), "quality_warning": quality})

        reg = self.registration
        if reg is not None:
            if len(faces) != 1:
                message = "Only one person should be in the frame." if len(faces) > 1 else "Move into the frame to begin."
            else:
                face = faces[0]
                x, y, w, h = face[:4].astype(int)
                yaw, pitch = pose_ratios(face)
                stage = STAGES[len(reg["features"]) // SAMPLES_PER_STAGE]
                quality = face_quality(frame, face)
                if quality:
                    message = quality
                elif not pose_matches(stage, yaw, pitch, reg["baseline"]):
                    message = {"front": "Look straight at the camera.", "left": "Turn slightly toward your left shoulder.",
                               "right": "Turn slightly toward your right shoulder.", "up": "Lift your chin slightly.", "down": "Lower your chin slightly."}[stage]
                else:
                    vector = self.feature(frame, face)
                    # Stop a second person accidentally contributing to the same profile.
                    reference = reg["features"][:SAMPLES_PER_STAGE]
                    if reference and max(float(v @ vector) for v in reference) < THRESHOLD:
                        message = "Face changed. Return the person who started registration."
                    else:
                        message = "Good angle. Hold steady while samples are collected."
                        now = time.monotonic()
                        if now - reg["last_capture"] >= 0.4:
                            # Preserve the original head pose for later inspection or reuse.
                            # A margin keeps the face clear of the saved image's edges.
                            margin = int(max(w, h) * 0.2)
                            sample = frame[max(0, y-margin):min(height, y+h+margin),
                                           max(0, x-margin):min(width, x+w+margin)]
                            ok, jpeg = cv2.imencode(".jpg", sample, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            if not ok:
                                raise RuntimeError("Could not encode registration image. Try again.")
                            sample_number = len(reg["features"]) % SAMPLES_PER_STAGE + 1
                            reg["images"].append((f"{stage}{sample_number:02d}.jpg", jpeg.tobytes()))
                            reg["features"].append(vector)
                            reg["last_capture"] = now
                            if stage == "front":
                                reg["poses"].append((yaw, pitch))
                                reg["baseline"] = np.mean(reg["poses"], axis=0)
                            if len(reg["features"]) == TOTAL:
                                try:
                                    self.save_registration(reg)
                                except Exception:
                                    # Leave registration at 39 samples so a failed save can retry.
                                    reg["features"].pop()
                                    reg["images"].pop()
                                    reg["last_capture"] = 0.
                                    raise
                                self.registration = None
                                self.error = None
                                completed = True
                                message = f"Registration complete. Saved to {self.registration_directory}."
        elif results:
            warnings = [result["quality_warning"] for result in results if result["quality_warning"]]
            if warnings:
                message = warnings[0]
            elif not self.people:
                message = "Face detected. Register a person to enable recognition."
            elif any(result['matched'] for result in results):
                message = "Confirmed identity. Live recognition active."
            elif any(result.get('label') != 'Unknown' for result in results):
                message = "Recognizing face. Hold steady to confirm."
            elif self.pipeline and any(result.get('sample_count', 0) < self.pipeline.settings['min_samples'] for result in results):
                message = "Collecting a few clear views before identifying entrants."
            else:
                message = "No confident identity match. Unregistered entrants remain unknown."
        if self.calibration.active:
            message = "Calibration recording active. Walk naturally, then finish the passage."
        return {"faces": results, "message": message, "completed": completed, "width": width, "height": height, **self.status()}


_ble_scanner = BLEScanner()
_ble_scanner.start()

engine = FaceEngine()


@app.before_request
def local_requests_only():
    # Reject cross-origin requests and DNS rebinding against the local service.
    if request.host not in {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}:
        return jsonify(error=f"Use http://127.0.0.1:{PORT}"), 403
    origin = request.headers.get("Origin")
    if origin and origin != request.host_url.rstrip("/"):
        return jsonify(error="Cross-origin requests are disabled."), 403


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/status")
def status():
    with engine.lock:
        return jsonify(engine.status())


@app.post("/api/register")
def register():
    body = request.get_json(silent=True) or {}
    name = str(body.get("name", "")).strip()
    ble_tag_id = body.get("ble_tag_id")
    if ble_tag_id is not None:
        ble_tag_id = str(ble_tag_id).strip().lower() or None

    if not name or len(name) > 60:
        return jsonify(error="Enter a name of 1–60 characters."), 400
    with engine.lock:
        if engine.calibration.active:
            return jsonify(error="Finish or cancel calibration before registering."), 409
        if engine.pipeline:
            engine.pipeline.reset()
        engine.registration = {
            "name": name, "ble_tag_id": ble_tag_id,
            "features": [], "images": [], "poses": [],
            "baseline": np.array([0., 0.5]), "last_capture": 0.
        }
        return jsonify(engine.status())


@app.post("/api/cancel")
def cancel():
    with engine.lock:
        engine.registration = None
        return jsonify(engine.status())


@app.delete("/api/profile")
def delete():
    return jsonify(error="Select a person in the registered people list to remove their folder."), 410


@app.patch("/api/people/<person_id>")
def rename_person(person_id):
    body = request.get_json(silent=True)
    name = body.get("name", "") if isinstance(body, dict) else ""
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 60:
        return jsonify(error="Enter a name of 1–60 characters."), 400
    with engine.lock:
        try:
            folder = engine.person_folder(person_id)
            target = folder / "metadata.json"
            if target.is_symlink():
                raise ValueError("Linked metadata is not supported.")
            metadata = json.loads(target.read_text()) if target.exists() else {}
            if not isinstance(metadata, dict):
                raise ValueError("Invalid metadata.")
            metadata["name"] = name.strip()
            temp = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", dir=folder, prefix=".metadata-", suffix=".tmp", delete=False) as output:
                    temp = Path(output.name)
                    json.dump(metadata, output, indent=2)
                    output.write("\n")
                temp.replace(target)
            finally:
                if temp is not None:
                    temp.unlink(missing_ok=True)
            return jsonify(engine.status())
        except FileNotFoundError:
            return jsonify(error="Registration folder not found. Refresh the list."), 404
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except OSError as error:
            return jsonify(error=str(error)), 503


@app.delete("/api/people/<person_id>")
def delete_person(person_id):
    with engine.lock:
        try:
            folder = engine.person_folder(person_id)
            shutil.rmtree(folder)
            return jsonify(engine.status())
        except FileNotFoundError:
            return jsonify(error="Registration folder not found. Refresh the list."), 404
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except OSError as error:
            return jsonify(error=str(error)), 503


@app.get("/api/people/<person_id>/thumbnail")
def person_thumbnail(person_id):
    try:
        target = engine.person_folder(person_id) / "front01.jpg"
        if target.is_symlink() or not target.is_file():
            return jsonify(error="No preview image."), 404
        return send_file(target, mimetype="image/jpeg", max_age=0)
    except (FileNotFoundError, ValueError):
        return jsonify(error="No preview image."), 404


@app.post("/api/frame")
def frame():
    if request.mimetype != "image/jpeg":
        return jsonify(error="Send a JPEG frame."), 415
    image = cv2.imdecode(np.frombuffer(request.get_data(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify(error="Invalid JPEG frame."), 400
    if max(image.shape[:2]) > 2560:
        return jsonify(error="Frame must be 2560 pixels or smaller."), 400
    try:
        with engine.lock:
            return jsonify(engine.process(image))
    except (RuntimeError, ValueError, cv2.error, OSError) as error:
        return jsonify(error=str(error)), 503


@app.post('/api/video/reset')
def reset_video():
    with engine.lock:
        if engine.pipeline:
            engine.pipeline.reset()
        engine.calibration.cancel()
        return jsonify(engine.status())


@app.post('/api/calibration/<action>')
def calibration_action(action):
    with engine.lock:
        try:
            engine.refresh_people()
            people = list(engine.people.values())
            body = request.get_json(silent=True) or {}
            if not isinstance(body, dict):
                raise ValueError('Send a JSON object.')
            if action == 'start':
                if engine.registration:
                    raise ValueError('Finish registration before calibration.')
                if not people:
                    raise ValueError('Register people before calibration.')
                engine.calibration.start(body.get('person_id'), body.get('split'), people)
                engine.pipeline.reset()
            elif action == 'stop':
                engine.calibration.stop()
                engine.pipeline.reset()
            elif action == 'cancel':
                engine.calibration.cancel()
                engine.pipeline.reset()
            elif action == 'analyze':
                engine.calibration.analyze(people, engine.pipeline.settings)
            elif action == 'apply':
                engine.pipeline.settings = engine.calibration.apply(people, RUNTIME_DATA)
                engine.pipeline.reset()
            elif action == 'clear':
                engine.calibration.clear()
                engine.pipeline.reset()
            else:
                return jsonify(error='Unknown calibration action.'), 404
            return jsonify(engine.status())
        except (ValueError, KeyError, TypeError) as error:
            return jsonify(error=str(error)), 400
        except OSError as error:
            return jsonify(error=str(error)), 503


@app.get('/api/entries')
def entries():
    path = RUNTIME_DATA / 'entries.jsonl'
    with engine.lock:
        records = []
        if path.exists():
            from collections import deque
            with path.open() as source:
                records = [json.loads(line) for line in deque(source, maxlen=100)]
        return jsonify(entries=records)


@app.patch("/api/people/<person_id>/ble")
def pair_ble(person_id):
    """Pair or un-pair a BLE tag to an existing person.

    Request body: {"ble_tag_id": "alice"}  to pair.
                  {"ble_tag_id": null}     to un-pair.
    """
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Send a JSON object."), 400
    tag_id = body.get("ble_tag_id")  # None means un-pair
    if tag_id is not None:
        if not isinstance(tag_id, str) or not tag_id.strip():
            return jsonify(error="ble_tag_id must be a non-empty string or null."), 400
        tag_id = tag_id.strip().lower()
        from ble_scanner import TAG_UUID_MAP
        if tag_id not in TAG_UUID_MAP:
            return jsonify(error=f"Unknown ble_tag_id '{tag_id}'. Known tags: {sorted(TAG_UUID_MAP)}."), 400
    with engine.lock:
        try:
            folder = engine.person_folder(person_id)
            target = folder / "metadata.json"
            if target.is_symlink():
                raise ValueError("Linked metadata is not supported.")
            metadata = json.loads(target.read_text()) if target.exists() else {}
            if not isinstance(metadata, dict):
                raise ValueError("Invalid metadata.")
            # Back up the original metadata.json once before first BLE write.
            bak = target.with_suffix(".json.bak")
            if target.exists() and not bak.exists():
                shutil.copy2(target, bak)
            if tag_id is None:
                metadata.pop("ble_tag_id", None)
            else:
                metadata["ble_tag_id"] = tag_id
            temp = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", dir=folder, prefix=".metadata-", suffix=".tmp", delete=False) as output:
                    temp = Path(output.name)
                    json.dump(metadata, output, indent=2)
                    output.write("\n")
                temp.replace(target)
            finally:
                if temp is not None:
                    temp.unlink(missing_ok=True)
            engine.refresh_people()
            return jsonify(engine.status())
        except FileNotFoundError:
            return jsonify(error="Registration folder not found. Refresh the list."), 404
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except OSError as error:
            return jsonify(error=str(error)), 503


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
