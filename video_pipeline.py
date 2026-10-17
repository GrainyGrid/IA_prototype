"""Quality selection, ByteTrack association and shared live/calibration decisions."""
from collections import defaultdict, deque
from datetime import datetime, timezone
import atexit
import hashlib
import json
import os
import tempfile
import uuid

import cv2
import numpy as np

_font_cache = None

DEFAULTS = dict(sharpness=20., min_size=100., max_yaw=.35, threshold=.60, margin=.04, min_samples=3)


def quality_metrics(frame, face):
    height, width = frame.shape[:2]
    x, y, w, h = map(float, face[:4])
    visible = x >= 0 and y >= 0 and x+w <= width and y+h <= height
    if not visible or min(w, h) < 50:
        return None
    crop = frame[int(y):int(y+h), int(x):int(x+w)]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    normalized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    eyes = face[4:8].reshape(2, 2)
    span = max(float(np.linalg.norm(eyes[1]-eyes[0])), 1.)
    direction = (eyes[np.argmax(eyes[:, 0])] - eyes[np.argmin(eyes[:, 0])]) / span
    yaw = abs(float((face[8:10] - eyes.mean(0)) @ direction) / span)
    return dict(sharpness=float(cv2.Laplacian(normalized, cv2.CV_64F).var()),
                size=min(w, h), yaw=yaw, brightness=float(gray.mean()),
                confidence=float(face[14]), overexposed=float((gray > 250).mean()))


def rejection(metrics, settings):
    if metrics is None:
        return 'Move closer and keep your entire face visible.'
    if metrics['confidence'] < .8:
        return 'Detection confidence is low. Face the camera more directly.'
    if metrics['size'] < settings['min_size']:
        return 'Move closer; your face is too small.'
    if metrics['brightness'] < 30 or metrics['overexposed'] > .5:
        return 'Use even light in front of your face.'
    if metrics['yaw'] > settings['max_yaw']:
        return 'Turn your face toward the camera.'
    if metrics['sharpness'] < settings['sharpness']:
        return f"Image is too soft (sharpness {metrics['sharpness']:.1f}; minimum {settings['sharpness']:g})."
    return None


def quality_weight(metrics):
    return max(.01, min(metrics['sharpness']/100, 1) * min(metrics['size']/180, 1) * max(.1, 1-metrics['yaw']))


def decision(samples, people, settings, now):
    """Select diverse recent views, aggregate and demand agreeing individual matches."""
    usable = [s for s in samples if 0 <= now-s['time'] <= 6 and rejection(s['quality'], settings) is None]
    usable = sorted(usable, key=lambda s: quality_weight(s['quality']), reverse=True)[:8]
    empty = dict(person_id=None, candidate_id=None, similarity=None, sample_count=len(usable), confirmed=False)
    if not people or not usable:
        return empty
    vectors = np.stack([s['embedding'] for s in usable])
    weights = np.array([quality_weight(s['quality']) for s in usable])
    mean = (vectors * weights[:, None]).sum(0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-8:
        return empty
    templates = np.stack([p['reference'] for p in people])
    scores = templates @ (mean / norm)
    best = int(np.argmax(scores))
    score = float(scores[best])
    runner_up = float(np.partition(scores, -2)[-2]) if len(scores) > 1 else -1.
    individual = vectors @ templates.T
    agreeing = ((individual.argmax(1) == best) & (individual[:, best] >= settings['threshold'])).sum()
    confirmed = (len(usable) >= settings['min_samples']
                 and score >= settings['threshold']
                 and score-runner_up >= settings['margin']
                 and agreeing >= settings['min_samples'])
    return dict(person_id=people[best]['id'] if confirmed else None,
                candidate_id=people[best]['id'] if score >= settings['threshold'] else None,
                similarity=score,
                sample_count=len(usable),
                confirmed=bool(confirmed))


def gallery_signature(people):
    digest = hashlib.sha256()
    for person in sorted(people, key=lambda p: p['id']):
        digest.update(person['id'].encode())
        digest.update(np.asarray(person['reference'], dtype=np.float32).tobytes())
    return digest.hexdigest()


class VideoPipeline:
    def __init__(self, directory, settings):
        self.directory = directory
        self.settings = dict(settings)
        self.tracker = None
        self.samples = defaultdict(lambda: deque(maxlen=24))
        self.last_seen = {}
        self.logged = set()
        self.session_id = uuid.uuid4().hex
        self.gallery = None
        self.last_time = None
        self.frame_size = None

    def reset(self):
        if self.tracker is not None:
            self.tracker.reset()
        self.samples.clear(); self.last_seen.clear(); self.logged.clear()
        self.session_id = uuid.uuid4().hex
        self.last_time = None

    def associate(self, faces, now, frame_size):
        if self.tracker is None:
            # Supervision imports matplotlib; keep its disposable font cache local.
            global _font_cache
            if _font_cache is None:
                _font_cache = tempfile.TemporaryDirectory(prefix='face-studio-fonts-')
                atexit.register(_font_cache.cleanup)
            os.environ.setdefault('MPLCONFIGDIR', _font_cache.name)
            import supervision as sv
            self.sv = sv
            self.tracker = sv.ByteTrack(track_activation_threshold=.85, lost_track_buffer=45,
                                        frame_rate=7, minimum_consecutive_frames=1)
        if self.last_time is not None and (now-self.last_time > 2.5 or frame_size != self.frame_size):
            self.reset()
        self.last_time, self.frame_size = now, frame_size
        boxes = np.asarray([f[:4] for f in faces], dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        detections = self.sv.Detections(xyxy=boxes,
                                      confidence=np.asarray([f[14] for f in faces], dtype=np.float32),
                                      class_id=np.zeros(len(faces), dtype=int),
                                      data={'index': np.arange(len(faces))})
        tracked = self.tracker.update_with_detections(detections)
        assignments = {int(index): int(track) for index, track in zip(tracked.data.get('index', []), tracked.tracker_id)}
        for index, track in assignments.items():
            self.last_seen[track] = now
        for track in list(self.last_seen):
            if now-self.last_seen[track] > 2.5:
                self.samples.pop(track, None); self.last_seen.pop(track, None)
        return assignments

    def observe(self, track, embedding, quality, people, now, record=True):
        signature = gallery_signature(people)
        if self.gallery != signature:
            self.samples.clear(); self.logged.clear(); self.gallery = signature
        samples = self.samples[track]
        # Never aggregate obviously different people after a tracker ID switch.
        usable = embedding is not None and rejection(quality, self.settings) is None
        if usable and samples and float(samples[-1]['embedding'] @ embedding) < .3:
            samples.clear()
        if usable and (not samples or now-samples[-1]['time'] >= .15):
            samples.append(dict(embedding=embedding.copy(), quality=quality, time=now))
        result = decision(samples, people, self.settings, now)
        if samples and now-samples[-1]['time'] > 1.15:
            result = dict(person_id=None, candidate_id=None, similarity=None, sample_count=0, confirmed=False)
        if result['confirmed'] and record and track not in self.logged:
            person = next(p for p in people if p['id'] == result['person_id'])
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            path = self.directory / 'entries.jsonl'
            event = dict(timestamp=datetime.now(timezone.utc).isoformat(), session_id=self.session_id,
                         track_id=track, person_id=person['id'], name=person['name'],
                         similarity=result['similarity'])
            with path.open('a') as output:
                os.chmod(path, 0o600)
                output.write(json.dumps(event)+'\n')
            self.logged.add(track)
        return result
