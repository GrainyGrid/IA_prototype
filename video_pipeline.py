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

DEFAULTS = dict(sharpness=12., min_size=38., max_yaw=.48, threshold=.60, margin=.04, min_samples=3)


def quality_metrics(frame, face):
    height, width = frame.shape[:2]
    x, y, w, h = map(float, face[:4])
    visible = x >= -2 and y >= -2 and x+w <= width + 2 and y+h <= height + 2
    if not visible or min(w, h) < 28:
        return None
    crop = frame[max(0, int(y)):min(height, int(y+h)), max(0, int(x)):min(width, int(x+w))]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    normalized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    eyes = face[4:8].reshape(2, 2)
    span = max(float(np.linalg.norm(eyes[1]-eyes[0])), 1.)
    direction = (eyes[np.argmax(eyes[:, 0])] - eyes[np.argmin(eyes[:, 0])]) / span
    yaw = abs(float((face[8:10] - eyes.mean(0)) @ direction) / span)
    raw_sharpness = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
    scale_factor = min(1.0, max(0.4, min(w, h) / 112.0))
    sharpness = raw_sharpness / scale_factor
    return dict(sharpness=sharpness,
                size=min(w, h), yaw=yaw, brightness=float(gray.mean()),
                confidence=float(face[14]), overexposed=float((gray > 250).mean()))


def rejection(metrics, settings):
    if metrics is None:
        return 'Move closer and keep your entire face visible.'
    if metrics['confidence'] < .65:
        return 'Detection confidence is low. Face the camera more directly.'
    if metrics['size'] < settings['min_size']:
        return 'Move closer; your face is too small.'
    if metrics['brightness'] < 18 or metrics['overexposed'] > .5:
        return 'Use even light in front of your face.'
    if metrics['yaw'] > settings['max_yaw']:
        return 'Turn your face toward the camera.'
    if metrics['sharpness'] < settings['sharpness']:
        return f"Image is too soft (sharpness {metrics['sharpness']:.1f}; minimum {settings['sharpness']:g})."
    return None


def quality_weight(metrics):
    fn_weight = min(max(float(metrics.get('feature_norm', 20.0)), 1.0) / 22.0, 1.0)
    return max(.01, min(metrics['sharpness']/100, 1) * min(metrics['size']/180, 1) * max(.1, 1-metrics['yaw']) * fn_weight)


def decision(samples, people, settings, now, active_ble_ids=None):
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
    probe_mean = mean / norm

    # Multi-pose template scoring across full enrolled people gallery
    scores = np.empty(len(people), dtype=np.float32)
    individual = np.empty((len(usable), len(people)), dtype=np.float32)
    for p_idx, p in enumerate(people):
        refs = np.atleast_2d(p.get('pose_references', p['reference']))
        scores[p_idx] = float(np.max(refs @ probe_mean))
        individual[:, p_idx] = np.max(vectors @ refs.T, axis=1)

    best = int(np.argmax(scores))
    score = float(scores[best])
    runner_up = float(np.partition(scores, -2)[-2]) if len(scores) > 1 else -1.
    agreeing = int(((individual.argmax(1) == best) & (individual[:, best] >= settings['threshold'])).sum())

    best_person = people[best]
    tag_id = (best_person.get('ble_tag_id') or '').strip().lower()
    if active_ble_ids is None:
        ble_ok = True  # Backward compatibility for unit tests that omit active_ble_ids
    else:
        ble_ok = bool(tag_id) and (tag_id in active_ble_ids)

    confirmed = bool(
        len(usable) >= settings['min_samples']
        and score >= settings['threshold']
        and score - runner_up >= settings['margin']
        and agreeing >= settings['min_samples']
        and ble_ok
    )
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
        if 'pose_references' in person and person['pose_references'] is not None:
            digest.update(np.asarray(person['pose_references'], dtype=np.float32).tobytes())
        if person.get('ble_tag_id'):
            digest.update(person['ble_tag_id'].encode())
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
        self.confirmed_person_tracks = {}

    def reset(self):
        if self.tracker is not None:
            self.tracker.reset()
        self.samples.clear(); self.last_seen.clear(); self.logged.clear()
        self.confirmed_person_tracks.clear()
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
            self.tracker = sv.ByteTrack(track_activation_threshold=.55, lost_track_buffer=45,
                                        frame_rate=20, minimum_consecutive_frames=1)
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

    def observe(self, track, embedding, quality, people, now, record=True, active_ble_ids=None):
        signature = gallery_signature(people)
        if self.gallery != signature:
            self.samples.clear(); self.logged.clear(); self.gallery = signature
            self.confirmed_person_tracks.clear()
        samples = self.samples[track]
        # Never aggregate obviously different people after a tracker ID switch.
        usable = embedding is not None and rejection(quality, self.settings) is None
        if usable and samples and float(samples[-1]['embedding'] @ embedding) < .3:
            samples.clear()
        if usable and (not samples or now-samples[-1]['time'] >= .15):
            samples.append(dict(embedding=embedding.copy(), quality=quality, time=now))
        result = decision(samples, people, self.settings, now, active_ble_ids=active_ble_ids)
        if samples and now-samples[-1]['time'] > 1.15:
            result = dict(person_id=None, candidate_id=None, similarity=None, sample_count=0, confirmed=False)

        # 1-to-1 Track-to-Tag Exclusivity Resolution
        if result['confirmed'] and result['person_id']:
            pid = result['person_id']
            prev_track, prev_score = self.confirmed_person_tracks.get(pid, (None, -1.0))
            if prev_track is not None and prev_track != track and (now - self.last_seen.get(prev_track, 0) < 1.0):
                if result['similarity'] <= prev_score:
                    # Demote this track: tag already bound to higher-scoring track
                    result['confirmed'] = False
                    result['person_id'] = None
                else:
                    # This track is higher scoring; replace binding
                    self.confirmed_person_tracks[pid] = (track, result['similarity'])
            else:
                self.confirmed_person_tracks[pid] = (track, result['similarity'])

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
