"""Labeled walking passages and held-out evaluation, using the live decision rule."""
from itertools import product
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np

from face_models import MODEL_ID
from video_pipeline import DEFAULTS, decision, gallery_signature, rejection


def atomic_json(path, value):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            os.chmod(temporary, 0o600)
            json.dump(value, output, allow_nan=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_settings(directory, defaults):
    path = directory / 'recognition_settings.json'
    if not path.exists():
        return dict(defaults)
    value = json.loads(path.read_text())
    settings = value['settings']
    if value['model_id'] != MODEL_ID or set(settings) != set(DEFAULTS):
        raise ValueError('Incompatible recognition settings.')
    if any(not isinstance(v, (float, int)) or not np.isfinite(v) for v in settings.values()):
        raise ValueError('Invalid recognition settings.')
    if not (0 <= settings['sharpness'] <= 10000 and 50 <= settings['min_size'] <= 500
            and 0 < settings['max_yaw'] <= 1 and 0 < settings['threshold'] < 1
            and 0 <= settings['margin'] < 1 and settings['min_samples'] in range(1, 9)):
        raise ValueError('Recognition settings out of range.')
    return settings


class Calibration:
    def __init__(self, directory):
        self.directory = directory / 'calibration'
        self.active = None
        self.report = None

    def takes(self):
        result = []
        if self.directory.exists():
            for path in sorted(self.directory.glob('*.json')):
                result.append(json.loads(path.read_text()))
        return result

    def status(self):
        active = self.active
        return dict(active=bool(active), frames=len(active['samples']) if active else 0,
                    elapsed=round(time.monotonic()-active['start'], 1) if active else 0,
                    skipped=active['skipped'] if active else 0,
                    takes=self.takes(), report=self.report)

    def start(self, expected, split, people):
        if self.active:
            raise ValueError('A recording is already in progress.')
        if expected is not None and expected not in {p['id'] for p in people}:
            raise ValueError('Select a valid registered person or Unregistered.')
        if split not in {'tuning', 'validation'}:
            raise ValueError('Choose tuning or validation.')
        if len(self.takes()) >= 200:
            raise ValueError('Calibration is limited to 200 passages. Clear old tests to continue.')
        if split == 'validation' and self.report is None:
            raise ValueError('Analyze tuning recordings before collecting validation.')
        if split == 'validation':
            self.report['can_apply'] = False
        if split == 'tuning':
            if any(t['split'] == 'validation' for t in self.takes()):
                raise ValueError('Clear tests before retuning; validation recordings must stay held out.')
            self.report = None
        self.active = dict(expected=expected, split=split, start=time.monotonic(), samples=[],
                           skipped=0, gallery=gallery_signature(people), id=uuid.uuid4().hex)

    def observe(self, observations, people, now):
        if self.active is None:
            return
        if self.active['gallery'] != gallery_signature(people):
            self.active = None
            raise ValueError('Registered people changed. Recording canceled; start a new test.')
        if now-self.active['start'] > 30 or len(self.active['samples']) >= 150:
            self.stop()
            return
        if len(observations) != 1 or observations[0]['embedding'] is None:
            self.active['skipped'] += 1
            return
        observation = observations[0]
        samples = self.active['samples']
        if samples and now-self.active['start']-samples[-1]['time'] < .15:
            return
        samples.append(dict(embedding=observation['embedding'].copy(), quality=observation['quality'],
                            time=now-self.active['start'], track=observation['track']))

    def stop(self):
        if self.active is None:
            raise ValueError('No recording in progress.')
        take = self.active
        if time.monotonic()-take['start'] < 2:
            raise ValueError('Record at least 2 seconds before finishing.')
        take['samples'] = [s for s in take['samples'] if s['track'] is not None]
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = self.directory / (take['id']+'.npz')
        with path.open('xb') as output:
            os.chmod(path, 0o600)
            np.savez_compressed(output, embeddings=np.stack([s['embedding'] for s in take['samples']]) if take['samples'] else np.empty((0,512),np.float32),
                times=np.array([s['time'] for s in take['samples']]), tracks=np.array([s['track'] for s in take['samples']]),
                metrics=np.array([[s['quality'][k] for k in ('sharpness','size','yaw','brightness','confidence','overexposed')] for s in take['samples']]))
        metadata = dict(id=take['id'], expected=take['expected'], split=take['split'], samples=len(take['samples']),
                        skipped=take['skipped'], gallery=take['gallery'], model_id=MODEL_ID)
        try:
            atomic_json(path.with_suffix('.json'), metadata)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        self.active = None

    def cancel(self):
        self.active = None

    def recordings(self, people):
        recordings = []
        signature = gallery_signature(people)
        for take in self.takes():
            if take['gallery'] != signature or take['model_id'] != MODEL_ID:
                raise ValueError('The gallery/model changed. Clear calibration tests and record new passages.')
            with np.load(self.directory / (take['id']+'.npz'), allow_pickle=False) as data:
                samples = [dict(embedding=v, time=float(t), track=int(track), quality=dict(zip(
                    ('sharpness','size','yaw','brightness','confidence','overexposed'), map(float, q))))
                    for v,t,track,q in zip(data['embeddings'],data['times'],data['tracks'],data['metrics'])]
            recordings.append((take, samples))
        return recordings

    @staticmethod
    def evaluate(recordings, people, settings):
        summary = dict(total=len(recordings), no_samples=sum(not samples for _, samples in recordings), correct=0, wrong=0, missed=0, false_accepts=0, duplicate_tracks=0,
                       known=0, unregistered=0, mean_delay=None, passages=[])
        delays = []
        for take, samples in recordings:
            tracks = {}; confirmed = {}; delay = None
            for sample in samples:
                track = sample['track']
                history = tracks.setdefault(track, [])
                if rejection(sample['quality'], settings) is None:
                    if history and float(history[-1]['embedding'] @ sample['embedding']) < .3:
                        history.clear()
                    history.append(sample)
                result = decision(history, people, settings, sample['time'])
                if history and sample['time']-history[-1]['time'] > 1:
                    result['confirmed'] = False
                if result['confirmed']:
                    confirmed.setdefault(track, result['person_id'])
                    if delay is None and result['person_id'] == take['expected']:
                        delay = sample['time']; delays.append(delay)
            predictions = set(confirmed.values())
            expected = take['expected']
            if expected is None:
                summary['unregistered'] += 1
                outcome = 'false_accept' if predictions else 'correct_unknown'
                summary['false_accepts'] += bool(predictions)
                summary['correct'] += not predictions
            else:
                summary['known'] += 1
                wrong = bool(predictions-{expected})
                missed = expected not in predictions
                summary['wrong'] += wrong; summary['missed'] += missed
                summary['correct'] += not wrong and not missed
                outcome = 'wrong' if wrong else 'missed' if missed else 'correct'
            summary['duplicate_tracks'] += max(0, len(confirmed)-len(predictions))
            summary['passages'].append(dict(id=take['id'], outcome=outcome, expected=expected,
                                           predictions=sorted(predictions), delay=delay))
        if delays:
            summary['mean_delay'] = round(float(np.mean(delays)), 2)
        return summary

    def analyze(self, people, current):
        if self.active:
            raise ValueError('Finish the recording before analysis.')
        recordings = self.recordings(people)
        tuning = [r for r in recordings if r[0]['split']=='tuning']
        validation = [r for r in recordings if r[0]['split']=='validation']
        if sum(r[0]['expected'] is not None for r in tuning) < 2 or not any(r[0]['expected'] is None for r in tuning):
            raise ValueError('Record at least 2 registered tuning passages and 1 unregistered tuning passage.')
        if self.report is None:
            best = None
            for sharp,size,yaw,threshold,count,margin in product([10.,15.,25.],[35.,50.,80.],[.30,.48],[.52,.56,.60,.66],[2,3],[.04,.08]):
                settings = dict(sharpness=sharp,min_size=size,max_yaw=yaw,threshold=threshold,min_samples=count,margin=margin)
                result = self.evaluate(tuning, people, settings)
                # Prioritize identity errors, then misses/duplicates. Prefer stricter ties.
                rank = (result['wrong']+result['false_accepts']+result['missed'], result['wrong']+result['false_accepts'], result['duplicate_tracks'],
                        -threshold, -count, -margin, -sharp, -size, yaw)
                if best is None or rank < best[0]:
                    best = (rank, settings, result)
            self.report = dict(settings=best[1], tuning=best[2], gallery=gallery_signature(people),
                               candidate_count=288)
        if self.report['gallery'] != gallery_signature(people):
            raise ValueError('Gallery changed; record new tests.')
        self.report['baseline_tuning'] = self.evaluate(tuning, people, current)
        self.report['validation'] = self.evaluate(validation, people, self.report['settings'])
        self.report['baseline_validation'] = self.evaluate(validation, people, current)
        counts = self.report['validation']
        counts['usable_known'] = sum(t['expected'] is not None and len(samples) >= 3 for t,samples in validation)
        counts['usable_unregistered'] = sum(t['expected'] is None and len(samples) >= 3 for t,samples in validation)
        self.report['can_apply'] = counts['usable_known'] >= 2 and counts['usable_unregistered'] >= 1
        self.report['take_ids'] = sorted(t['id'] for t,_ in recordings)
        return self.report

    def apply(self, people, directory):
        if self.active or not self.report or not self.report.get('can_apply'):
            raise ValueError('Analyze at least 2 fresh registered validation passages and 1 unregistered validation passage first.')
        if self.report['take_ids'] != sorted(t['id'] for t in self.takes()):
            raise ValueError('Recordings changed. Compare settings again before applying.')
        if self.report['gallery'] != gallery_signature(people):
            raise ValueError('Gallery changed; calibration is stale.')
        result = self.report['validation']
        if result['missed'] >= result['known'] or result['wrong'] or result['false_accepts']:
            raise ValueError('Validation has wrong identities, false accepts, or no correctly recognized registered passages. Collect better data and retune before applying.')
        atomic_json(directory / 'recognition_settings.json', dict(model_id=MODEL_ID, settings=self.report['settings'],
                    gallery=self.report['gallery'], validation=result))
        self.report['applied'] = True
        return dict(self.report['settings'])

    def clear(self):
        self.active = None; self.report = None
        if self.directory.exists():
            for path in self.directory.iterdir():
                if path.is_file() and path.suffix in {'.npz','.json'}:
                    path.unlink()
