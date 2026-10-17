import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import app as service


class FaceAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.profile_patch = patch.object(service, 'PROFILE', Path(self.temp.name) / 'profile.npz')
        self.registration_patch = patch.object(service, 'REGISTRATIONS', Path(self.temp.name) / 'registration')
        self.profile_patch.start()
        self.registration_patch.start()
        self.runtime_patch = patch.object(service, 'RUNTIME_DATA', Path(self.temp.name) / 'data')
        self.runtime_patch.start()
        self.engine = service.FaceEngine()
        self.engine.pipeline = None
        self.engine_patch = patch.object(service, 'engine', self.engine)
        self.engine_patch.start()
        self.client = service.app.test_client()
        self.base = 'http://127.0.0.1:5000'
        self.owner, self.other = np.eye(512, dtype=np.float32)[:2]
        self.face = np.array([100, 100, 180, 200, 150, 160, 230, 160, 190, 190, 165, 220, 215, 220, .99], np.float32)
        self.image = np.random.default_rng(0).integers(75, 230, (480, 640, 3), dtype=np.uint8)

    def tearDown(self):
        self.engine_patch.stop()
        self.registration_patch.stop()
        self.profile_patch.stop()
        self.runtime_patch.stop()
        self.temp.cleanup()

    def post(self, path, **kwargs):
        return self.client.post(path, base_url=self.base, **kwargs)

    def enroll(self, name='Alex', vector=None):
        vector = self.owner if vector is None else vector
        _, jpeg = cv2.imencode('.jpg', self.image)
        images = [(f'{stage}{number:02d}.jpg', jpeg.tobytes()) for stage in service.STAGES for number in range(1, 9)]
        self.engine.save_registration({'name': name, 'features': [vector] * 40, 'images': images})
        return Path(self.temp.name) / self.engine.registration_directory

    def detector(self, count=1):
        detector = Mock()
        detector.detect.return_value = (None, np.stack([self.face] * count))
        self.engine.detector = detector
        return detector

    def test_reject_bad_frames_and_cross_origin(self):
        self.assertEqual(self.post('/api/frame', data=b'junk', content_type='image/jpeg').status_code, 400)
        self.assertEqual(self.post('/api/frame', data=b'junk', content_type='text/plain').status_code, 415)
        self.assertEqual(self.post('/api/register', json={'name': 'Alex'}, headers={'Origin': 'https://example.com'}).status_code, 403)
        self.assertEqual(self.client.get('/api/status', base_url='http://evil.example:5000').status_code, 403)

    def test_cancel_preserves_all_existing_registrations(self):
        folder = self.enroll()
        self.post('/api/register', json={'name': 'New'})
        self.post('/api/cancel')
        self.assertEqual(self.engine.status()['person_count'], 1)
        self.assertTrue(folder.exists())
        self.assertFalse((service.REGISTRATIONS / 'person002').exists())

    def test_pose_gates_all_angles(self):
        baseline = [0, .5]
        for stage in service.STAGES[1:]:
            self.assertFalse(service.pose_matches(stage, 0, .5, baseline))
        for stage, yaw, pitch in [('front', 0, .5), ('left', .22, .5), ('right', -.22, .5), ('up', 0, .4), ('down', 0, .6)]:
            self.assertTrue(service.pose_matches(stage, yaw, pitch, baseline))

    def test_normalized_reference_rejects_three_outlier_samples(self):
        samples = np.stack([self.other] * 3 + [self.owner] * 37)
        self.assertEqual(float(np.sort(samples @ self.other)[-3:].mean()), 1.)
        reference = service.reference_embedding(samples)
        self.assertAlmostEqual(float(np.linalg.norm(reference)), 1.)
        self.assertLess(float(reference @ self.other), service.THRESHOLD)
        np.testing.assert_allclose(service.reference_embedding(samples * 4), reference)
        with self.assertRaises(ValueError):
            service.reference_embedding(np.zeros((40, 512), dtype=np.float32))

    def test_live_threshold_and_quality_checks(self):
        self.enroll('Owner')
        self.detector()
        for score in [0., service.THRESHOLD - .01, service.THRESHOLD + .01, 1.]:
            probe = np.zeros(512, dtype=np.float32)
            probe[:2] = [score, np.sqrt(1-score**2)]
            with patch.object(self.engine, 'feature', return_value=probe):
                face = self.engine.process(self.image)['faces'][0]
            self.assertAlmostEqual(face['similarity'], score, places=5)
            self.assertEqual(face['matched'], score >= service.THRESHOLD)
            self.assertEqual(face['label'], 'Owner' if score >= service.THRESHOLD else 'Unknown')
        with patch.object(self.engine, 'feature', return_value=self.owner) as extract:
            face = self.engine.process(np.zeros_like(self.image))['faces'][0]
            extract.assert_not_called()
            self.assertFalse(face['matched'])
            self.assertIsNone(face['similarity'])

    def test_sharpness_is_consistent_across_crop_scales(self):
        gray = np.random.default_rng(7).integers(70, 230, (128, 128), dtype=np.uint8)
        enlarged = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_NEAREST)
        self.assertAlmostEqual(service.sharpness_score(gray), service.sharpness_score(enlarged))
        self.assertGreater(service.sharpness_score(gray), service.BLUR_THRESHOLD)
        smeared = cv2.GaussianBlur(gray, (0, 0), 4)
        self.assertLess(service.sharpness_score(smeared), service.BLUR_THRESHOLD)

    def test_blurry_face_is_rejected_with_measured_score(self):
        blurred = cv2.GaussianBlur(self.image, (0, 0), 10)
        warning = service.face_quality(blurred, self.face)
        self.assertIn('sharpness', warning)
        self.assertIn('minimum', warning)
        self.assertIsNone(service.face_quality(self.image, self.face))

    def test_identifies_multiple_faces_and_unknown_in_one_frame(self):
        self.enroll('Alex', self.owner)
        self.enroll('Blair', self.other)
        self.detector(3)
        unknown = np.eye(512, dtype=np.float32)[2]
        with patch.object(self.engine, 'feature', side_effect=[self.owner, self.other, unknown]):
            result = self.engine.process(self.image)
        self.assertEqual([face['label'] for face in result['faces']], ['Alex', 'Blair', 'Unknown'])
        self.assertEqual([face['person_id'] for face in result['faces']], ['person001', 'person002', None])
        self.assertEqual(result['person_count'], 2)

    def test_external_deletion_stops_recognition_without_restart(self):
        first = self.enroll('Alex', self.owner)
        second = self.enroll('Blair', self.other)
        # The old global profile must never resurrect a deleted registration.
        np.savez(service.PROFILE, name='Alex', features=np.stack([self.owner] * 40))
        self.detector()
        shutil.rmtree(first)
        with patch.object(self.engine, 'feature', return_value=self.owner):
            face = self.engine.process(self.image)['faces'][0]
        self.assertEqual(face['label'], 'Unknown')
        self.assertEqual(self.engine.status()['person_count'], 1)
        self.assertTrue(second.exists())
        self.assertEqual(service.FaceEngine().status()['person_count'], 1)
        shutil.rmtree(service.REGISTRATIONS)
        self.assertFalse(self.engine.status()['registered'])

    def test_management_rename_and_remove_selected_folder(self):
        first = self.enroll('Alex')
        second = self.enroll('Blair', self.other)
        response = self.client.patch('/api/people/person001', base_url=self.base, json={'name': 'Renamed'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(service.FaceEngine().people['person001']['name'], 'Renamed')
        self.assertEqual(json.loads((first / 'metadata.json').read_text())['name'], 'Renamed')
        self.detector()
        with patch.object(self.engine, 'feature', return_value=self.owner):
            self.assertEqual(self.engine.process(self.image)['faces'][0]['label'], 'Renamed')
        with self.client.get('/api/people/person001/thumbnail', base_url=self.base) as preview:
            self.assertEqual(preview.status_code, 200)
        response = self.client.delete('/api/people/person001', base_url=self.base)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(first.exists())
        self.assertTrue(second.exists())
        self.assertEqual(response.get_json()['person_count'], 1)
        self.assertEqual(self.client.delete('/api/people/person001', base_url=self.base).status_code, 404)
        with self.assertRaises(ValueError):
            self.engine.person_folder('../data')
        with self.assertRaises(ValueError):
            self.engine.person_folder('..')

    def test_broken_profile_is_removed_from_cache_and_listed(self):
        folder = self.enroll()
        (folder / service.PROFILE_NAME).write_bytes(b'PK\x03\x04broken')
        status = self.engine.status()
        self.assertEqual(status['person_count'], 0)
        self.assertEqual(status['invalid_profiles'][0]['id'], 'person001')
        self.assertFalse(status['registered'])

    def test_registration_saves_all_images_and_retains_existing_people(self):
        original = self.enroll('Existing', self.other)
        detector = self.detector(2)
        self.post('/api/register', json={'name': 'Alex'})
        with patch.object(self.engine, 'feature', return_value=self.owner):
            self.assertEqual(self.engine.process(self.image)['count'], 0)
            detector.detect.return_value = (None, np.stack([self.face]))
            self.assertEqual(self.engine.process(np.zeros_like(self.image))['count'], 0)
            for index in range(service.TOTAL):
                stage = service.STAGES[index // service.SAMPLES_PER_STAGE]
                pose = {'front': (0, .5), 'left': (.22, .5), 'right': (-.22, .5), 'up': (0, .4), 'down': (0, .6)}[stage]
                with patch.object(service, 'pose_ratios', return_value=pose), patch.object(service.time, 'monotonic', return_value=index + 1):
                    if index == 39:
                        with patch.object(self.engine, 'save_registration', side_effect=OSError('Disk full')):
                            with self.assertRaises(OSError):
                                self.engine.process(self.image)
                        self.assertEqual(len(self.engine.registration['images']), 39)
                    result = self.engine.process(self.image)
            self.assertTrue(result['completed'])
            self.assertEqual(result['person_count'], 2)
            folder = service.REGISTRATIONS / 'person002'
            expected = [f'{stage}{number:02d}.jpg' for stage in service.STAGES for number in range(1, 9)]
            self.assertEqual(sorted(p.name for p in folder.glob('*.jpg')), sorted(expected))
            for filename in expected:
                self.assertIsNotNone(cv2.imread(str(folder / filename)))
            metadata = json.loads((folder / 'metadata.json').read_text())
            self.assertEqual(metadata['images'], expected)
            with np.load(folder / service.PROFILE_NAME, allow_pickle=False) as archived:
                self.assertAlmostEqual(float(np.linalg.norm(archived['reference_embedding'])), 1.)
            if os.name != 'nt':
                self.assertEqual((folder / service.PROFILE_NAME).stat().st_mode & 0o777, 0o600)
            self.assertTrue(original.exists())
            self.assertEqual(service.FaceEngine().status()['person_count'], 2)
            with patch.object(service.np, 'savez_compressed', side_effect=OSError('Disk full')):
                with self.assertRaises(OSError):
                    self.enroll('Failed')
            self.assertFalse((service.REGISTRATIONS / 'person003').exists())
            self.assertEqual(self.engine.status()['person_count'], 2)

    @unittest.skipUnless((service.ROOT / 'models/w600k_r50.onnx').exists(), 'Download models for inference smoke test')
    def test_real_models_blank_frame(self):
        _, jpeg = cv2.imencode('.jpg', np.zeros((480, 640, 3), np.uint8))
        response = self.post('/api/frame', data=jpeg.tobytes(), content_type='image/jpeg')
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()['faces'], [])


if __name__ == '__main__':
    unittest.main()
