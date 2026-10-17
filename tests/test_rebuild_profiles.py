import unittest
from unittest.mock import patch

import numpy as np

import app as service
from face_models import MODEL_ID, PROFILE_NAME
from rebuild_profiles import rebuild_folder
import test_app as fixtures


class MigrationTests(unittest.TestCase):
    setUp = fixtures.FaceAppTests.setUp
    tearDown = fixtures.FaceAppTests.tearDown
    enroll = fixtures.FaceAppTests.enroll
    detector = fixtures.FaceAppTests.detector

    def test_migration_preserves_old_profile_and_is_repeatable(self):
        folder = self.enroll()
        (folder / PROFILE_NAME).unlink()
        old = folder / 'profile.npz'
        old.write_bytes(b'original-sface-archive')
        self.detector()
        with patch.object(self.engine, 'feature', return_value=self.owner):
            self.assertEqual(rebuild_folder(self.engine, folder), 'converted 40 photos')
        self.assertEqual(old.read_bytes(), b'original-sface-archive')
        with np.load(folder / PROFILE_NAME, allow_pickle=False) as profile:
            self.assertEqual(profile['model_id'].item(), MODEL_ID)
            self.assertEqual(profile['features'].shape, (40, 512))
        with patch.object(self.engine, 'feature') as feature:
            self.assertEqual(rebuild_folder(self.engine, folder), 'already converted')
            feature.assert_not_called()

    def test_ambiguous_photo_does_not_create_partial_archive(self):
        folder = self.enroll()
        (folder / PROFILE_NAME).unlink()
        self.detector(2)
        with self.assertRaises(ValueError):
            rebuild_folder(self.engine, folder)
        self.assertFalse((folder / PROFILE_NAME).exists())
        self.assertEqual(list(folder.glob('.arcface-*')), [])

    def test_different_checkpoint_is_never_used_for_matching(self):
        folder = self.enroll()
        np.savez(folder / PROFILE_NAME, name='Wrong model', model_id='other-arcface-checkpoint',
                 features=np.stack([self.owner] * 40))
        self.assertEqual(self.engine.status()['person_count'], 0)
        self.assertEqual(len(self.engine.status()['invalid_profiles']), 1)


if __name__ == '__main__':
    unittest.main()
