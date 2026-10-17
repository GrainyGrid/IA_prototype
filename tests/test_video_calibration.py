import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import app as service
import test_app as fixtures
from calibration import Calibration, load_settings
from video_pipeline import DEFAULTS, VideoPipeline, decision, gallery_signature


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.owner,self.other,self.unknown = np.eye(512,dtype=np.float32)[:3]
        self.people = [dict(id='a',name='Alex',reference=self.owner),dict(id='b',name='Blair',reference=self.other)]
        self.quality = dict(sharpness=100.,size=150.,yaw=0.,brightness=100.,confidence=.99,overexposed=0.)

    def tearDown(self):
        self.temp.cleanup()

    def samples(self, vector, count=5):
        return [dict(embedding=vector,quality=dict(self.quality),time=i*.2,track=1) for i in range(count)]

    def test_votes_unknown_and_candidate_separation(self):
        self.assertFalse(decision(self.samples(self.owner,2),self.people,DEFAULTS,.2)['confirmed'])
        self.assertEqual(decision(self.samples(self.owner),self.people,DEFAULTS,.8)['person_id'],'a')
        self.assertFalse(decision(self.samples(self.unknown),self.people,DEFAULTS,.8)['confirmed'])
        near_duplicate = [self.people[0],dict(id='duplicate',reference=self.owner)]
        self.assertFalse(decision(self.samples(self.owner),near_duplicate,DEFAULTS,.8)['confirmed'])
        poor = self.samples(self.owner)
        for sample in poor: sample['quality']['sharpness']=1
        self.assertFalse(decision(poor,self.people,DEFAULTS,.8)['confirmed'])

    def test_one_log_per_track_and_gallery_deletion(self):
        pipeline=VideoPipeline(self.path,DEFAULTS)
        for i in range(6):
            result=pipeline.observe(7,self.owner,self.quality,self.people,i*.2)
        self.assertTrue(result['confirmed'])
        events=[json.loads(s) for s in (self.path/'entries.jsonl').read_text().splitlines()]
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['person_id'],'a')
        pipeline.observe(7,None,self.quality,self.people,2.1)
        self.assertFalse(pipeline.observe(7,None,self.quality,self.people,2.2)['confirmed'])
        self.assertFalse(pipeline.observe(7,self.owner,self.quality,self.people[1:],2.4)['confirmed'])
        self.assertEqual((self.path/'entries.jsonl').stat().st_mode & 0o777,0o600)

    def test_bytetrack_keeps_two_moving_faces_separate_and_resets_after_gap(self):
        pipeline=VideoPipeline(self.path,DEFAULTS)
        def face(x): return np.array([x,100,100,120,0,0,0,0,0,0,0,0,0,0,.99],np.float32)
        ids=pipeline.associate([face(100),face(350)],0,(640,480))
        next_ids=pipeline.associate([face(110),face(340)],.2,(640,480))
        self.assertEqual(ids,next_ids)
        self.assertNotEqual(ids[0],ids[1])
        session=pipeline.session_id
        pipeline.associate([],3,(640,480))
        self.assertNotEqual(pipeline.session_id,session)

    def save_take(self, calibration, expected, split, vector):
        calibration.active=dict(id=__import__('uuid').uuid4().hex,expected=expected,split=split,start=0,
            gallery=gallery_signature(self.people),skipped=0,samples=self.samples(vector))
        with patch('calibration.time.monotonic',return_value=3): calibration.stop()

    def test_tune_validate_and_apply_with_identical_live_rule(self):
        c=Calibration(self.path)
        with self.assertRaises(ValueError): c.start('a','validation',self.people)
        self.save_take(c,'a','tuning',self.owner)
        self.save_take(c,'b','tuning',self.other)
        self.save_take(c,None,'tuning',self.unknown)
        report=c.analyze(self.people,DEFAULTS)
        self.assertEqual(report['candidate_count'],144)
        self.assertFalse(report['can_apply'])
        candidate=dict(report['settings'])
        with self.assertRaises(ValueError): c.apply(self.people,self.path)
        self.save_take(c,'a','validation',self.owner)
        self.save_take(c,'b','validation',self.other)
        self.save_take(c,None,'validation',self.unknown)
        report=c.analyze(self.people,DEFAULTS)
        self.assertEqual(report['settings'],candidate)
        self.assertEqual(report['validation']['correct'],3)
        self.assertTrue(report['can_apply'])
        with self.assertRaises(ValueError): c.start('a','tuning',self.people)
        applied=c.apply(self.people,self.path)
        self.assertEqual(load_settings(self.path,DEFAULTS),applied)
        c.report['validation']['false_accepts']=1
        with self.assertRaises(ValueError): c.apply(self.people,self.path)
        with self.assertRaises(ValueError): c.apply(self.people[1:],self.path)

    def test_no_detection_passage_is_counted_as_missed_and_clear_preserves_settings(self):
        c=Calibration(self.path)
        c.active=dict(id='empty',expected='a',split='tuning',start=0,gallery=gallery_signature(self.people),skipped=5,samples=[])
        with patch('calibration.time.monotonic',return_value=3): c.stop()
        recordings=c.recordings(self.people)
        evaluated=c.evaluate(recordings,self.people,DEFAULTS)
        self.assertEqual(evaluated['missed'],1)
        self.assertEqual(evaluated['no_samples'],1)
        (self.path/'recognition_settings.json').write_text('preserved')
        c.clear()
        self.assertEqual(c.takes(),[])
        self.assertEqual((self.path/'recognition_settings.json').read_text(),'preserved')


class PipelineAppTests(unittest.TestCase):
    setUp=fixtures.FaceAppTests.setUp
    tearDown=fixtures.FaceAppTests.tearDown
    enroll=fixtures.FaceAppTests.enroll
    detector=fixtures.FaceAppTests.detector
    post=fixtures.FaceAppTests.post

    def test_production_pipeline_waits_and_logs_then_calibration_bypasses_blur_gate(self):
        self.enroll()
        self.engine.pipeline=VideoPipeline(service.RUNTIME_DATA,DEFAULTS)
        self.detector()
        with patch.object(self.engine,'feature',return_value=self.owner):
            for i in range(3):
                with patch('app.time.monotonic',return_value=10+i*.3): result=self.engine.process(self.image)
                self.assertEqual(result['faces'][0]['matched'],i==2)
        self.assertEqual(len((service.RUNTIME_DATA/'entries.jsonl').read_text().splitlines()),1)
        response=self.post('/api/calibration/start',json={'person_id':'person001','split':'tuning'})
        self.assertEqual(response.status_code,200,response.get_json())
        blurred=cv2.GaussianBlur(self.image,(0,0),10)
        with patch.object(self.engine,'feature',return_value=self.owner) as extract:
            self.engine.process(blurred)
            extract.assert_called_once()
        self.assertEqual(len(self.engine.calibration.active['samples']),1)
        self.assertEqual(self.post('/api/register',json={'name':'Someone'}).status_code,409)
        self.assertEqual(self.post('/api/calibration/cancel').status_code,200)
        self.assertIsNone(self.engine.calibration.active)
        self.assertEqual(self.post('/api/calibration/apply').status_code,400)
        self.assertEqual(self.post('/api/calibration/no-such-action').status_code,404)


if __name__ == '__main__': unittest.main()
