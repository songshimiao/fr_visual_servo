"""Exercise paired orchestration without opening cameras or moving a robot."""
import argparse
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from algorithm import camera_calibration as calibration
from algorithm.camera_stream import CameraSettings


class DualCalibrationTests(unittest.TestCase):
    def run_pair(self, cmd, tmp, fail_second=False):
        args = argparse.Namespace(camera='orbbec', camera_model='gemini305',
            single_camera=False, secondary_camera_serial=None, camera_width=640,
            camera_height=480, camera_fps=30, web_host='127.0.0.1', web_port=0,
            web_fps=10, web_quality=80, no_web=True, cmd=cmd, num_images=4,
            num_poses=3, intrinsics='305_intrinsics.json', out=str(Path(tmp) / '305.json'),
            save_images_dir=str(Path(tmp) / 'images'), ip='unused', robot_model='FR5')
        settings = [CameraSettings('orbbec', model='gemini305'),
                    CameraSettings('orbbec', model='gemini335')]
        cameras = [Mock(metadata={'model': s.model, 'role': role})
                   for s, role in zip(settings, ('eye_in_hand', 'eye_to_hand'))]
        records = []
        def collect(camera, *a, preview, **kw):
            preview.set_feedback(saved=4, target=4)
            records.append((camera, preview))
            return ([1] * 4, [2] * 4, (640, 480))
        def collect_hand(camera, *a, preview, **kw):
            records.append((camera, preview))
            return ([np.eye(3)] * 3, [np.zeros(3)] * 3,
                    [np.eye(3)] * 3, [np.zeros(3)] * 3)
        fake_robot = types.ModuleType('robot.fairino')
        fake_robot.Robot = Mock()
        fake_model = types.ModuleType('robot.fairino_robot')
        fake_model.FairinoRobot = Mock()
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.dict(sys.modules,
                {'robot.fairino': fake_robot, 'robot.fairino_robot': fake_model}))
            create = stack.enter_context(patch.object(calibration, 'create_camera',
                side_effect=[cameras[0], RuntimeError('335 absent')] if fail_second else cameras))
            stack.enter_context(patch.object(calibration, '_make_charuco_board', return_value=(None, None)))
            stack.enter_context(patch.object(calibration, 'calibration_path',
                side_effect=lambda model, kind='camera_intrinsics': str(Path(tmp) / f'{kind}_{model}.json')))
            stack.enter_context(patch.object(calibration, 'collect_charuco_images', side_effect=collect))
            stack.enter_context(patch.object(calibration, 'calibrate_intrinsics', side_effect=lambda *a: {'camera_matrix': []}))
            stack.enter_context(patch.object(calibration, 'collect_hand_eye_samples', side_effect=collect_hand))
            solve = stack.enter_context(patch.object(calibration, 'calibrate_hand_eye',
                side_effect=lambda *a: {'mode': a[-1]}))
            stack.enter_context(patch.object(calibration, 'validate_calibration'))
            stack.enter_context(patch.object(calibration, 'load_intrinsics_for_cv', return_value=(None, None)))
            if fail_second:
                with self.assertRaisesRegex(RuntimeError, '335 absent'):
                    calibration.run_dual_calibration(args, *settings)
            else:
                calibration.run_dual_calibration(args, *settings)
        return cameras, records, solve

    def test_each_camera_saves_own_intrinsics(self):
        with tempfile.TemporaryDirectory() as tmp:
            cameras, records, _ = self.run_pair('intrinsic', tmp)
            import json
            primary = json.loads((Path(tmp) / '305.json').read_text())
            secondary = json.loads((Path(tmp) / 'camera_intrinsics_gemini335.json').read_text())
            self.assertEqual(primary['camera']['model'], 'gemini305')
            self.assertEqual(secondary['camera']['model'], 'gemini335')
            self.assertEqual(len(records), 2)
            self.assertIsNot(records[0][1], records[1][1])
            for camera in cameras:
                camera.stop.assert_called_once()

    def test_fixed_hand_eye_modes_are_never_shared(self):
        with tempfile.TemporaryDirectory() as tmp:
            cameras, records, solve = self.run_pair('extrinsic', tmp)
            self.assertCountEqual([call.args[-1] for call in solve.call_args_list],
                                  ['eye_in_hand', 'eye_to_hand'])
            fake_modes = {p.name: __import__('json').loads(p.read_text())['mode']
                          for p in Path(tmp).glob('*.json')}
            self.assertEqual(fake_modes['305.json'], 'eye_in_hand')
            self.assertEqual(fake_modes['hand_eye_calibration_gemini335.json'], 'eye_to_hand')
            for camera in cameras:
                camera.stop.assert_called_once()

    def test_second_camera_start_failure_releases_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            cameras, _, _ = self.run_pair('intrinsic', tmp, fail_second=True)
            cameras[0].stop.assert_called_once()
            cameras[1].stop.assert_not_called()
            self.assertFalse(list(Path(tmp).glob('*.json')))


if __name__ == '__main__':
    unittest.main()
