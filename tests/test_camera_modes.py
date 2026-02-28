from time import monotonic
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.live_recognition import RecognizedFace
from gui.camera import CameraCapture


class CameraModeTests(unittest.TestCase):
    def render(self, active, recognized):
        camera = CameraCapture()
        camera.enable_recognition(active)
        box = (10, 30, 50, 50)
        if recognized:
            camera.set_recognition([RecognizedFace(box, 'An (S01)')], monotonic())
        handle = Mock()
        handle.isOpened.return_value = True
        handle.read.return_value = (True, np.zeros((100,100,3), dtype=np.uint8))
        frames = []

        def publish(kind, value):
            if kind == 'frame':
                frames.append(value)
                camera.stop()

        with patch('gui.camera.cv2.VideoCapture', return_value=handle), patch(
            'gui.camera.FaceDetector'
        ) as detector, patch('gui.camera.ImageDraw.Draw') as draw, patch.object(camera, '_publish', side_effect=publish):
            detector.return_value.detect_faces.return_value = [box]
            camera._run()
            self.assertEqual(len(frames), 1)
            self.assertTrue(np.any(frames[0][30,10]))
            handle.release.assert_called_once()
            return draw.return_value.text.call_args_list

    def test_camera_only_draws_boxes_without_text(self):
        self.assertEqual(self.render(False, True), [])

    def test_attendance_draws_unknown_or_identity(self):
        self.assertEqual(self.render(True, False)[0].args[1], 'Unknown')
        self.assertEqual(self.render(True, True)[0].args[1], 'An (S01)')

    def test_disabling_clears_labels_and_rejects_late_results(self):
        camera = CameraCapture()
        result = RecognizedFace((10,10,20,20), 'An (S01)')
        camera.enable_recognition(True)
        camera.set_recognition([result], monotonic())
        camera.enable_recognition(False)
        camera.set_recognition([result], monotonic())
        self.assertEqual(camera._recognition, (0, []))
