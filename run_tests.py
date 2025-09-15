import unittest
from unittest.mock import patch

# This is the critical part: we patch the problematic modules directly.
# This prevents them from being loaded with their side effects.
with patch.dict('sys.modules', {
    'pyaudio': unittest.mock.MagicMock(),
    'vosk': unittest.mock.MagicMock(),
    'cv2': unittest.mock.MagicMock(),
    'deepface': unittest.mock.MagicMock(),
    'tensorflow': unittest.mock.MagicMock(),
    'tf_keras': unittest.mock.MagicMock()
}):
    # Now, we can safely import your test file inside this secure block.
    # The problematic modules will already be mocked.
    import test_voice_assistant

if __name__ == '__main__':
    # Discover all tests in the current directory and run them.
    suite = unittest.TestLoader().discover('.')
    unittest.TextTestRunner().run(suite)