import unittest
from unittest.mock import patch, MagicMock
from voice_assistant_no_button import detect_mood_from_face, get_llm_response, clean_response

class TestVoiceAssistant(unittest.TestCase):
    
    @patch('voice_assistant_no_button.cv2.VideoCapture')
    @patch('voice_assistant_no_button.DeepFace.analyze', return_value=[{'dominant_emotion': 'happy'}])
    @patch('voice_assistant_no_button.time.sleep')
    def test_detect_mood_from_face_success(self, mock_sleep, mock_analyze, mock_video_capture):
        """Test successful mood detection."""
        mock_video_capture.return_value.isOpened.return_value = True
        mock_video_capture.return_value.read.return_value = (True, 'fake_frame')
        mood = detect_mood_from_face()
        self.assertEqual(mood, "happy")
    
    @patch('voice_assistant_no_button.cv2.VideoCapture')
    @patch('voice_assistant_no_button.datetime')
    def test_detect_mood_from_face_fallback(self, mock_datetime, mock_video_capture):
        """Test mood detection falls back when camera fails."""
        mock_video_capture.return_value.isOpened.return_value = False
        mock_datetime.datetime.now.return_value.hour = 10
        
        mood = detect_mood_from_face()
        self.assertEqual(mood, "happy")
    
    @patch('voice_assistant_no_button.OpenAI')
    def test_get_llm_response_happy_mood(self, mock_openai):
        """Test LLM response with a happy mood."""
        mock_openai.return_value.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Hello there! What can I do for you today?"))]
        )
        response = get_llm_response("Hello", "happy")
        self.assertIn("Hello there! What can I do for you today?", response)
    
    def test_clean_response(self):
        """Test that markdown characters are removed."""
        text = "Hello! **This is a test** with *some formatting*."
        cleaned_text = clean_response(text)
        self.assertEqual(cleaned_text, "Hello! This is a test with some formatting.")