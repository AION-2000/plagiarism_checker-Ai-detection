# tests/test_plagiarism_checker.py
import unittest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from plagiarism_checker import PlagiarismChecker

class TestPlagiarismChecker(unittest.TestCase):
    def setUp(self):
        self.checker = PlagiarismChecker()
        self.sample_text = "The quick brown fox jumps over the lazy dog."
    
    def test_plagiarism_detection(self):
        # Test plagiarism detection with known match
        self.checker.generate_source_database(100)
        results = self.checker.check_plagiarism(self.sample_text, threshold=0.7)
        
        self.assertNotIn('error', results)
        self.assertGreater(len(results['source_matches']), 0)
    
    def test_ai_detection(self):
        # Test AI detection
        ai_results = self.checker.check_ai_content(self.sample_text)
        
        self.assertNotIn('error', ai_results)
        self.assertIn('ai_probability', ai_results)
        self.assertIn('indicators', ai_results)
    
    def test_empty_text(self):
        # Test with empty text
        results = self.checker.check_plagiarism("", threshold=0.7)
        self.assertIn('error', results)
        
        ai_results = self.checker.check_ai_content("")
        self.assertEqual(ai_results['ai_probability'], 0)

if __name__ == '__main__':
    unittest.main()