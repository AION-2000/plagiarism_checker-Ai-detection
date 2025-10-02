# src/ai_detector.py
import re
import random
import math
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize

class AIDetector:
    """Detects if text might be AI-generated using linguistic analysis"""
    def __init__(self):
        # Compact list of common AI phrases
        self.ai_indicators = [
            "in conclusion", "furthermore", "moreover", "additionally",
            "it is important to note", "it should be noted", "it is evident that",
            "it is clear that", "it is apparent that", "it is crucial to",
            "it is essential to", "it is vital to", "it is imperative to",
            "it is necessary to", "it is recommended to", "it is suggested to",
            "it is believed that", "it is thought that", "it is considered that",
            "it is assumed that", "it is expected that", "it is anticipated that"
        ]
        
        # AI sentence patterns (compact regex patterns)
        self.ai_patterns = [
            r"it is \w+ to \w+",
            r"it is \w+ that",
            r"the \w+ of the \w+",
            r"one of the most \w+",
            r"it is worth noting that",
            r"it should be emphasized that",
            r"it must be remembered that"
        ]
    
    def detect_ai_content(self, text, use_web=True):
        """Detect if text might be AI-generated"""
        if not text or len(text.strip()) < 50:
            return {
                'ai_probability': 0,
                'indicators': [],
                'web_results': [],
                'analysis': 'Text too short for analysis'
            }
        
        # Basic linguistic analysis
        indicators = self._analyze_linguistic_patterns(text)
        
        # Web-based AI detection
        web_results = []
        if use_web:
            web_results = self._web_ai_detection(text)
        
        # Calculate overall probability
        ai_probability = self._calculate_ai_probability(indicators, web_results)
        
        return {
            'ai_probability': ai_probability,
            'indicators': indicators,
            'web_results': web_results,
            'analysis': self._generate_analysis(ai_probability, indicators, web_results)
        }
    
    def _analyze_linguistic_patterns(self, text):
        """Analyze text for AI linguistic patterns"""
        indicators = []
        sentences = sent_tokenize(text)
        
        if len(sentences) < 3:
            return indicators
        
        # Check for AI phrases
        text_lower = text.lower()
        for phrase in self.ai_indicators:
            if phrase in text_lower:
                indicators.append({
                    'type': 'phrase',
                    'pattern': phrase,
                    'count': text_lower.count(phrase)
                })
        
        # Check for AI patterns
        for pattern in self.ai_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                indicators.append({
                    'type': 'pattern',
                    'pattern': pattern,
                    'count': len(matches)
                })
        
        # Check sentence length uniformity
        sentence_lengths = [len(sent.split()) for sent in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((x - avg_length) ** 2 for x in sentence_lengths) / len(sentence_lengths)
        
        if variance < 5:  # Low variance suggests uniform sentence lengths
            indicators.append({
                'type': 'structure',
                'pattern': 'uniform_sentence_length',
                'variance': round(variance, 2),
                'avg_length': round(avg_length, 1)
            })
        
        # Check vocabulary richness
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and len(word) > 2]
        unique_words = len(set(words))
        total_words = len(words)
        
        if total_words > 0:
            richness = unique_words / total_words
            if richness < 0.5:  # Low vocabulary richness
                indicators.append({
                    'type': 'vocabulary',
                    'pattern': 'low_vocabulary_richness',
                    'richness': round(richness, 3)
                })
        
        return indicators
    
    def _web_ai_detection(self, text):
        """Use web search for AI detection (mock implementation)"""
        # In a real implementation, this would call AI detection APIs
        # For now, we'll simulate with mock results
        
        # Extract key phrases from text
        sentences = sent_tokenize(text)
        key_phrases = [sent[:50] for sent in sentences[:3]]
        
        mock_results = []
        for phrase in key_phrases:
            # Simulate web search results
            mock_results.append({
                'source': 'AI Content Analyzer',
                'confidence': random.uniform(0.3, 0.9),
                'phrase': phrase,
                'indicators': ['formal tone', 'structured sentences']
            })
        
        return mock_results
    
    def _calculate_ai_probability(self, indicators, web_results):
        """Calculate overall AI probability"""
        base_score = 0
        
        # Score based on linguistic indicators
        for indicator in indicators:
            if indicator['type'] == 'phrase':
                base_score += min(indicator['count'] * 5, 20)
            elif indicator['type'] == 'pattern':
                base_score += min(indicator['count'] * 3, 15)
            elif indicator['type'] == 'structure':
                base_score += 10
            elif indicator['type'] == 'vocabulary':
                base_score += 15
        
        # Score based on web results
        if web_results:
            web_confidence = sum(result['confidence'] for result in web_results) / len(web_results)
            base_score += web_confidence * 30
        
        # Normalize to 0-100 scale
        ai_probability = min(base_score, 100)
        
        return round(ai_probability, 1)
    
    def _generate_analysis(self, ai_probability, indicators, web_results):
        """Generate analysis text based on results"""
        if ai_probability < 30:
            return "Text appears to be human-written with natural language patterns."
        elif ai_probability < 60:
            return "Text shows some AI-like characteristics but may still be human-written."
        elif ai_probability < 80:
            return "Text has significant AI characteristics. Likely AI-generated or heavily edited."
        else:
            return "Text strongly indicates AI generation. Multiple AI patterns detected."