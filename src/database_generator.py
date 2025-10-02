# src/database_generator.py
import random

class DatabaseGenerator:
    """Generates a large database of unique documents on demand"""
    def __init__(self):
        self.templates = [
            "The quick brown fox {action} over the lazy dog.",
            "A journey of a thousand miles {beginning} with a single step.",
            "To be or not to be, that is the {question}.",
            "All that glitters is not {substance}.",
            "Actions speak louder than {communication}.",
            "The early bird catches the {opportunity}.",
            "Don't count your chickens before they {hatch}.",
            "A picture is worth a thousand {words}.",
            "Better late than {never}.",
            "Rome wasn't built in a {day}."
        ]
        
        self.variations = {
            'action': ['jumps', 'leaps', 'hops', 'skips', 'bounds'],
            'beginning': ['begins', 'starts', 'commences', 'initiates'],
            'question': ['question', 'query', 'inquiry', 'dilemma'],
            'substance': ['gold', 'silver', 'diamond', 'treasure'],
            'communication': ['words', 'actions', 'gestures', 'signals'],
            'opportunity': ['worm', 'chance', 'moment', 'opening'],
            'hatch': ['hatch', 'emerge', 'appear', 'come out'],
            'words': ['words', 'pictures', 'images', 'descriptions'],
            'never': ['never', 'late', 'absent', 'missing'],
            'day': ['day', 'night', 'week', 'month']
        }
    
    def generate_database(self, size=1000):
        """Generate a database of unique documents"""
        documents = []
        used_combinations = set()
        
        while len(documents) < size:
            template = random.choice(self.templates)
            
            # Replace placeholders with random variations
            for placeholder, variations in self.variations.items():
                if f"{{{placeholder}}}" in template:
                    variation = random.choice(variations)
                    template = template.replace(f"{{{placeholder}}}", variation)
            
            # Ensure uniqueness
            if template not in used_combinations:
                used_combinations.add(template)
                documents.append(template)
        
        return documents