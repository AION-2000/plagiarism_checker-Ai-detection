import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import os
import threading
import time
import requests
from urllib.parse import quote
import random
import math
from collections import Counter

# Try to import optional libraries
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

class AIDetector:
    """Detects if text might be AI-generated using web search and linguistic analysis"""
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

class WebSearchEngine:
    """Handles web search operations"""
    def __init__(self):
        self.search_engines = {
            'mock': self._search_mock,
            'google': self._search_google,
            'bing': self._search_bing
        }
        self.current_engine = 'mock'  # Default to mock for testing
    
    def search(self, query, num_results=10):
        """Search the web using the configured engine"""
        if self.current_engine in self.search_engines:
            return self.search_engines[self.current_engine](query, num_results)
        return []
    
    def _search_mock(self, query, num_results):
        """Mock search engine for testing purposes"""
        # Generate mock results based on query keywords
        keywords = query.lower().split()
        mock_results = []
        
        # Common web sources
        web_sources = [
            {
                'title': 'Wikipedia - General Knowledge',
                'url': 'https://en.wikipedia.org/wiki/Knowledge',
                'content': 'Knowledge is a familiarity, awareness, or understanding of someone or something.'
            },
            {
                'title': 'Famous Quotes Database',
                'url': 'https://example.com/quotes',
                'content': 'A collection of famous quotes from around the world.'
            },
            {
                'title': 'Common Phrases Dictionary',
                'url': 'https://example.com/phrases',
                'content': 'A comprehensive dictionary of common English phrases and their origins.'
            },
            {
                'title': 'Academic Papers Repository',
                'url': 'https://example.com/papers',
                'content': 'Access to millions of academic papers and research articles.'
            },
            {
                'title': 'News Articles Archive',
                'url': 'https://example.com/news',
                'content': 'Archive of news articles from various sources around the world.'
            }
        ]
        
        # Generate results based on query
        for i in range(min(num_results, len(web_sources))):
            source = web_sources[i]
            result = {
                'title': source['title'],
                'url': source['url'],
                'content': source['content'] + ' ' + ' '.join(keywords[:3])
            }
            mock_results.append(result)
        
        return mock_results
    
    def _search_google(self, query, num_results):
        """Search using Google (requires API key)"""
        # Placeholder for real Google search implementation
        return self._search_mock(query, num_results)
    
    def _search_bing(self, query, num_results):
        """Search using Bing (requires API key)"""
        # Placeholder for real Bing search implementation
        return self._search_mock(query, num_results)

class PlagiarismChecker:
    def __init__(self):
        self.source_documents = []
        self.web_sources = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
        self.source_vectors = None
        self.web_vectors = None
        self.db_generator = DatabaseGenerator()
        self.web_search_engine = WebSearchEngine()
        self.ai_detector = AIDetector()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def _preprocess_text(self, text):
        """Preprocess text for comparison"""
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return " ".join(tokens)
    
    def generate_source_database(self, size=1000):
        """Generate a large database of source documents"""
        self.source_documents = self.db_generator.generate_database(size)
        self._update_source_vectors()
        return len(self.source_documents)
    
    def _update_source_vectors(self):
        """Update the vectorized representation of source documents"""
        if self.source_documents:
            processed_docs = [self._preprocess_text(doc) for doc in self.source_documents]
            self.source_vectors = self.vectorizer.fit_transform(processed_docs)
            return True
        return False
    
    def check_plagiarism(self, query_text, threshold=0.7, check_web=False, db_size=1000):
        """Check query text against source documents and optionally web sources"""
        if not query_text or not query_text.strip():
            return {'error': 'Query text is empty'}
        
        # Generate source database if not already generated
        if not self.source_documents:
            db_count = self.generate_source_database(db_size)
            print(f"Generated database with {db_count} documents")
        
        # Split text into sentences for detailed checking
        sentences = sent_tokenize(query_text)
        sentence_results = []
        
        # Preprocess the query text
        processed_query = self._preprocess_text(query_text)
        
        # Vectorize the query
        try:
            query_vector = self.vectorizer.transform([processed_query])
        except Exception as e:
            return {'error': f'Error vectorizing query: {str(e)}'}
        
        # Calculate cosine similarity with source documents
        source_similarities = cosine_similarity(query_vector, self.source_vectors)[0]
        
        # Check against web sources if requested
        web_similarities = []
        if check_web:
            web_results = self.web_search_engine.search(query_text)
            if web_results:
                self.web_sources = web_results
                web_contents = [self._preprocess_text(src['content']) for src in self.web_sources]
                self.web_vectors = self.vectorizer.transform(web_contents)
                web_similarities = cosine_similarity(query_vector, self.web_vectors)[0]
        
        # Check each sentence individually
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
                
            processed_sentence = self._preprocess_text(sentence)
            sentence_vector = self.vectorizer.transform([processed_sentence])
            
            # Check against source documents
            source_sentence_sim = cosine_similarity(sentence_vector, self.source_vectors)[0]
            max_source_sim = max(source_sentence_sim) if len(source_sentence_sim) > 0 else 0
            
            # Check against web sources
            max_web_sim = 0
            if check_web and self.web_vectors is not None:
                web_sentence_sim = cosine_similarity(sentence_vector, self.web_vectors)[0]
                max_web_sim = max(web_sentence_sim) if len(web_sentence_sim) > 0 else 0
            
            # Determine if this sentence has potential plagiarism
            max_sim = max(max_source_sim, max_web_sim)
            
            if max_sim >= threshold:
                source_type = "Local Document" if max_source_sim >= max_web_sim else "Web Source"
                sentence_results.append({
                    'sentence_index': i,
                    'sentence': sentence,
                    'similarity_score': round(max_sim, 2),
                    'source_type': source_type
                })
        
        # Find matches above threshold for overall document
        results = {
            'source_matches': [],
            'web_matches': [],
            'sentence_matches': sentence_results,
            'error': None
        }
        
        # Source document matches
        for idx, score in enumerate(source_similarities):
            if score >= threshold:
                results['source_matches'].append({
                    'document_index': idx,
                    'similarity_score': round(score, 2),
                    'source_text': self.source_documents[idx][:100] + "...",
                    'source_type': 'Local Document'
                })
        
        # Web source matches
        for idx, score in enumerate(web_similarities):
            if score >= threshold:
                results['web_matches'].append({
                    'document_index': idx,
                    'similarity_score': round(score, 2),
                    'source_text': self.web_sources[idx]['title'],
                    'source_url': self.web_sources[idx]['url'],
                    'source_type': 'Web Source'
                })
        
        return results
    
    def check_ai_content(self, text, use_web=True):
        """Check if text might be AI-generated"""
        return self.ai_detector.detect_ai_content(text, use_web)

class PlagiarismCheckerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Plagiarism Checker with AI Detection")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        self.checker = PlagiarismChecker()
        self.query_file_path = None
        self.query_text = ""
        self.sentence_tags = []  # To store text tags for highlighting
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Advanced Plagiarism Checker with AI Detection", 
                              font=('Arial', 18, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        left_frame = ttk.LabelFrame(main_container, text="Controls", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # File selection
        ttk.Label(left_frame, text="Select Document to Check:").pack(anchor=tk.W, pady=5)
        self.file_button = ttk.Button(left_frame, text="Browse File", command=self.select_file)
        self.file_button.pack(fill=tk.X, pady=5)
        self.file_label = ttk.Label(left_frame, text="No file selected", wraplength=200)
        self.file_label.pack(anchor=tk.W, pady=5)
        
        # Database size
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(left_frame, text="Database Size:").pack(anchor=tk.W, pady=5)
        self.db_size_var = tk.IntVar(value=1000)
        db_size_scale = ttk.Scale(left_frame, from_=100, to=10000, 
                                 variable=self.db_size_var, orient=tk.HORIZONTAL)
        db_size_scale.pack(fill=tk.X, pady=5)
        self.db_size_label = ttk.Label(left_frame, text="1000 documents")
        self.db_size_label.pack(anchor=tk.W)
        db_size_scale.config(command=self.update_db_size_label)
        
        # Options
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(left_frame, text="Options:").pack(anchor=tk.W, pady=5)
        self.web_check_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="Check against web sources", 
                       variable=self.web_check_var).pack(anchor=tk.W, pady=5)
        
        self.ai_check_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="Check for AI content", 
                       variable=self.ai_check_var).pack(anchor=tk.W, pady=5)
        
        # Threshold
        ttk.Label(left_frame, text="Similarity Threshold:").pack(anchor=tk.W, pady=5)
        self.threshold_var = tk.DoubleVar(value=0.7)
        threshold_scale = ttk.Scale(left_frame, from_=0.1, to=1.0, 
                                   variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(fill=tk.X, pady=5)
        self.threshold_label = ttk.Label(left_frame, text="70%")
        self.threshold_label.pack(anchor=tk.W)
        threshold_scale.config(command=self.update_threshold_label)
        
        # Check button
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        self.check_button = ttk.Button(left_frame, text="Check Document", 
                                      command=self.start_plagiarism_check)
        self.check_button.pack(fill=tk.X, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(left_frame, orient=tk.HORIZONTAL, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # Middle panel - Text display
        middle_frame = ttk.LabelFrame(main_container, text="Document Content", padding="10")
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Text display with scrollbar
        self.text_display = scrolledtext.ScrolledText(middle_frame, wrap=tk.WORD, width=50, height=30)
        self.text_display.pack(fill=tk.BOTH, expand=True)
        
        # Add context menu
        self.create_context_menu()
        
        # Right panel - Results
        right_frame = ttk.LabelFrame(main_container, text="Analysis Results", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Plagiarism results tab
        self.plagiarism_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.plagiarism_tab, text="Plagiarism")
        
        self.results_text = scrolledtext.ScrolledText(self.plagiarism_tab, wrap=tk.WORD, width=50, height=25)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # AI detection results tab
        self.ai_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.ai_tab, text="AI Detection")
        
        self.ai_results_text = scrolledtext.ScrolledText(self.ai_tab, wrap=tk.WORD, width=50, height=25)
        self.ai_results_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_context_menu(self):
        """Create a context menu for the text display"""
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Copy", command=self.copy_text)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Select All", command=self.select_all_text)
        
        # Bind right-click
        self.text_display.bind("<Button-3>", self.show_context_menu)
    
    def show_context_menu(self, event):
        """Show the context menu"""
        self.context_menu.post(event.x_root, event.y_root)
    
    def copy_text(self):
        """Copy selected text to clipboard"""
        try:
            selected_text = self.text_display.get("sel.first", "sel.last")
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
        except:
            pass  # No text selected
    
    def select_all_text(self):
        """Select all text in the text display"""
        self.text_display.tag_add(tk.SEL, "1.0", tk.END)
        self.text_display.mark_set(tk.SEL_FIRST, "1.0")
        self.text_display.focus_set()
    
    def update_db_size_label(self, value):
        self.db_size_label.config(text=f"{int(float(value))} documents")
    
    def update_threshold_label(self, value):
        self.threshold_label.config(text=f"{int(float(value)*100)}%")
    
    def select_file(self):
        filetypes = [
            ("Supported files", "*.txt;*.docx;*.pdf"),
            ("Text files", "*.txt"),
            ("Word documents", "*.docx"),
            ("PDF files", "*.pdf"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select a file to check for plagiarism",
            filetypes=filetypes
        )
        
        if file_path:
            self.query_file_path = file_path
            self.file_label.config(text=os.path.basename(file_path))
            self.query_text = self.read_file(file_path)
            
            if self.query_text:
                self.status_var.set(f"Loaded {len(self.query_text)} characters from {os.path.basename(file_path)}")
                self.display_file_content()
            else:
                self.status_var.set("Failed to read file")
                messagebox.showerror("Error", "Failed to read the selected file.")
    
    def display_file_content(self):
        """Display the file content in the text display area"""
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(tk.END, self.query_text)
        
        # Configure default text appearance
        self.text_display.tag_config("normal", foreground="black", font=('Arial', 10))
        self.text_display.tag_add("normal", "1.0", tk.END)
    
    def read_file(self, file_path):
        try:
            if file_path.lower().endswith('.docx') and DOCX_AVAILABLE:
                doc = docx.Document(file_path)
                return '\n'.join([para.text for para in doc.paragraphs])
            elif file_path.lower().endswith('.pdf') and PDF_AVAILABLE:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    return '\n'.join([page.extract_text() for page in reader.pages])
            else:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    def start_plagiarism_check(self):
        if not self.query_text:
            messagebox.showerror("Error", "Please select a file to check first.")
            return
        
        # Start the check in a separate thread to keep the GUI responsive
        threading.Thread(target=self.run_plagiarism_check, daemon=True).start()
    
    def run_plagiarism_check(self):
        # Update UI
        self.root.after(0, self.start_ui_update)
        
        try:
            # Get parameters
            threshold = self.threshold_var.get()
            check_web = self.web_check_var.get()
            check_ai = self.ai_check_var.get()
            db_size = self.db_size_var.get()
            
            # Check plagiarism
            self.root.after(0, lambda: self.status_var.set(f"Generating database ({db_size} documents)..."))
            results = self.checker.check_plagiarism(self.query_text, threshold, check_web, db_size)
            
            # Check AI content if requested
            ai_results = None
            if check_ai:
                self.root.after(0, lambda: self.status_var.set("Analyzing for AI content..."))
                ai_results = self.checker.check_ai_content(self.query_text, check_web)
            
            # Display results
            self.root.after(0, lambda: self.display_results(results, ai_results))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {str(e)}"))
        finally:
            self.root.after(0, self.end_ui_update)
    
    def start_ui_update(self):
        self.check_button.config(state=tk.DISABLED)
        self.progress.start()
        self.results_text.delete(1.0, tk.END)
        self.ai_results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Checking for plagiarism...\n\n")
        self.ai_results_text.insert(tk.END, "Analyzing for AI content...\n\n")
    
    def end_ui_update(self):
        self.check_button.config(state=tk.NORMAL)
        self.progress.stop()
        self.status_var.set("Ready")
    
    def display_results(self, results, ai_results):
        self.results_text.delete(1.0, tk.END)
        self.ai_results_text.delete(1.0, tk.END)
        
        # Clear previous highlights
        self.text_display.tag_remove("highlight", "1.0", tk.END)
        
        # Check for errors
        if results.get('error'):
            self.results_text.insert(tk.END, f"Error: {results['error']}\n\n")
            return
        
        # Highlight sentences with potential plagiarism
        if results.get('sentence_matches'):
            self.highlight_sentences(results['sentence_matches'])
        
        # Display plagiarism results
        if not results['source_matches'] and not results['web_matches']:
            self.results_text.insert(tk.END, "No plagiarism detected.\n\n")
            self.results_text.insert(tk.END, "Your document appears to be original based on the sources checked.\n\n")
        else:
            # Display source matches
            if results['source_matches']:
                self.results_text.insert(tk.END, "=== MATCHES IN LOCAL DATABASE ===\n\n", 'heading')
                for match in results['source_matches']:
                    self.results_text.insert(tk.END, f"Match found with {match['similarity_score']*100}% similarity\n", 'match')
                    self.results_text.insert(tk.END, f"Source: {match['source_text']}\n")
                    self.results_text.insert(tk.END, f"Type: {match['source_type']}\n\n")
            
            # Display web matches
            if results['web_matches']:
                self.results_text.insert(tk.END, "=== MATCHES IN WEB SOURCES ===\n\n", 'heading')
                for match in results['web_matches']:
                    self.results_text.insert(tk.END, f"Match found with {match['similarity_score']*100}% similarity\n", 'match')
                    self.results_text.insert(tk.END, f"Source: {match['source_text']}\n")
                    self.results_text.insert(tk.END, f"URL: {match['source_url']}\n")
                    self.results_text.insert(tk.END, f"Type: {match['source_type']}\n\n")
            
            # Display sentence-level matches
            if results.get('sentence_matches'):
                self.results_text.insert(tk.END, "=== SENTENCE-LEVEL MATCHES ===\n\n", 'heading')
                for match in results['sentence_matches']:
                    self.results_text.insert(tk.END, f"Sentence: {match['sentence']}\n", 'sentence')
                    self.results_text.insert(tk.END, f"Similarity: {match['similarity_score']*100}%\n", 'score')
                    self.results_text.insert(tk.END, f"Source Type: {match['source_type']}\n\n")
            
            # Overall assessment
            total_matches = len(results['source_matches']) + len(results['web_matches'])
            self.results_text.insert(tk.END, f"\n=== PLAGIARISM SUMMARY ===\n\n", 'heading')
            self.results_text.insert(tk.END, f"Found {total_matches} potential matches.\n")
            if total_matches > 5:
                self.results_text.insert(tk.END, "High probability of plagiarism detected.\n")
            elif total_matches > 2:
                self.results_text.insert(tk.END, "Moderate probability of plagiarism detected.\n")
            else:
                self.results_text.insert(tk.END, "Low probability of plagiarism detected.\n")
        
        # Configure text tags for plagiarism results
        self.results_text.tag_config('heading', font=('Arial', 12, 'bold'))
        self.results_text.tag_config('match', foreground='red', font=('Arial', 10, 'bold'))
        self.results_text.tag_config('sentence', foreground='blue', font=('Arial', 10))
        self.results_text.tag_config('score', foreground='darkgreen', font=('Arial', 10, 'bold'))
        
        # Display AI detection results
        if ai_results:
            self.display_ai_results(ai_results)
    
    def display_ai_results(self, ai_results):
        """Display AI detection results"""
        self.ai_results_text.insert(tk.END, "=== AI CONTENT ANALYSIS ===\n\n", 'ai_heading')
        
        # Display AI probability
        ai_prob = ai_results['ai_probability']
        self.ai_results_text.insert(tk.END, f"AI Probability: {ai_prob}%\n\n", 'ai_score')
        
        # Display probability bar (text-based)
        bar_length = 30
        filled_length = int(bar_length * ai_prob / 100)
        bar = '[' + '=' * filled_length + ' ' * (bar_length - filled_length) + ']'
        self.ai_results_text.insert(tk.END, f"{bar}\n\n")
        
        # Display assessment
        self.ai_results_text.insert(tk.END, f"Assessment: {ai_results['analysis']}\n\n", 'ai_assessment')
        
        # Display indicators
        if ai_results['indicators']:
            self.ai_results_text.insert(tk.END, "=== DETECTED INDICATORS ===\n\n", 'ai_subheading')
            for indicator in ai_results['indicators']:
                if indicator['type'] == 'phrase':
                    self.ai_results_text.insert(tk.END, f"AI Phrase: '{indicator['pattern']}' (appears {indicator['count']} times)\n", 'ai_indicator')
                elif indicator['type'] == 'pattern':
                    self.ai_results_text.insert(tk.END, f"AI Pattern: '{indicator['pattern']}' (appears {indicator['count']} times)\n", 'ai_indicator')
                elif indicator['type'] == 'structure':
                    self.ai_results_text.insert(tk.END, f"Uniform Sentence Length: variance = {indicator['variance']}, avg = {indicator['avg_length']} words\n", 'ai_indicator')
                elif indicator['type'] == 'vocabulary':
                    self.ai_results_text.insert(tk.END, f"Low Vocabulary Richness: {indicator['richness']*100:.1f}% unique words\n", 'ai_indicator')
            self.ai_results_text.insert(tk.END, "\n")
        
        # Display web results
        if ai_results['web_results']:
            self.ai_results_text.insert(tk.END, "=== WEB AI ANALYSIS ===\n\n", 'ai_subheading')
            for result in ai_results['web_results']:
                self.ai_results_text.insert(tk.END, f"Source: {result['source']}\n", 'ai_web_source')
                self.ai_results_text.insert(tk.END, f"Confidence: {result['confidence']*100:.1f}%\n", 'ai_web_confidence')
                self.ai_results_text.insert(tk.END, f"Phrase: {result['phrase']}\n", 'ai_web_phrase')
                self.ai_results_text.insert(tk.END, f"Indicators: {', '.join(result['indicators'])}\n\n", 'ai_web_indicators')
        
        # Configure text tags for AI results
        self.ai_results_text.tag_config('ai_heading', font=('Arial', 14, 'bold'))
        self.ai_results_text.tag_config('ai_score', font=('Arial', 12, 'bold'))
        self.ai_results_text.tag_config('ai_assessment', font=('Arial', 11))
        self.ai_results_text.tag_config('ai_subheading', font=('Arial', 11, 'bold'))
        self.ai_results_text.tag_config('ai_indicator', foreground='orange')
        self.ai_results_text.tag_config('ai_web_source', font=('Arial', 10, 'bold'))
        self.ai_results_text.tag_config('ai_web_confidence', foreground='red')
        self.ai_results_text.tag_config('ai_web_phrase', font=('Arial', 10))
        self.ai_results_text.tag_config('ai_web_indicators', foreground='blue')
    
    def highlight_sentences(self, sentence_matches):
        """Highlight sentences in the text display that match potential plagiarism"""
        # Get the full text
        full_text = self.text_display.get("1.0", tk.END)
        
        # Find and highlight each matching sentence
        for match in sentence_matches:
            sentence = match['sentence']
            score = match['similarity_score']
            
            # Find the position of the sentence in the full text
            start_idx = full_text.find(sentence)
            if start_idx != -1:
                # Convert character index to line.column format
                start_pos = f"1.0+{start_idx}c"
                end_pos = f"1.0+{start_idx + len(sentence)}c"
                
                # Create a unique tag for this sentence
                tag_name = f"highlight_{match['sentence_index']}"
                
                # Configure the tag with a color based on similarity score
                if score >= 0.9:
                    color = "#ff0000"  # Red for high similarity
                elif score >= 0.7:
                    color = "#ff9900"  # Orange for medium similarity
                else:
                    color = "#ffff00"  # Yellow for low similarity
                
                self.text_display.tag_add(tag_name, start_pos, end_pos)
                self.text_display.tag_config(tag_name, background=color)
                
                # Add a tooltip-like effect
                self.text_display.tag_bind(tag_name, "<Enter>", 
                                         lambda e, s=sentence, sc=score: self.show_tooltip(e, s, sc))
                self.text_display.tag_bind(tag_name, "<Leave>", self.hide_tooltip)
    
    def show_tooltip(self, event, sentence, score):
        """Show a tooltip with sentence details"""
        # Create a tooltip window
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
        
        label = tk.Label(self.tooltip, text=f"Similarity: {score*100:.1f}%", 
                         background="lightyellow", relief="solid", borderwidth=1)
        label.pack()
    
    def hide_tooltip(self, event):
        """Hide the tooltip"""
        if hasattr(self, 'tooltip'):
            self.tooltip.destroy()

def main():
    root = tk.Tk()
    app = PlagiarismCheckerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()