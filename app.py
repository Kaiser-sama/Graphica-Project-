from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import re
import os
import io
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class PDFSummarizer:
    def __init__(self):
        """Initialize PDF Summarizer"""
        self.setup_nltk()
    
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file object"""
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
                    
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
            
        return text
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep sentence endings
        text = re.sub(r'[^\w\s\.\!\?]', '', text)
        
        return text.strip()
    
    def extractive_summarization(self, text, num_sentences=3):
        """Extractive summarization using sentence scoring"""
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Tokenize into words and remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
            
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]
        
        # Calculate word frequencies
        word_freq = Counter(words)
        
        # Score sentences based on word frequencies
        sentence_scores = {}
        for sentence in sentences:
            sentence_words = word_tokenize(sentence.lower())
            sentence_words = [word for word in sentence_words if word.isalnum()]
            
            score = 0
            word_count = 0
            
            for word in sentence_words:
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            
            if word_count > 0:
                sentence_scores[sentence] = score / word_count
        
        # Get top sentences
        top_sentences = sorted(sentence_scores.items(), 
                             key=lambda x: x[1], 
                             reverse=True)[:num_sentences]
        
        # Sort by original order in text
        summary_sentences = []
        for sentence in sentences:
            for top_sentence, _ in top_sentences:
                if sentence == top_sentence:
                    summary_sentences.append(sentence)
                    break
        
        return ' '.join(summary_sentences)
    
    def summarize_pdf(self, pdf_file, num_sentences=3):
        """Main method to summarize a PDF"""
        try:
            print("Extracting text from PDF...")
            text = self.extract_text_from_pdf(pdf_file)
            
            if not text.strip():
                return {
                    'success': False,
                    'error': 'Could not extract text from PDF',
                    'summary': '',
                    'word_count': 0,
                    'method': 'extractive'
                }
            
            print("Cleaning text...")
            cleaned_text = self.clean_text(text)
            word_count = len(cleaned_text.split())
            
            print("Generating extractive summary...")
            summary = self.extractive_summarization(cleaned_text, num_sentences)
            
            return {
                'success': True,
                'summary': summary,
                'word_count': word_count,
                'original_length': len(text),
                'summary_length': len(summary),
                'method': 'extractive'
            }
            
        except Exception as e:
            print(f"Error in summarization: {e}")
            return {
                'success': False,
                'error': str(e),
                'summary': '',
                'word_count': 0,
                'method': 'extractive'
            }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

# Routes
@app.route('/')
def home():
    return jsonify({
        'message': 'Diddy Party PDF Summarizer API',
        'status': 'running',
        'endpoints': {
            '/summarize': 'POST - Upload PDF and get summary',
            '/summarize_text': 'POST - Summarize text directly',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'API is running'})

@app.route('/summarize', methods=['POST'])
def summarize_pdf():
    """Endpoint to upload and summarize PDF"""
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Only PDF files are allowed'}), 400
        
        # Get parameters
        num_sentences = int(request.form.get('num_sentences', 3))
        
        # Initialize summarizer
        summarizer = PDFSummarizer()
        
        # Create a temporary file-like object
        pdf_content = io.BytesIO(file.read())
        
        # Generate summary
        result = summarizer.summarize_pdf(pdf_content, num_sentences)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in summarize endpoint: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/summarize_text', methods=['POST'])
def summarize_text():
    """Endpoint to summarize text directly (without PDF)"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': 'No text provided'}), 400
        
        text = data['text']
        num_sentences = data.get('num_sentences', 3)
        method = data.get('method', 'extractive')
        
        if not text.strip():
            return jsonify({'success': False, 'error': 'Empty text provided'}), 400
        
        # Initialize summarizer
        summarizer = PDFSummarizer()
        
        # Clean text
        cleaned_text = summarizer.clean_text(text)
        word_count = len(cleaned_text.split())
        
        # Generate summary
        summary = summarizer.extractive_summarization(cleaned_text, num_sentences)
        
        return jsonify({
            'success': True,
            'summary': summary,
            'word_count': word_count,
            'original_length': len(text),
            'summary_length': len(summary),
            'method': method
        })
        
    except Exception as e:
        print(f"Error in summarize_text endpoint: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🎉 Starting Diddy Party PDF Summarizer Flask App...")
    print("Available endpoints:")
    print("- POST /summarize - Upload PDF file for summarization")
    print("- POST /summarize_text - Summarize text directly")
    print("- GET /health - Health check")
    print("- GET / - API info")
    print("\n🚀 Server will run on http://localhost:5000")
    print("💡 Make sure to install dependencies: pip install flask flask-cors PyPDF2 nltk")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)