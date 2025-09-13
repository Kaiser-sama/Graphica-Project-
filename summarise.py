import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import re
from transformers import pipeline
import argparse
import os

class PDFSummarizer:
    def __init__(self, method='extractive'):
        """
        Initialize PDF Summarizer
        
        Args:
            method (str): 'extractive' or 'abstractive'
        """
        self.method = method
        self.setup_nltk()
        
        if method == 'abstractive':
            print("Loading BART model for abstractive summarization...")
            self.summarizer = pipeline("summarization", 
                                     model="facebook/bart-large-cnn",
                                     device=-1)  # Use CPU (-1) or GPU (0)
    
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF file
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            str: Extracted text
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                    
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
            
        return text
    
    def clean_text(self, text):
        """
        Clean and preprocess text
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep sentence endings
        text = re.sub(r'[^\w\s\.\!\?]', '', text)
        
        return text.strip()
    
    def extractive_summarization(self, text, num_sentences=3):
        """
        Extractive summarization using sentence scoring
        
        Args:
            text (str): Input text
            num_sentences (int): Number of sentences in summary
            
        Returns:
            str: Summary
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Tokenize into words and remove stopwords
        stop_words = set(stopwords.words('english'))
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
    
    def abstractive_summarization(self, text, max_length=150, min_length=50):
        """
        Abstractive summarization using BART model
        
        Args:
            text (str): Input text
            max_length (int): Maximum summary length
            min_length (int): Minimum summary length
            
        Returns:
            str: Summary
        """
        # Split text into chunks if too long (BART has token limits)
        max_chunk_length = 1024
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_chunk_length):
            chunk = ' '.join(words[i:i + max_chunk_length])
            chunks.append(chunk)
        
        summaries = []
        for chunk in chunks:
            try:
                summary = self.summarizer(chunk, 
                                        max_length=max_length, 
                                        min_length=min_length, 
                                        do_sample=False)
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                print(f"Error in abstractive summarization: {e}")
                # Fallback to extractive
                summaries.append(self.extractive_summarization(chunk, 2))
        
        return ' '.join(summaries)
    
    def summarize_pdf(self, pdf_path, num_sentences=3, max_length=150):
        """
        Main method to summarize a PDF
        
        Args:
            pdf_path (str): Path to PDF file
            num_sentences (int): Number of sentences for extractive method
            max_length (int): Max length for abstractive method
            
        Returns:
            str: Summary
        """
        print(f"Extracting text from {pdf_path}...")
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            return "Could not extract text from PDF."
        
        print("Cleaning text...")
        cleaned_text = self.clean_text(text)
        
        print(f"Generating {self.method} summary...")
        
        if self.method == 'extractive':
            summary = self.extractive_summarization(cleaned_text, num_sentences)
        else:
            summary = self.abstractive_summarization(cleaned_text, max_length)
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='PDF Summarizer')
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('--method', choices=['extractive', 'abstractive'], 
                       default='extractive', help='Summarization method')
    parser.add_argument('--sentences', type=int, default=3, 
                       help='Number of sentences for extractive method')
    parser.add_argument('--max_length', type=int, default=150,
                       help='Maximum length for abstractive method')
    parser.add_argument('--output', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: File {args.pdf_path} not found.")
        return
    
    # Initialize summarizer
    summarizer = PDFSummarizer(method=args.method)
    
    # Generate summary
    summary = summarizer.summarize_pdf(args.pdf_path, 
                                     args.sentences, 
                                     args.max_length)
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    print(summary)
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"\nSummary saved to {args.output}")

if __name__ == "__main__":
    main()