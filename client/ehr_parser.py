import re
import os
from pypdf import PdfReader
from docx import Document

class EHRParser:
    def __init__(self):
        # Regex patterns for temperature extraction
        # e.g., "Temp: 101.2 F", "Temperature: 38.5 C", "Body Temp: 99.8"
        self.temp_pattern = re.compile(
            r'(?:temp(?:erature)?|body\s+temp)\s*(?::|is|=)?\s*(\d{2,3}(?:\.\d+)?)\s*(?:°?\s*[cfCF])?',
            re.IGNORECASE
        )
        
        # Regex patterns for dengue diagnosis status
        # e.g., "Dengue: Positive", "NS1 Antigen - POSITIVE", "Dengue IgM: detected"
        self.dengue_pos_patterns = [
            re.compile(r'dengue.*(?:positive|detected|reactive)', re.IGNORECASE),
            re.compile(r'ns1.*(?:positive|detected|reactive)', re.IGNORECASE),
            re.compile(r'igm.*(?:positive|detected|reactive)', re.IGNORECASE),
            re.compile(r'dengue\s+fever\s+confirmed', re.IGNORECASE)
        ]
        self.dengue_neg_patterns = [
            re.compile(r'dengue.*(?:negative|not\s+detected|non-reactive)', re.IGNORECASE),
            re.compile(r'ns1.*(?:negative|not\s+detected|non-reactive)', re.IGNORECASE),
            re.compile(r'igm.*(?:negative|not\s+detected|non-reactive)', re.IGNORECASE)
        ]

    def extract_text_from_pdf(self, file_path):
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""

    def extract_text_from_docx(self, file_path):
        try:
            doc = Document(file_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
            return ""

    def extract_text_from_file(self, file_path):
        _, ext = os.path.splitext(file_path.lower())
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_path)
        else:
            # Fallback to standard text reading
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading text file {file_path}: {e}")
                return ""

    def parse_ehr(self, text):
        """
        Parses the EHR text and extracts:
        - temperature: float (normalized to Celsius)
        - dengue_status: int (1 for Positive, 0 for Negative, -1 for Undefined)
        """
        # 1. Temperature parsing
        temp = None
        temp_match = self.temp_pattern.search(text)
        if temp_match:
            try:
                raw_temp = float(temp_match.group(1))
                # Check unit from text around match
                context = text[max(0, temp_match.start() - 5):min(len(text), temp_match.end() + 10)].lower()
                
                # Default is Fahrenheit if > 45, otherwise Celsius
                is_celsius = 'c' in context and 'f' not in context
                if raw_temp > 45 and not is_celsius:
                    # Convert F to C
                    temp = (raw_temp - 32) * 5.0 / 9.0
                else:
                    temp = raw_temp
            except ValueError:
                pass
        
        # Default fallback if temp not parsed
        if temp is None:
            temp = 37.0 # Normal body temp in C

        # 2. Dengue status parsing
        dengue_status = -1
        
        # Check positives first
        for pattern in self.dengue_pos_patterns:
            if pattern.search(text):
                dengue_status = 1
                break
        
        # If not positive, check negatives
        if dengue_status == -1:
            for pattern in self.dengue_neg_patterns:
                if pattern.search(text):
                    dengue_status = 0
                    break
                    
        return {
            "temperature_c": round(temp, 2),
            "dengue_status": dengue_status
        }
