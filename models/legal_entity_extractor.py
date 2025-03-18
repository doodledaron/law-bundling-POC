import re
import logging
from typing import Dict, List, Any, Optional

# Configure logger
logger = logging.getLogger(__name__)

class LegalEntityExtractor:
    """
    Specialized entity extractor for legal documents, building on LayoutLM output.
    """
    
    def __init__(self):
        """Initialize the legal entity extractor"""
        logger.info("Initializing Legal Entity Extractor")
        
        # Entity patterns for common legal document fields
        self.patterns = {
            # Common patterns for legal documents
            'parties': [
                r'(?:between|party of the first part):\s*(.*?)(?=\s*\(|and|,|\n)',
                r'(?:and|party of the second part):\s*(.*?)(?=\s*\(|and|,|\n)'
            ],
            'dates': [
                r'(?:dated|effective date|as of)[\s:]+(\d{1,2}[thstndrd]*\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4})',
                r'(\d{1,2}[thstndrd]*\s+(?:day\s+of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4})'
            ],
            'clause_headers': [
                r'(\d+\.?\s+[A-Z][A-Za-z\s]+)(?:\.|:)',
                r'((?:Article|Section|Clause)\s+\d+\.?\s*[A-Z][A-Za-z\s]+)(?:\.|:)'
            ],
            'signatures': [
                r'(?:signature|signed|executed).*?:\s*(.*?)(?=\s*\n)',
                r'(?:By|Name):\s*(.*?)(?=\s*\n)'
            ],
            'governing_law': [
                r'(?:governed by|construed in accordance with).*?(?:laws of|law of)\s+(.*?)(?=\.|,|\n)',
            ],
            'confidentiality': [
                r'(?:confidential information|confidentiality).*?(?:defined as|shall mean|refers to)\s+(.*?)(?=\.|,|\n)'
            ],
            'duration': [
                r'(?:term|duration|period).*?(?:of|for)\s+(\d+(?:\.\d+)?)\s+(?:year|month|day|week)s?',
                r'(?:shall remain in effect for|shall continue for|shall remain in force for)\s+(\d+(?:\.\d+)?)\s+(?:year|month|day|week)s?'
            ]
        }
    
    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract legal entities from the document text.
        
        Args:
            text: The document text to extract entities from
            
        Returns:
            Dictionary containing extracted entities
        """
        logger.info("Extracting legal entities from text")
        
        # Initialize results dictionary
        results = {
            'parties': [],
            'dates': [],
            'clause_headers': [],
            'signatures': [],
            'governing_law': [],
            'confidentiality': [],
            'duration': []
        }
        
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Extract entities using patterns
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    for match in matches:
                        match = match.strip() if isinstance(match, str) else match
                        if match and match not in results[entity_type]:
                            results[entity_type].append(match)
        
        logger.info(f"Extracted entities: {sum(len(v) for v in results.values())} total")
        return results
    
    def extract_with_layout(self, text: str, layout_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract legal entities using both text and layout information.
        
        Args:
            text: The document text
            layout_info: Layout information from LayoutLM
            
        Returns:
            Dictionary containing extracted entities
        """
        logger.info("Extracting legal entities with layout information")
        
        # Get basic extractions from text
        results = self.extract_from_text(text)
        
        # Enhance extractions using layout information
        # This would use the spatial information to better identify entities
        # For example, signatures are typically at the bottom of the document,
        # parties at the top, and clauses in the middle.
        
        # This is a placeholder for the layout-enhanced extraction logic
        
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better entity extraction.
        
        Args:
            text: The text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove any non-breaking spaces or special unicode whitespace
        text = text.replace('\u00A0', ' ')
        
        # Normalize quotes
        text = text.replace(''', "'").replace(''', "'").replace('"', '"').replace('"', '"')
        
        return text 