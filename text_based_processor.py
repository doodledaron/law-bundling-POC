import os
import base64
from typing import Dict, List, Optional, Union
from datetime import datetime
import re
import io
from google import genai
from google.genai import types
from PIL import Image
from config import Config

class TextBasedProcessor:
    # Costing constants for Gemini 2.0 Flash (user-provided rates, per token)
    INPUT_COST_PER_M_TOKENS = 0.10    # USD ($0.10 per 1M input tokens)
    OUTPUT_COST_PER_M_TOKENS = 0.40   # USD ($0.40 per 1M output tokens)

    def __init__(self):
        self.client = genai.Client(
            api_key=Config.GEMINI_API_KEY,
            vertexai=False
        )
        self.model = Config.MODEL_NAME
        self.generation_config = Config.GENERATION_CONFIG
    
    def _safe_extract_text(self, response) -> str:
        """Safely extract text from Gemini response."""
        if response is None:
            return "No response received from AI model"
        
        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        elif hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                return part.text.strip()
        
        return "Unable to extract text from AI response"
    
    def _extract_token_usage(self, response) -> Dict:
        """Extract token usage information from Gemini response."""
        usage_info = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
        
        try:
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                if hasattr(response.usage_metadata, 'prompt_token_count'):
                    usage_info["input_tokens"] = response.usage_metadata.prompt_token_count
                if hasattr(response.usage_metadata, 'candidates_token_count'):
                    usage_info["output_tokens"] = response.usage_metadata.candidates_token_count
                if hasattr(response.usage_metadata, 'total_token_count'):
                    usage_info["total_tokens"] = response.usage_metadata.total_token_count
                else:
                    usage_info["total_tokens"] = usage_info["input_tokens"] + usage_info["output_tokens"]
        except Exception as e:
            # If token extraction fails, just continue without token info
            pass
        
        return usage_info
    
    def _calculate_estimated_cost(self, input_tokens: int, output_tokens: int, num_images: int = 0) -> float:
        """Calculate the estimated cost for processing based on tokens and images."""
        cost_input_text = (input_tokens / 1000000.0) * self.INPUT_COST_PER_M_TOKENS
        cost_output_text = (output_tokens / 1000000.0) * self.OUTPUT_COST_PER_M_TOKENS

        total_cost = cost_input_text + cost_output_text
        return round(total_cost, 6) # Round to 6 decimal places for currency
    
    def _parse_analysis_string(self, analysis_text: str) -> Dict[str, str]:
        """Parse the AI-generated analysis string into a structured dictionary."""
        parsed_data = {
            "SUMMARY": "Not available",
            "KEY DATES": "Not available",
            "ULTIMATE KEY DATE": "undated",
            "MAIN PARTIES": "Not available",
            "CASE/REFERENCE NUMBERS": "Not available"
        }

        # First, clean up the text by removing any introduction text
        analysis_text = re.sub(r"^.*?(?=\d+\s*\..*?SUMMARY|\d+\s*\.\s*\*\*SUMMARY\*\*|Here\'s|The following)", "", analysis_text, flags=re.DOTALL | re.IGNORECASE)
        
        # Try multiple extraction methods to handle different formats
        self._extract_from_numbered_sections(analysis_text, parsed_data)
        self._extract_from_markdown_style(analysis_text, parsed_data)
        self._extract_from_free_text(analysis_text, parsed_data)
        
        return parsed_data
        
    def _extract_from_numbered_sections(self, analysis_text: str, parsed_data: Dict[str, str]):
        """Extract data from standard numbered sections format: '1. SUMMARY: text'"""
        # Regex to find numbered section headers with various formats
        header_pattern = re.compile(
            r"^\s*(?:\d+)\.\s*(?:\*\*)?([A-Z\s\/_'-]+(?:NUMBERS)?)(?:\*\*)?\s*(?:\([^\)]*\))?:\s*", 
            re.MULTILINE | re.IGNORECASE
        )
        matches = list(header_pattern.finditer(analysis_text))

        for i, match in enumerate(matches):
            # Full key name as it appears after the number
            key_name_from_match = match.group(1).strip().upper()
            
            current_key_normalized = None
            # Normalize to one of our standard internal keys
            for standard_key in parsed_data.keys():
                if key_name_from_match.startswith(standard_key.upper()):
                    current_key_normalized = standard_key
                    break
            
            if not current_key_normalized:
                continue

            start_of_value = match.end()
            end_of_value = len(analysis_text)
            if i + 1 < len(matches):  # If there's a next section header
                end_of_value = matches[i+1].start()
            
            value = analysis_text[start_of_value:end_of_value].strip()
            value = self._clean_field_value(current_key_normalized, value)
            
            # Update parsed_data if value is meaningful
            if value and value != "Not available" and not (current_key_normalized == "ULTIMATE KEY DATE" and value == "undated" and parsed_data[current_key_normalized] != "undated"):
                parsed_data[current_key_normalized] = value
    
    def _extract_from_markdown_style(self, analysis_text: str, parsed_data: Dict[str, str]):
        """Extract data from markdown-style formatting with bold headers: '**SUMMARY:** text'"""
        # Patterns for each field type with markdown-style formatting
        field_patterns = {
            "SUMMARY": r"\*\*SUMMARY(?:\s*\([^)]*\))?\:\*\*\s*(.*?)(?=\n\s*\*\*|\Z)",
            "KEY DATES": r"\*\*KEY DATES\:\*\*\s*(.*?)(?=\n\s*\*\*|\Z)",
            "ULTIMATE KEY DATE": r"\*\*ULTIMATE KEY DATE\:\*\*\s*(.*?)(?=\n\s*\*\*|\Z)",
            "MAIN PARTIES": r"\*\*MAIN PARTIES\:\*\*\s*(.*?)(?=\n\s*\*\*|\Z)",
            "CASE/REFERENCE NUMBERS": r"\*\*CASE(?:\/|\s+)REFERENCE NUMBERS\:\*\*\s*(.*?)(?=\n\s*\*\*|\Z)"
        }
        
        # Try to extract each field
        for field, pattern in field_patterns.items():
            match = re.search(pattern, analysis_text, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                value = self._clean_field_value(field, value)
                
                # Update if we have a meaningful value
                if value and value != "Not available" and not (field == "ULTIMATE KEY DATE" and value == "undated" and parsed_data[field] != "undated"):
                    parsed_data[field] = value
    
    def _extract_from_free_text(self, analysis_text: str, parsed_data: Dict[str, str]):
        """Extract data from free-text format by looking for key phrases"""
        # Look for keywords/phrases that indicate field content in unstructured text
        
        # SUMMARY
        if parsed_data["SUMMARY"] == "Not available":
            summary_indicators = [
                r"(?:document|this|the)\s+(?:is|appears to be)\s+(?:a|an)\s+(.*?)(?:\.|\n|$)",
                r"(?:this|the)\s+document\s+(?:contains|presents|shows|displays)\s+(.*?)(?:\.|\n|$)",
                r"(?:this|the)\s+(?:is|appears to be)\s+(?:a|an)\s+(.*?)(?:from|by|between|dated).*?(?:\.|\n|$)"
            ]
            
            for pattern in summary_indicators:
                match = re.search(pattern, analysis_text, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    if len(value.split()) <= 30:  # Reasonable length for summary
                        parsed_data["SUMMARY"] = value
                        break
        
        # KEY DATES
        if parsed_data["KEY DATES"] == "Not available":
            dates_patterns = [
                r"(?:key|important|significant)\s+dates?.*?(?:include|are|:)\s+(.*?)(?=\n\n|\n[A-Z]|\Z)",
                r"(?:dates?|temporal references).*?(?:found|mentioned|:)\s+(.*?)(?=\n\n|\n[A-Z]|\Z)"
            ]
            
            for pattern in dates_patterns:
                match = re.search(pattern, analysis_text, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    value = self._clean_field_value("KEY DATES", value)
                    if value:
                        parsed_data["KEY DATES"] = value
                        break
        
        # ULTIMATE KEY DATE
        if parsed_data["ULTIMATE KEY DATE"] == "undated":
            date_patterns = [
                r"(?:primary|main|ultimate|key)\s+date.*?(?:is|:)\s+(.*?)(?=\n\n|\n[A-Z]|\Z)",
                r"(?:document|contract|agreement)\s+(?:is|was)\s+dated\s+(.*?)(?=\n|\.|$)",
                r"(?:dated|date[d:])(?:\s+(?:on|as))?\s+(.*?)(?=\n|\.|$)"
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, analysis_text, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    value = self._clean_field_value("ULTIMATE KEY DATE", value)
                    if value and value != "undated":
                        parsed_data["ULTIMATE KEY DATE"] = value
                        break
        
        # MAIN PARTIES
        if parsed_data["MAIN PARTIES"] == "Not available":
            parties_patterns = [
                r"(?:main|primary|key)\s+parties.*?(?:include|are|:)\s+(.*?)(?=\n\n|\n[A-Z]|\Z)",
                r"(?:parties|entities|individuals)\s+(?:involved|mentioned).*?(?:include|are|:)\s+(.*?)(?=\n\n|\n[A-Z]|\Z)",
                r"(?:between|involving)\s+(.*?)(?:and|,)\s+(.*?)(?=\n|\.|$)"
            ]
            
            for pattern in parties_patterns:
                match = re.search(pattern, analysis_text, re.IGNORECASE | re.DOTALL)
                if match:
                    if len(match.groups()) > 1:  # Multiple capture groups
                        value = f"{match.group(1).strip()} and {match.group(2).strip()}"
                    else:
                        value = match.group(1).strip()
                        
                    value = self._clean_field_value("MAIN PARTIES", value)
                    if value:
                        parsed_data["MAIN PARTIES"] = value
                        break
        
        # CASE/REFERENCE NUMBERS
        if parsed_data["CASE/REFERENCE NUMBERS"] == "Not available":
            ref_patterns = [
                r"(?:case|reference|file|docket)\s+(?:number|no|id|identifier).*?(?:is|are|:)\s+(.*?)(?=\n\n|\n[A-Z]|\Z)",
                r"(?:case|reference|file|docket).*?(?:number|no|id|identifier).*?(?:is|are|:)\s+(.*?)(?=\n\n|\n[A-Z]|\Z)",
                r"(?:identified|referenced)\s+(?:as|by|with).*?(?:number|no|id).*?(?:is|are|:)?\s+(.*?)(?=\n|\.|$)"
            ]
            
            for pattern in ref_patterns:
                match = re.search(pattern, analysis_text, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    value = self._clean_field_value("CASE/REFERENCE NUMBERS", value)
                    if value:
                        parsed_data["CASE/REFERENCE NUMBERS"] = value
                        break
                        
    def _clean_field_value(self, field_key: str, value: str) -> str:
        """Clean and normalize field values based on the field type"""
        if not value:
            return ""
            
        # Remove any trailing periods or other punctuation
        value = value.strip()
        value = re.sub(r"[.;,]+$", "", value).strip()
        
        # Specific cleaning for different field types
        if field_key == "ULTIMATE KEY DATE":
            if "undated" in value.lower():
                return "undated"
                
            # Remove instructional remnants
            value = re.sub(r"\.?\s*If no specific rule applies.*", "", value, flags=re.IGNORECASE | re.DOTALL).strip()
            
            # If it's now empty, return undated
            if not value:
                return "undated"
                
        elif field_key in ["SUMMARY", "KEY DATES", "MAIN PARTIES", "CASE/REFERENCE NUMBERS"]:
            # Remove instructional remnants
            value = re.sub(r"\.?\s*If none, state \"None\".*", "", value, flags=re.IGNORECASE | re.DOTALL).strip()
            
            # Check for "None" indicators
            if value.lower() in ["none", "not found", "not available", "n/a", "nil"]:
                return "None"
                
            # If it's now empty, return None
            if not value:
                return "None"
        
        return value
        
    def process_table_image(self, image_data: bytes, context: str = "") -> str:
        """
        Process an image containing a table and extract its content as structured text.
        
        Args:
            image_data: Bytes of the image containing the table
            context: Optional context about the table (e.g., caption, surrounding text)
            
        Returns:
            String representation of the table content
        """
        try:
            # Prepare the image for Gemini
            mime_type = "image/png"  # Assume PNG, but could be determined from image header
            image_part = types.Part.from_bytes(data=image_data, mime_type=mime_type)
            
            # Create prompt for table extraction
            prompt = f"""
Extract all information from this table image into a well-structured text format.
Follow these guidelines:
1. Preserve the table structure using plain text formatting
2. Include all headers, row labels, and cell values
3. Maintain alignment where possible for readability
4. If numbers have specific units, include them
5. If the table has a title or caption, include it at the beginning
6. For any unclear or illegible cells, indicate with [?]



Additional context about this table: {context}

Return ONLY the extracted table content, formatted as text.
"""
            
            # Generate content using Gemini
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image_part],
                config=types.GenerateContentConfig(**self.generation_config)
            )
            
            # Extract and return the table text
            table_text = self._safe_extract_text(response)
            
            # Format the output with a clear header
            formatted_output = f"TABLE CONTENT:\n{table_text}"
            
            return formatted_output
            
        except Exception as e:
            return f"TABLE EXTRACTION ERROR: Unable to process table image. Error: {str(e)}"
    
    def process_figure_image(self, image_data: bytes, context: str = "") -> str:
        """
        Process an image containing a figure and extract a description of its content.
        
        Args:
            image_data: Bytes of the image containing the figure
            context: Optional context about the figure (e.g., caption, surrounding text)
            
        Returns:
            String description of the figure content
        """
        try:
            # Prepare the image for Gemini
            mime_type = "image/png"  # Assume PNG, but could be determined from image header
            image_part = types.Part.from_bytes(data=image_data, mime_type=mime_type)
            
            # Create prompt for figure description
            prompt = f"""
Analyze this figure/image and provide a detailed description of its content.
Follow these guidelines:
1. Describe what the figure shows (chart, graph, diagram, illustration, etc.)
2. Identify the main components, labels, and any visible data points
3. Explain the key information or message conveyed by the figure
4. If it's a chart or graph, describe the trends, patterns, or relationships shown
5. Include any visible figure numbers, titles, or captions

Important considerations for figure analysis:
- Recognize that this may be a small fragment of a larger figure due to layout analysis
- If this appears to be a tiny part of a larger element (e.g., part of a signature, corner of a diagram):
  * State that it appears to be a fragment and identify what larger element it likely belongs to
  * Note if it contains insufficient information for a meaningful analysis on its own
  * If possible, suggest what the complete figure might be based on visible elements
- For signatures or handwritten elements:
  * If only a portion is visible, simply identify it as "partial signature" or similar
  * Don't attempt to identify the name if it's not clearly legible
- For figures that span multiple images:
  * Focus on describing the visible portion only
  * Indicate if the figure appears to continue beyond the boundaries of this image

Additional context about this figure: {context}

Return a clear, informative description that captures all important aspects of this figure.
"""
            
            # Generate content using Gemini
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image_part],
                config=types.GenerateContentConfig(**self.generation_config)
            )
            
            # Extract and return the figure description
            figure_description = self._safe_extract_text(response)
            
            # Format the output with a clear header
            formatted_output = f"FIGURE DESCRIPTION:\n{figure_description}"
            
            return formatted_output
            
        except Exception as e:
            return f"FIGURE EXTRACTION ERROR: Unable to process figure image. Error: {str(e)}"
    
    def summarize_document_text(self, document_text: str, filename: str) -> Dict:
        """
        Process and summarize document text content (including OCR text and extracted table/figure content).
        Uses the same prompt structure as document_processor.py.
        
        Args:
            document_text: Full text content of the document (OCR + table/figure extraction)
            filename: Name of the original document file
            
        Returns:
            Dictionary with document summary information
        """
        try:
            if not document_text.strip():
                return {
                    "file_type": "text",
                    "file_name": filename,
                    "summary": "Not available",
                    "date": "undated",
                    "extracted_info": {"full_analysis": "No text content could be extracted."},
                    "token_usage": self._extract_token_usage(None),
                    "estimated_cost": 0.0,
                    "error": "No text content could be extracted from the document",
                    "status": "error",
                    "processed_at": datetime.now().isoformat()
                }
            
            prompt = f"""

Analyze this text document. It can be of various types and may not contain all listed elements, summary is a must.

Pre-processing (apply before any other step):
  • OCR & LAYOUT ERROR COMPENSATION: 
      - Fix cut-off words when obvious (e.g., “eneral” → “General”).
      - Correct numeric OCR mistakes (e.g., “1ooo” → “1000”, “5O” → “50”).
      - Recognize concatenated entity names (e.g., “NINGBOKEYUANFINECHEMICALSCOLTD”) and split into proper words when clearly a person or organisation name (e.g., “NINGBO KEYUAN FINE CHEMICALS CO LTD”); if unsure or not clearly a name, leave as-is.
      - Ignore duplicated lines from overlapping layout boxes; if duplicates differ, include unique info.
      - Disregard page numbers, volume references, or other metadata (e.g., “ABOD Vol 1 Page 226”).
      - Verify that dates are reasonable (e.g., “32-Feb-2020” is invalid).
      - When in doubt, include the information rather than omit it.

1. SUMMARY (max 25 words): In a single concise sentence, state the document type and include all required fields for that type (e.g., Date, Title, Person (From), Organisation (To)), then mention the subject and primary case number if applicable. Keep it smooth and avoid introductory phrases.  
If multiple document-type indicators appear, try to combine them into an appropriate way of summarizing the document.
• **No introductory phrases (“This is…”, “The document is…”)—begin directly with the structured sentence.**  
**Examples of desired output style (model should mimic these patterns):**  
   • “Letter dated 15 March 2022 from Alice Wong (ABC Corp.) to John Tan (XYZ Ltd.) regarding Q1 budget approval; Ref: LT/2022/03-001.”  
   • “Invoice INV-2022-045 dated 02 June 2022 from Global Supplies Pte. Ltd. to Oceanic Trading Sdn. Bhd.; Amount Due: USD 12,500; Due Date: 30 June 2022.”  
   • “Affidavit sworn 22 April 2023 by Mark Chan (Advocate & Solicitor) in Support of Application for Injunction; Suit No. S-2023/045.”  
   • “Email dated 05 November 2022 from David Lee (PixelMedia) to Marketing Team; CC: HR Department; Subject: Q4 Campaign Launch Timeline.”  

2. KEY DATES: List all significant dates. If none, state "None".

3. ULTIMATE KEY DATE: Determine the single most representative date using these guidelines. If no rule applies or no date found, state "undated". If the dates are in other languages, make sure to translate them to English.  
    - General: Prioritize explicit dates (signing, effective, creation).  
    - Affidavits: Sign-off/"sworn on" date.  
    - Date Ranges: Latest date in range (e.g., "Statement from 1 Apr 2011 to 1 Jun 2011" → 1 Jun 2011).  
    - Future Dates: "undated" (unless it's an event/due date).  
    - Forms/Meeting Invitations (if no primary date): "Return by"/event date.  
    - Textual Dates: Only if in titles (Minutes/Agendas) or no other date.  
    - Minutes/Agendas: Meeting date from title.  
    - Drawings/Plans: Latest revision date.  
    - Agreements/Contracts: Execution date.  
    - Fax/Date Stamps: Only if no other relevant date.  
    - Handwritten Dates (on typed docs): Use this date. (If fully handwritten, use its date.)  
    - Tables/Spreadsheets/Financials (ordered, no creation date): Latest date (Excel: first sheet).  
    - Header/Footer/Print Dates: Last resort.

4. MAIN PARTIES: Primary entities (e.g., “Person (From), Organisation (From), Person (To), Organisation (To), Person (Between), Organisation (Between), Person (CC), Organisation (CC), Attendees” etc., depending on document type). If none, state "None".

5. CASE/REFERENCE NUMBERS: Important IDs (e.g., “Invoice No:”, “Case No:”, “Court Reference:”). If none, state "None".

6. DOCUMENT TYPE & REQUIRED FIELDS. 
For any detected document type, provide a natural, context-specific classification (e.g., “Court Judgment” rather than “Court Document: Judgment”; “Demand Letter” rather than “Letter: Demand Letter”; “Application Form” rather than “Form: Application Form”) and extract elements unique to that classification. 
Be flexible as long as the classification makes sense.
When specifying the document type, you do not need to use the exact guideline label verbatim; instead, use a natural, concise description that conveys the same meaning (e.g., “Civil Code” rather than “LEGISLATION/ACT”).
For each detected type, capture only that type's “Required Fields.” 
If a required field is not present in the text, state “None.” 
Do not infer fields that are not explicitly present. 

**Examples of required fields for each document type:**  
   • ADVICE (Advice Letter)  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • AFFIDAVIT/STATEMENT  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • AGENDA  
     - Required Fields: Date; Title; To (distribution list - Person/Organisation); Organisation (From).  

   • AGREEMENT/CONTRACT/DEED  
     - Required Fields: Date; Title; Person (Between); Organisation (Between).  

   • ANNUAL REPORT  
     - Required Fields: Date; Title; Person (From); Organisation (From).  

   • ARTICLE (Media Article/Release or scientific Article)  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • AUTHORITY (Search Warrants, etc.)  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • BOARD PAPERS  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • BROCHURE  
     - Required Fields: Date; Title; Person (From); Organisation (From).  

   • CERTIFICATE  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • COURT DOCUMENTS (e.g., Statements of Claim, Subpoenas, Judgements)  
     - Required Fields: Date; Title; Person (Between); Organisation (Between).  
     - **If classified as a Court Document, further identify its specific subtype—e.g., Judgment, Statement of Claim, Subpoena—based on headings and content, and note subtype-specific elements (e.g., Court Division, Judges, Parties, Outcome for a Judgment).**  

   • CURRICULUM VITAE/IDENTIFICATION  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • DIAGRAM/PLAN  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • DIARY ENTRY  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • DIVIDER/FILE COVER  
     - Required Fields: Date; Title.  

   • DOCUMENT (catch-all; use only if no other type applies)  
     - Required Fields: Date; Title.  

   • EMAIL  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To); Person (CC); Organisation (CC); Person (BCC); Organisation (BCC). (Also capture “Email Time” separately if available.)  

   • FACSIMILE  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To); Person (CC); Organisation (CC).  

   • FAX TRANSMISSION REPORT  
     - Required Fields: Date; Title (use “Untitled” if missing).  

   • FILE NOTE  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • FINANCIAL DOCUMENT  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • FORM  
     - Required Fields: Date; Title; Person (To); Person (From); Organisation (To); Organisation (From).  

   • HANDWRITTEN NOTE/NOTE  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • INVOICE/STATEMENT  
     - Required Fields: Date; Title; Person (From); Person (To); Organisation (To).  

   • LEGISLATION/ACT  
     - Required Fields: Date; Title; Organisation (From).  

   • LETTER  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To); Person (CC); Organisation (CC).  

   • LIST  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • MANUAL/GUIDELINES  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • MAP  
     - Required Fields: Date; Title; Person (From); Organisation (From).  

   • MEDIA ARTICLE/RELEASE  
     - Required Fields: Date; Title; Person (From); Organisation (From).  

   • MEMORANDUM  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • MINUTES OF MEETING  
     - Required Fields: Date; Title; Person (Attendees); Organisation (Attendees).  

   • NOTICE  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • PERMIT  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • PHOTOGRAPH  
     - Required Fields: Date; Title; Person (From); Organisation (From).  

   • PHYSICAL MEDIA  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • PRESENTATION  
     - Required Fields: Date; Title; Person (From); Organisation (From).  

   • RECEIPT  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • REMITTANCE  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • REPORT  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • RFI-RFO  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • SEARCH/COMPANY SEARCH  
     - Required Fields: Date; Title; Person (From); Organisation (From).  

   • SOCIAL MEDIA/MESSAGING  
     - Required Fields: Date; Title; Person (From); Organisation (From).  

   • SPECIFICATION  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • SUBMISSIONS  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • TABLE/SPREADSHEET  
     - Required Fields: Date; Title; Person (From); Organisation (From); Person (To); Organisation (To).  

   • TIMESHEET  
     - Required Fields: Date; Title; Person (From); Organisation (From).  

   • TRANSCRIPT  
     - Required Fields: Date; Title; Person (From); Organisation (From).  

   • WEB PAGE  
     - Required Fields: Title; Date; Organisation (From).  


IMPORTANT: Format your response exactly with the numbered format:  
“1. SUMMARY:”,  
“2. KEY DATES:”,  
“3. ULTIMATE KEY DATE:”,  
“4. MAIN PARTIES:”,  
“5. CASE/REFERENCE NUMBERS:”,  
“6. DOCUMENT TYPE & REQUIRED FIELDS:”.  
Do not add any other text before, between, or after these sections.

Document content:  
{document_text}
"""
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=types.GenerateContentConfig(**self.generation_config)
            )
            
            analysis_text = self._safe_extract_text(response)
            parsed_analysis = self._parse_analysis_string(analysis_text)
            token_usage = self._extract_token_usage(response)
            estimated_cost = self._calculate_estimated_cost(
                token_usage.get("input_tokens", 0),
                token_usage.get("output_tokens", 0)
            )

            return {
                "file_type": "text",
                "file_name": filename,
                "summary": parsed_analysis.get("SUMMARY", "Not available"),
                "date": parsed_analysis.get("ULTIMATE KEY DATE", "undated"),
                "extracted_info": {
                    "key_dates": parsed_analysis.get("KEY DATES", "Not available"),
                    "main_parties": parsed_analysis.get("MAIN PARTIES", "Not available"),
                    "case_reference_numbers": parsed_analysis.get("CASE/REFERENCE NUMBERS", "Not available"),
                    "full_analysis": analysis_text
                },
                "token_usage": token_usage,
                "estimated_cost": estimated_cost,
                "processed_at": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "file_type": "text",
                "file_name": filename,
                "summary": "Error during processing",
                "date": "undated",
                "extracted_info": {"full_analysis": f"Error: {str(e)}"},
                "token_usage": self._extract_token_usage(None),
                "estimated_cost": 0.0,
                "error": str(e),
                "status": "error",
                "processed_at": datetime.now().isoformat()
            } 