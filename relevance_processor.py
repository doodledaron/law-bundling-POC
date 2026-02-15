import os
import time
import logging
import json
import re
from typing import Dict, List, Optional
from datetime import datetime
from config import Config

logger = logging.getLogger(__name__)

# Import Google AI conditionally since it requires Python 3.9+
try:
    from google import genai
    from google.genai import types
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    # Running on Python 3.7 - Google AI not available
    GOOGLE_AI_AVAILABLE = False
    genai = None
    types = None

class RelevanceProcessor:
    """
    Processor for generating token-efficient legal relevance extractions.
    Reuses the same Gemini client pattern as TextBasedProcessor.
    """
    # Costing constants for Gemini 2.0 Flash (same as TextBasedProcessor)
    INPUT_COST_PER_M_TOKENS = 0.10    # USD ($0.10 per 1M input tokens)
    OUTPUT_COST_PER_M_TOKENS = 0.40   # USD ($0.40 per 1M output tokens)
    
    # Token efficiency limits
    MAX_EXTRACTED_CHARS = 16000  # ~4000 tokens
    MAX_EVIDENCE_QUOTES = 6
    MAX_QUOTE_WORDS = 25
    
    def __init__(self):
        if GOOGLE_AI_AVAILABLE:
            self.client = genai.Client(
                api_key=Config.GEMINI_API_KEY,
                vertexai=False
            )
            self.model = Config.MODEL_NAME
            self.generation_config = Config.GENERATION_CONFIG
        else:
            # Google AI not available on Python 3.7
            self.client = None
            self.model = None
            self.generation_config = None
            logger.warning("Google AI (Gemini) not available on Python 3.7. Relevance processing will be disabled.")
    
    def _call_with_retry(self, contents, max_retries=4, base_delay=2.0, max_delay=60.0):
        """
        Call Gemini generate_content with exponential backoff retry.
        Reuses the same retry pattern as TextBasedProcessor.
        """
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(**self.generation_config)
                )
                return response
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                status_code = getattr(e, 'code', None) or getattr(e, 'status_code', None)
                
                # Determine if this error is retryable
                is_retryable = False
                if isinstance(e, (ConnectionError, TimeoutError, OSError)):
                    is_retryable = True
                elif status_code in (429, 500, 503):
                    is_retryable = True
                elif any(keyword in error_str for keyword in [
                    '429', 'rate limit', 'resource exhausted', 'quota',
                    '500', 'internal', '503', 'unavailable', 'overloaded',
                    'deadline exceeded', 'timeout', 'connection'
                ]):
                    is_retryable = True
                
                if not is_retryable or attempt >= max_retries:
                    logger.error(f"Gemini API call failed (attempt {attempt + 1}/{max_retries + 1}, non-retryable): {e}")
                    raise
                
                # Calculate delay with exponential backoff + jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                import random
                delay = delay + random.uniform(0, delay * 0.1)
                
                logger.warning(
                    f"Gemini API call failed (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)
        
        raise last_exception
    
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
            pass
        
        return usage_info
    
    def _calculate_estimated_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the estimated cost for processing based on tokens."""
        cost_input_text = (input_tokens / 1000000.0) * self.INPUT_COST_PER_M_TOKENS
        cost_output_text = (output_tokens / 1000000.0) * self.OUTPUT_COST_PER_M_TOKENS
        total_cost = cost_input_text + cost_output_text
        return round(total_cost, 6)
    
    def _smart_slice_text(self, text: str, max_chars: int) -> str:
        """
        Smart text slicing that preserves legally operative sections.
        Instead of blind truncation, extracts:
        - First portion (front-matter: headnote, overview, held)
        - Last portion (conclusions, orders)
        - Keyword windows around legal-operational terms
        
        Args:
            text: Full document text
            max_chars: Maximum characters to return
            
        Returns:
            Strategically sliced text preserving key legal content
        """
        if len(text) <= max_chars:
            return text
        
        # Allocate budget: 40% front, 30% back, 30% keyword windows
        front_budget = int(max_chars * 0.40)
        back_budget = int(max_chars * 0.30)
        keyword_budget = max_chars - front_budget - back_budget
        
        # Extract front and back portions
        front_text = text[:front_budget]
        back_text = text[-back_budget:]
        
        # Find keyword windows around legal-operational terms
        legal_keywords = [
            r'\bheld\b', r'\bholding\b', r'\bconcluded\b', r'\bdismissed\b', r'\ballowed\b',
            r'\bshall\b', r'\bmust\b', r'\bliable\b', r'\bliability\b', r'\bbreach\b',
            r'\benforceable\b', r'\bvest\b', r'\bvested\b', r'\bvesting\b',
            r'\blimitation\b', r'\bnot begin to run\b', r'\bpostpone\b',
            r'\brequired to\b', r'\bobligated to\b', r'\bestablishes\b',
            r'\bprovides that\b', r'\bthe court held\b', r'\bit was held\b'
        ]
        
        window_size = 800  # Increased to 800 chars for complete sentences/clauses
        
        # Combine all keywords into one pattern
        pattern = '|'.join(legal_keywords)
        
        # Find all matches and create windows
        import re
        raw_windows = []
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = max(0, match.start() - window_size)
            end = min(len(text), match.end() + window_size)
            raw_windows.append((start, end))
        
        # Merge overlapping windows to avoid duplication
        if raw_windows:
            # Sort by start position
            raw_windows.sort()
            merged_windows = [raw_windows[0]]
            
            for current_start, current_end in raw_windows[1:]:
                last_start, last_end = merged_windows[-1]
                
                # If overlapping or adjacent, merge
                if current_start <= last_end + 50:  # Allow small gaps
                    merged_windows[-1] = (last_start, max(last_end, current_end))
                else:
                    merged_windows.append((current_start, current_end))
            
            # Extract text from merged windows, respecting budget
            keyword_windows = []
            total_window_chars = 0
            
            for start, end in merged_windows:
                window_text = text[start:end]
                window_len = len(window_text)
                
                # Check if adding this window exceeds budget
                if total_window_chars + window_len <= keyword_budget:
                    keyword_windows.append(window_text)
                    total_window_chars += window_len
                else:
                    # Add partial window if budget allows
                    remaining_budget = keyword_budget - total_window_chars
                    if remaining_budget > 200:  # Only add if meaningful
                        keyword_windows.append(window_text[:remaining_budget])
                    break
            
            logger.info(f"Smart slice: extracted {len(keyword_windows)} keyword windows ({total_window_chars} chars)")
        else:
            keyword_windows = []
        
        # Combine portions
        if keyword_windows:
            middle_text = '\n\n[...keyword excerpt...]\n\n'.join(keyword_windows)
            sliced_text = f"{front_text}\n\n[...middle sections omitted...]\n\n{middle_text}\n\n[...]\n\n{back_text}"
        else:
            # No keyword windows found, just use front + back
            sliced_text = f"{front_text}\n\n[...middle sections omitted...]\n\n{back_text}"
        
        # Final safety check
        if len(sliced_text) > max_chars:
            # If still too long, just take front + back
            adjusted_front = int(max_chars * 0.6)
            adjusted_back = max_chars - adjusted_front - 50  # Reserve for separator
            sliced_text = f"{text[:adjusted_front]}\n\n[...]\n\n{text[-adjusted_back:]}"
        
        return sliced_text
    
    def generate_relevance(self, combined_text: str, filename: str) -> Dict:
        """
        Generate token-efficient legal relevance extraction from document text.
        
        Args:
            combined_text: Full extracted text from the document
            filename: Original filename
            
        Returns:
            Dictionary with relevance extraction results
        """
        # Check if Google AI is available
        if not GOOGLE_AI_AVAILABLE or self.client is None:
            return {
                "job_id": None,
                "filename": filename,
                "document_type": "unknown",
                "relevances": [{
                    "relevance_text": "Google AI not available on Python 3.7",
                    "pinpoints": [],
                    "evidence_quotes": []
                }],
                "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "estimated_cost": 0.0,
                "processing_completed_at": datetime.now().isoformat(),
                "status": "disabled"
            }
        
        try:
            if not combined_text.strip():
                return {
                    "filename": filename,
                    "document_type": "unknown",
                    "relevances": [{
                        "relevance_text": "No text content could be extracted from the document",
                        "pinpoints": [],
                        "evidence_quotes": []
                    }],
                    "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "estimated_cost": 0.0,
                    "processing_completed_at": datetime.now().isoformat(),
                    "status": "error"
                }
            
            # Apply token budget - use smart slicing instead of blind truncation
            if len(combined_text) > self.MAX_EXTRACTED_CHARS:
                logger.warning(f"Document text exceeds {self.MAX_EXTRACTED_CHARS} chars, using smart slice for token efficiency")
                combined_text = self._smart_slice_text(combined_text, self.MAX_EXTRACTED_CHARS)
                text_truncated = True
            else:
                text_truncated = False
            
            # Build the relevance extraction prompt
            prompt = self._build_relevance_prompt(combined_text)
            
            # Call Gemini with retry
            logger.info(f"Generating relevance extraction for {filename}")
            response = self._call_with_retry(contents=[prompt])
            
            # Extract response text
            response_text = self._safe_extract_text(response)
            
            # Parse the structured response
            relevance_output = self._parse_relevance_response(response_text)
            
            # Extract token usage and calculate cost
            token_usage = self._extract_token_usage(response)
            estimated_cost = self._calculate_estimated_cost(
                token_usage.get("input_tokens", 0),
                token_usage.get("output_tokens", 0)
            )
            
            # No quality validation - just extract and return
            
            # Build final result
            result = {
                "filename": filename,
                "document_type": relevance_output.get("document_type", "unknown"),
                "relevances": relevance_output.get("relevances", []),
                "token_usage": token_usage,
                "estimated_cost": estimated_cost,
                "processing_completed_at": datetime.now().isoformat(),
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating relevance: {str(e)}")
            return {
                "filename": filename,
                "document_type": "unknown",
                "relevances": [{
                    "relevance_text": f"Error during relevance generation: {str(e)}",
                    "pinpoints": [],
                    "evidence_quotes": []
                }],
                "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "estimated_cost": 0.0,
                "processing_completed_at": datetime.now().isoformat(),
                "error": str(e),
                "status": "error"
            }
    
    def _build_relevance_prompt(self, document_text: str) -> str:
        """Build the relevance extraction prompt."""
        
        prompt = f"""You are a legal document analyzer specializing in extracting legally operative content and producing concise, traceable relevance summaries in professional index style.

Your task is to identify ONLY the legally relevant portions of this document and generate structured relevance entries grounded strictly in the document text.

DOCUMENT TYPE CLASSIFICATION:

First, classify this document as one of:
- case (court judgment, decision, or case law)
- statute (legislation, act, regulation)
- notice (guideline, circular, notice)
- other (any other legal document type)

PRINCIPLE-BASED EXTRACTION GUIDANCE:

Prioritize text that contains legally operative language, including:

- Judicial conclusions (e.g., "the Court held", "it was concluded", "the appeal is dismissed")
- Statements of legal tests or standards
- Findings of liability (e.g., "liable", "breach", "dishonest assistance")
- Statutory mechanisms (e.g., "shall", "must", "vest in", "enforceable")
- Limitation triggers (e.g., "shall not begin to run until")
- Regulatory obligations (e.g., "A bank shall", "is required to")

Avoid extracting:

- Background narrative
- Chronological fact summaries
- Lists of parties
- Administrative history unless legally significant
- Financial or evidentiary detail unless directly tied to the legal rule

STRUCTURAL EXTRACTION RULES BY DOCUMENT TYPE:

For CASES:
- Prioritize Overview, Headnote, Digest, Introduction, Issues, and Held/Decision sections
- Extract the ratio decidendi (legal principle or holding)
- Focus on doctrinal reasoning and legal conclusions
- If no headnote exists, identify paragraphs expressing the Court's determination

For STATUTES:
- Extract operative provisions only
- Focus on binding language ("shall", "must", "is liable")
- Include adjacent subsections only if necessary for clarity
- Do not reproduce the entire Act

For NOTICES/GUIDELINES:
- Extract mandatory obligations
- Identify the regulated entity and required conduct
- Focus on compliance mechanisms rather than explanatory background

STYLE GUIDANCE — INDEX RELEVANCE EXAMPLES:

Match the tone and structure below.

Example — Statutory Mechanism:
Relevance:
Part VIIA provides that upon issuance of a transfer certificate, the transferor's business vests in the transferee and existing rights and proceedings become enforceable by or against the transferee. The provision establishes statutory substitution without further assurance.

Example — Limitation:
Relevance:
Section 32 postpones the commencement of the limitation period where an action is based on fraud or deliberate concealment, such that time does not begin to run until discovery of the fraud. The section establishes a statutory postponement mechanism.

Example — Judicial Holding:
Relevance:
The Court held that a party who dishonestly assists in a breach of fiduciary duty may be directly liable notwithstanding that it was not the primary wrongdoer. The judgment clarifies the scope of secondary liability.

STYLE PRINCIPLES:

- Begin with the legal rule or mechanism.
- State the legal consequence.
- Avoid factual storytelling.
- Avoid argumentative or persuasive tone.
- Be neutral and doctrinal.
- Length: 2–4 lines per relevance entry.

MULTIPLE RELEVANCE ENTRIES:

If the document contains multiple distinct legal principles or mechanisms that are independently significant, produce multiple relevance entries as separate items in the output array.

Each relevance entry must:
- Address a distinct legal rule or mechanism.
- Be supported by its own evidence quotes.
- Avoid duplication of reasoning.

RELEVANCE TEXT REQUIREMENTS:

Each relevance entry must:

1. Clearly state the legal rule, mechanism, or holding.
2. State the legal consequence or effect.
3. Explain why it is legally significant.
4. Be grounded strictly in the document text.

PINPOINTS REQUIREMENTS:

- Reference actual sections/paragraphs/pages from the document
- Never invent pinpoints
- Format: {{"reference": "s 32(1)", "page": 5, "type": "section"}} or {{"reference": "Para [42]", "page": 12, "type": "paragraph"}}
- Maximum 10 pinpoints per relevance entry

EVIDENCE QUOTES REQUIREMENTS:

- Maximum {self.MAX_EVIDENCE_QUOTES} quotes per relevance entry.
- Each quote maximum {self.MAX_QUOTE_WORDS} words.
- Quotes must be verbatim.
- Quotes must directly support a statement in the relevance text.
- Each quote must reference a pinpoint using pinpoint_ref.
- Format: {{"text": "quote text here", "page": 5, "pinpoint_ref": "s 32(1)", "support_for": "vesting mechanism"}}

OUTPUT FORMAT (JSON):

{{
  "document_type": "case|statute|notice|other",
  "relevances": [
    {{
      "relevance_text": "2-4 line doctrinal relevance paragraph",
      "pinpoints": [
        {{"reference": "s 32(1)", "page": 5, "type": "section"}}
      ],
      "evidence_quotes": [
        {{"text": "quote up to {self.MAX_QUOTE_WORDS} words", "page": 5, "pinpoint_ref": "s 32(1)", "support_for": "brief label"}}
      ]
    }}
  ]
}}

CRITICAL RULES:

- Every legal claim must be grounded in the document.
- Do not infer beyond what the document states.
- Never fabricate quotes, page numbers, statutory references, or holdings.
- If only one legally significant principle exists, return a single relevance entry.
- If none can be confidently identified, return a single relevance entry explaining that no clear operative legal principle could be located.

DOCUMENT TEXT:
{document_text}

Return ONLY the JSON output. No commentary.
"""
        
        return prompt
    
    def _parse_relevance_response(self, response_text: str) -> Dict:
        """
        Parse Gemini's relevance response into structured format.
        Robust parser that handles malformed JSON, trailing commas, markdown blocks.
        """
        try:
            # Strategy 1: Try to parse as clean JSON first
            try:
                parsed = json.loads(response_text)
                return parsed
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    parsed = json.loads(json_str)
                    return parsed
                except json.JSONDecodeError:
                    pass
            
            # Strategy 3: Find largest JSON object heuristically (first { to last })
            first_brace = response_text.find('{')
            last_brace = response_text.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_candidate = response_text[first_brace:last_brace + 1]
                
                # Try to clean common JSON issues
                # Remove trailing commas before closing braces/brackets
                json_candidate = re.sub(r',(\s*[}\]])', r'\1', json_candidate)
                
                try:
                    parsed = json.loads(json_candidate)
                    return parsed
                except json.JSONDecodeError:
                    pass
            
            # Strategy 4: Try to extract JSON array if model returned just array
            first_bracket = response_text.find('[')
            last_bracket = response_text.rfind(']')
            
            if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                array_candidate = response_text[first_bracket:last_bracket + 1]
                array_candidate = re.sub(r',(\s*\])', r'\1', array_candidate)
                
                try:
                    relevances_array = json.loads(array_candidate)
                    if isinstance(relevances_array, list):
                        return {
                            "document_type": "unknown",
                            "relevances": relevances_array
                        }
                except json.JSONDecodeError:
                    pass
            
            # Fallback: Text-based parsing
            logger.warning("Could not parse JSON response, attempting text-based parsing")
            
            result = {
                "document_type": "unknown",
                "relevances": []
            }
            
            # Extract document type
            doc_type_match = re.search(r'document_type["\s:]+([a-z]+)', response_text, re.IGNORECASE)
            if doc_type_match:
                result["document_type"] = doc_type_match.group(1).lower()
            
            # Try to extract relevances array
            relevances_match = re.search(r'"relevances"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
            if relevances_match:
                relevances_text = relevances_match.group(1)
                # Try to parse individual relevance objects with nested structures
                # More permissive regex that handles nested braces
                relevance_objects = []
                depth = 0
                current_obj = ""
                in_relevances = False
                
                for char in relevances_text:
                    if char == '{':
                        if depth == 0:
                            current_obj = "{"
                        else:
                            current_obj += char
                        depth += 1
                        in_relevances = True
                    elif char == '}':
                        depth -= 1
                        current_obj += char
                        if depth == 0 and in_relevances:
                            relevance_objects.append(current_obj)
                            current_obj = ""
                            in_relevances = False
                    elif in_relevances:
                        current_obj += char
                
                for rel_obj in relevance_objects:
                    try:
                        # Clean trailing commas
                        rel_obj = re.sub(r',(\s*[}\]])', r'\1', rel_obj)
                        relevance_data = json.loads(rel_obj)
                        result["relevances"].append(relevance_data)
                    except:
                        pass
            
            # Final fallback: extract any relevance_text found
            if not result["relevances"]:
                relevance_match = re.search(r'relevance_text["\s:]+["\'](.+?)["\']', response_text, re.DOTALL)
                if relevance_match:
                    result["relevances"].append({
                        "relevance_text": relevance_match.group(1).strip(),
                        "pinpoints": [],
                        "evidence_quotes": []
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing relevance response: {str(e)}")
            return {
                "document_type": "unknown",
                "relevances": [{
                    "relevance_text": "Error parsing AI response",
                    "pinpoints": [],
                    "evidence_quotes": []
                }]
            }
    
