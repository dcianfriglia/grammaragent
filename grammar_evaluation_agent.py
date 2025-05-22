"""
Enhanced Grammar Evaluation Agent

A comprehensive text evaluation system that analyzes text across four dimensions:
correctness, readability, formatting, and style. Combines traditional NLP tools 
with LLM-based analysis and supports customization through external reference files.

Author: Grammar Evaluation Project
License: MIT
"""

import language_tool_python
import spacy
import json
import re
import os
import csv
import yaml
import base64
import httpx
from dotenv import load_dotenv
from openai import OpenAI


class GrammarEvaluationAgent:
    """
    A comprehensive text evaluation agent that combines traditional NLP tools with
    LLM capabilities to analyze text quality across multiple dimensions.
    
    The agent evaluates text based on:
    - Correctness: Grammar, spelling, and syntax
    - Readability: Clarity, flow, and ease of understanding
    - Formatting: Consistent presentation and structure
    - Style: Tone, voice, and stylistic guidelines adherence
    """

    def __init__(self, api_key, styleguide_path=None, glossary_path=None, 
                 style_checks_path=None, api_base=None, proxy_config=None):
        """
        Initialize the Grammar Evaluation Agent.
        
        Args:
            api_key (str): OpenAI API key
            styleguide_path (str, optional): Path to styleguide YAML file
            glossary_path (str, optional): Path to glossary CSV file
            style_checks_path (str, optional): Path to style checks YAML file
            api_base (str, optional): Base URL for the OpenAI API
            proxy_config (dict, optional): Proxy configuration
        """
        # Configure OpenAI client with proxy support if needed
        openai_args = {"api_key": api_key}
        if api_base:
            openai_args["base_url"] = api_base
        
        if proxy_config and proxy_config.get('url'):
            openai_args["http_client"] = self._create_http_client(proxy_config)
            
        self.client = OpenAI(**openai_args)
        
        # Initialize NLP tools
        self.language_tool = language_tool_python.LanguageTool('en-US')
        self.nlp = spacy.load('en_core_web_sm')
        
        # Load external reference files
        self.styleguide = self._load_yaml_file(styleguide_path) if styleguide_path else self._load_default_styleguide()
        self.glossary = self._load_csv_file(glossary_path) if glossary_path else self._load_default_glossary()
        self.style_checks = self._load_yaml_file(style_checks_path) if style_checks_path else self._load_default_style_checks()
        
    def _create_http_client(self, proxy_config):
        """Create an HTTP client with proxy configuration."""
        transport = httpx.HTTPTransport(proxy=proxy_config['url'])
        
        if proxy_config.get('auth'):
            auth = proxy_config['auth']
            auth_header = base64.b64encode(f'{auth[0]}:{auth[1]}'.encode()).decode()
            transport = httpx.HTTPTransport(
                proxy=proxy_config['url'],
                headers={"Proxy-Authorization": f"Basic {auth_header}"}
            )
        
        return httpx.Client(transport=transport)
            
    def _load_yaml_file(self, file_path):
        """Load YAML file or return empty dict if file not found."""
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Warning: Could not load YAML file {file_path}: {e}")
            return {}
            
    def _load_csv_file(self, file_path):
        """Load CSV file into dictionary."""
        result = {}
        try:
            with open(file_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if "term" in row and "preferred" in row:
                        result[row["term"].lower()] = row["preferred"]
            return result
        except FileNotFoundError as e:
            print(f"Warning: Could not load CSV file {file_path}: {e}")
            return {}

    def _load_default_styleguide(self):
        """Load default AP style guide rules."""
        return {
            "general_rules": {
                "abbreviations": "Use periods in most two-letter abbreviations: U.S., U.N., U.K., B.A., B.C.",
                "numbers": "Spell out numbers below 10; use figures for 10 and above.",
                "dates": "Use AP date format: Month Day, Year (e.g., January 1, 2023)",
                "titles": "Capitalize formal titles when used directly before a name.",
                "oxford_comma": "Do not use the Oxford comma in a simple series."
            },
            "punctuation": {
                "quotes": "Periods and commas go inside quotation marks.",
                "dashes": "Use an em dash (—) with no spaces to denote an abrupt change.",
                "semicolons": "Use semicolons to separate elements of a series when the items contain commas."
            },
            "capitalization": {
                "headlines": "Capitalize the first word and all nouns, pronouns, adjectives, verbs, adverbs, and subordinating conjunctions.",
                "titles": "Capitalize titles when they are used directly before names.",
                "seasons": "Do not capitalize seasons unless part of a formal name."
            }
        }
        
    def _load_default_glossary(self):
        """Load default glossary terms."""
        return {
            "website": "website",
            "web site": "website",
            "internet": "internet",
            "e-mail": "email",
            "on-line": "online",
            "data base": "database",
            "cyber security": "cybersecurity",
            "health care": "healthcare"
        }
        
    def _load_default_style_checks(self):
        """Load default style checks with examples."""
        return {
            "active_voice": {
                "description": "Prefer active voice over passive voice.",
                "regex_patterns": [
                    r"\bwas\s+\w+ed\b",
                    r"\bwere\s+\w+ed\b",
                    r"\bis\s+being\s+\w+ed\b",
                    r"\bare\s+being\s+\w+ed\b"
                ]
            },
            "conciseness": {
                "description": "Avoid wordiness and redundant phrases.",
                "phrases_to_avoid": [
                    "in order to",
                    "due to the fact that",
                    "for the purpose of",
                    "in the event that",
                    "at this point in time"
                ]
            },
            "jargon": {
                "description": "Avoid unnecessary jargon and explain technical terms.",
                "jargon_terms": [
                    "leverage",
                    "synergy",
                    "paradigm shift",
                    "circle back",
                    "ideate",
                    "core competencies"
                ]
            }
        }
        
    def check_with_language_tool(self, text):
        """Check text using LanguageTool for grammatical errors."""
        matches = self.language_tool.check(text)
        errors = []
        
        for match in matches:
            errors.append({
                "message": match.message,
                "context": match.context,
                "offset": match.offset,
                "length": len(match.matchedText) if hasattr(match, 'matchedText') else 0,
                "replacements": match.replacements[:3] if match.replacements else [],
                "rule_id": match.ruleId,
                "category": match.category
            })
            
        return errors
    
    def check_with_spacy(self, text):
        """Check text using SpaCy for syntax and entity analysis."""
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        analysis = {
            "sentence_count": len(sentences),
            "very_long_sentences": [str(sent) for sent in sentences if len(sent.text.split()) > 30],
            "unclear_references": [],
            "sentence_beginnings": {}
        }
        
        # Check for unclear pronoun references
        for sent in sentences:
            pronouns = [token for token in sent if token.pos_ == "PRON"]
            for pronoun in pronouns:
                if pronoun.i > 0 and pronoun.nbor(-1).pos_ not in ["NOUN", "PROPN"]:
                    analysis["unclear_references"].append({
                        "pronoun": str(pronoun),
                        "context": str(sent)
                    })
        
        # Analyze sentence beginnings (for variety)
        for sent in sentences:
            for token in sent:
                if token.is_alpha and not token.is_punct:
                    first_word = token.text.lower()
                    analysis["sentence_beginnings"][first_word] = analysis["sentence_beginnings"].get(first_word, 0) + 1
                    break
                    
        return analysis
    
    def check_readability(self, text):
        """Calculate basic readability metrics."""
        # Basic word count
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        
        # Basic sentence count
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Average words per sentence
        avg_words_per_sentence = word_count / max(1, sentence_count)
        
        # Count syllables (approximate)
        def count_syllables(word):
            word = word.lower()
            if len(word) <= 3:
                return 1
            count = 0
            vowels = "aeiouy"
            if word[0] in vowels:
                count += 1
            for i in range(1, len(word)):
                if word[i] in vowels and word[i-1] not in vowels:
                    count += 1
            if word.endswith("e"):
                count -= 1
            if count == 0:
                count = 1
            return count
        
        syllable_count = sum(count_syllables(word) for word in words)
        
        # Approximate Flesch Reading Ease
        if sentence_count > 0 and word_count > 0:
            flesch = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
        else:
            flesch = 0
            
        # Count complex words (words with 3+ syllables)
        complex_words = len([word for word in words if count_syllables(word) >= 3])
        
        return {
            "flesch_reading_ease": round(flesch, 2),
            "syllables_per_word": round(syllable_count / max(1, word_count), 2),
            "avg_words_per_sentence": round(avg_words_per_sentence, 2),
            "complex_words": complex_words,
            "word_count": word_count,
            "sentence_count": sentence_count
        }
    
    def check_formatting(self, text):
        """Analyze text formatting consistency."""
        formatting = {
            "issues": [],
            "consistency": {
                "bullet_points": {"styles": [], "consistent": True},
                "quotes": {"styles": [], "consistent": True},
                "spacing": {"double_spaces": 0, "consistent": True},
                "dashes": {"styles": [], "consistent": True}
            }
        }
        
        # Check for bullet point consistency
        bullet_patterns = re.findall(r'^\s*([\*\-•◦‣▪▫➤➢☞✓])\s', text, re.MULTILINE)
        if bullet_patterns:
            formatting["consistency"]["bullet_points"]["styles"] = list(set(bullet_patterns))
            if len(set(bullet_patterns)) > 1:
                formatting["consistency"]["bullet_points"]["consistent"] = False
                formatting["issues"].append({
                    "issue": "Inconsistent bullet point styles",
                    "details": f"Found multiple bullet point styles: {', '.join(set(bullet_patterns))}"
                })
        
        # Check for quote consistency
        straight_quotes = len(re.findall(r'["\']', text))
        curly_quotes = len(re.findall(r'[""'']', text))
        if straight_quotes > 0 and curly_quotes > 0:
            formatting["consistency"]["quotes"]["styles"] = ["straight", "curly"]
            formatting["consistency"]["quotes"]["consistent"] = False
            formatting["issues"].append({
                "issue": "Mixed quote styles",
                "details": "Found both straight (\", ') and curly ("", '') quotes"
            })
        
        # Check for double spaces
        double_spaces = len(re.findall(r'[.!?]\s{2,}[A-Z]', text))
        formatting["consistency"]["spacing"]["double_spaces"] = double_spaces
        if double_spaces > 0:
            formatting["consistency"]["spacing"]["consistent"] = False
            formatting["issues"].append({
                "issue": "Inconsistent spacing after periods",
                "details": f"Found {double_spaces} instances of double spaces after periods"
            })
            
        return formatting
    
    def check_glossary_compliance(self, text):
        """Check if the text uses preferred terms from the glossary."""
        issues = []
        text_lower = text.lower()
        
        for term, preferred in self.glossary.items():
            if term != preferred and term in text_lower:
                pattern = re.compile(r'(.{0,20}' + re.escape(term) + r'.{0,20})', re.IGNORECASE)
                matches = pattern.findall(text)
                
                for match in matches:
                    issues.append({
                        "term": term,
                        "preferred": preferred,
                        "context": match.strip()
                    })
        
        return issues
    
    def check_style_compliance(self, text):
        """Check text against style checks."""
        results = {
            "active_voice": [],
            "conciseness": [],
            "jargon": [],
            "ap_style": []
        }
        
        # Check active voice
        if "active_voice" in self.style_checks:
            for pattern in self.style_checks["active_voice"].get("regex_patterns", []):
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    results["active_voice"].append({
                        "text": match.group(0),
                        "context": text[max(0, match.start()-20):min(len(text), match.end()+20)]
                    })
        
        # Check conciseness
        if "conciseness" in self.style_checks:
            for phrase in self.style_checks["conciseness"].get("phrases_to_avoid", []):
                if phrase.lower() in text.lower():
                    pattern = re.compile(r'(.{0,20}' + re.escape(phrase) + r'.{0,20})', re.IGNORECASE)
                    matches = pattern.findall(text)
                    
                    for match in matches:
                        results["conciseness"].append({
                            "phrase": phrase,
                            "context": match.strip()
                        })
        
        # Check jargon
        if "jargon" in self.style_checks:
            for term in self.style_checks["jargon"].get("jargon_terms", []):
                if re.search(r'\b' + re.escape(term) + r'\b', text.lower()):
                    pattern = re.compile(r'(.{0,20}\b' + re.escape(term) + r'\b.{0,20})', re.IGNORECASE)
                    matches = pattern.findall(text)
                    
                    for match in matches:
                        results["jargon"].append({
                            "term": term,
                            "context": match.strip()
                        })
        
        # Check AP Style compliance
        if re.search(r'\b[0-9]\b', text):
            number_matches = re.finditer(r'\b([0-9])\b', text)
            for match in number_matches:
                results["ap_style"].append({
                    "rule": "AP Style: Spell out numbers below 10",
                    "text": match.group(0),
                    "context": text[max(0, match.start()-20):min(len(text), match.end()+20)]
                })
        
        return results
    
    def check_with_llm(self, text):
        """Use the LLM as an expert evaluator with our detailed agent prompt."""
        # Load the agent prompt from file or use default
        try:
            with open("agent_prompt.md", "r") as f:
                system_prompt = f.read()
        except FileNotFoundError:
            print("Warning: agent_prompt.md not found. Using simplified prompt.")
            system_prompt = """
            You are a professional editor and grammar expert. Analyze the text for grammar, spelling, punctuation, clarity, and style issues.
            
            Evaluate the text in these four distinct categories:
            1. CORRECTNESS (grammar, spelling, and syntax)
            2. READABILITY (clarity, flow, and ease of understanding)
            3. FORMATTING (consistent presentation and document structure)
            4. STYLE (tone, voice, and adherence to AP Style guidelines)
            
            Provide your response as a structured JSON with an array of issues and category scores.
            """
        
        # Prepare tool outputs to share with the agent
        tool_results = {
            "language_tool_results": self.check_with_language_tool(text),
            "spacy_results": self.check_with_spacy(text),
            "readability_results": self.check_readability(text),
            "formatting_results": self.check_formatting(text),
            "glossary_issues": self.check_glossary_compliance(text),
            "style_issues": self.check_style_compliance(text)
        }
        
        # Convert to a simpler format for the LLM
        tool_results_simplified = self._simplify_tool_results(tool_results)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please evaluate the following text:\n\n{text}\n\nHere are the results from the analysis tools:\n{json.dumps(tool_results_simplified, indent=2)}"}
                ]
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"LLM evaluation error: {e}")
            return {
                "error": str(e), 
                "issues": [],
                "category_scores": {
                    "correctness": 0,
                    "readability": 0,
                    "formatting": 0,
                    "style": 0
                }
            }

    def _simplify_tool_results(self, results):
        """Simplify tool results to a format suitable for the LLM."""
        simplified = {}
        
        # Simplify LanguageTool results
        simplified["grammar_issues"] = [
            {
                "text": issue.get("context", ""),
                "message": issue.get("message", ""),
                "category": issue.get("category", "")
            } for issue in results["language_tool_results"][:20]
        ]
        
        # Include readability metrics
        simplified["readability"] = {
            key: value for key, value in results["readability_results"].items()
            if key in ["flesch_reading_ease", "avg_words_per_sentence", "complex_words"]
        }
        
        # Include formatting issues
        simplified["formatting_issues"] = results["formatting_results"].get("issues", [])
        
        # Include glossary and style issues
        simplified["glossary_issues"] = results["glossary_issues"][:20]
        simplified["style_issues"] = {
            category: issues[:10] for category, issues in results["style_issues"].items()
        }
        
        return simplified

    def agent_evaluate_text(self, text):
        """
        Use the LLM as the primary agent to orchestrate text evaluation.
        
        Args:
            text (str): Text to evaluate
            
        Returns:
            dict: Comprehensive evaluation results
        """
        # Run all tools to gather data
        tool_results = {
            "language_tool_results": self.check_with_language_tool(text),
            "spacy_results": self.check_with_spacy(text),
            "readability_results": self.check_readability(text),
            "formatting_results": self.check_formatting(text),
            "glossary_issues": self.check_glossary_compliance(text),
            "style_issues": self.check_style_compliance(text)
        }
        
        # Simplify tool results for the agent
        simplified_results = self._simplify_tool_results(tool_results)
        
        # Load agent prompt
        try:
            with open("agent_prompt.md", "r") as f:
                system_prompt = f.read()
        except FileNotFoundError:
            print("Warning: agent_prompt.md not found. Using simplified prompt.")
            system_prompt = """
            You are a text evaluation assistant that analyzes text for quality across four dimensions:
            correctness, readability, formatting, and style. Use the provided tool results to
            evaluate the text and provide a comprehensive report with scores and detailed error analysis.
            """
        
        # Ask the agent to evaluate based on the tools and text
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please evaluate this text using the tool results:\n\nTEXT TO EVALUATE:\n{text}\n\nTOOL RESULTS:\n{json.dumps(simplified_results, indent=2)}"}
                ]
            )
            
            # Parse and return the agent's evaluation
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Error with agent evaluation: {e}")
            # Fallback to basic evaluation
            return self._create_fallback_evaluation(tool_results)

    def _create_fallback_evaluation(self, tool_results):
        """Create a basic evaluation when LLM is unavailable."""
        errors = []
        category_errors = {"correctness": [], "readability": [], "formatting": [], "style": []}
        
        # Process LanguageTool results
        for error in tool_results["language_tool_results"]:
            severity = 3 if error.get("category") in ["TYPOS", "GRAMMAR"] else 4
            error_item = {
                "error_text": error.get("context", ""),
                "error_type": error.get("category", "Grammar"),
                "category": "correctness",
                "severity": severity,
                "reasoning": error.get("message", ""),
                "suggestion": error.get("replacements", []),
                "source": "LanguageTool"
            }
            errors.append(error_item)
            category_errors["correctness"].append(error_item)
        
        # Calculate category scores
        category_scores = {}
        weights = {"correctness": 0.4, "readability": 0.25, "formatting": 0.15, "style": 0.2}
        
        for category in ["correctness", "readability", "formatting", "style"]:
            score = 100
            for error in category_errors[category]:
                penalty = {1: 30, 2: 15, 3: 10, 4: 5}.get(error["severity"], 5)
                score -= penalty
            category_scores[category] = max(0, score)
        
        overall_score = sum(category_scores[cat] * weights[cat] for cat in weights.keys())
        
        # Check compliance
        has_severe_errors = any(error["severity"] <= 3 for error in category_errors["correctness"])
        compliance_flag = 0 if has_severe_errors else 1
        
        return {
            "errors": errors,
            "scores": {
                "overall": round(overall_score, 1),
                "correctness": round(category_scores["correctness"], 1),
                "readability": round(category_scores["readability"], 1),
                "formatting": round(category_scores["formatting"], 1),
                "style": round(category_scores["style"], 1)
            },
            "compliance_flag": compliance_flag,
            "total_errors": len(errors),
            "error_counts_by_severity": {
                f"severity_{i}": sum(1 for error in errors if error["severity"] == i)
                for i in range(1, 5)
            },
            "error_counts_by_category": {
                category: len(category_errors[category])
                for category in category_errors
            },
            "readability_metrics": tool_results["readability_results"]
        }


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Initialize agent
    agent = GrammarEvaluationAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_API_BASE"),
        styleguide_path="styleguide.yaml",
        glossary_path="glossary.csv",
        style_checks_path="style_checks.yaml"
    )
    
    # Sample text
    sample_text = """
    Our company have been operating in the cyber security space for over 10 years. 
    Their main product is software that help businesses manage there operations.
    """
    
    # Evaluate text
    result = agent.agent_evaluate_text(sample_text)
    print(json.dumps(result, indent=2))
