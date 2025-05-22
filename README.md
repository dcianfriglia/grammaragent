# Grammar Evaluation Agent

A comprehensive text evaluation system that analyzes text quality across four dimensions: correctness, readability, formatting, and style. The agent combines traditional NLP tools with LLM capabilities and supports customization through external reference files.

## Features

- **Multi-dimensional Analysis**: Evaluates correctness, readability, formatting, and style
- **LLM Integration**: Uses OpenAI's GPT models for advanced text analysis
- **Customizable**: Supports external style guides, glossaries, and style checks
- **Proxy Support**: Works with HTTP proxies for enterprise environments
- **Comprehensive Scoring**: Provides detailed scores and compliance flags
- **AP Style Guidelines**: Built-in support for Associated Press style rules

## Quick Start

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/grammar-evaluation-agent.git
cd grammar-evaluation-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the SpaCy English model:
```bash
python -m spacy download en_core_web_sm
```

### Configuration

1. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
USE_PROXY=false
PROXY_URL=your_proxy_url_here
```

2. (Optional) Customize the configuration files in the `config/` directory:
   - `styleguide.yaml` - Style guidelines and AP Style rules
   - `glossary.csv` - Preferred terminology
   - `style_checks.yaml` - Custom style patterns
   - `agent_prompt.md` - LLM evaluation prompt

### Basic Usage

```python
from grammar_evaluation_agent import GrammarEvaluationAgent
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the agent
agent = GrammarEvaluationAgent(
    api_key=os.getenv("OPENAI_API_KEY"),
    styleguide_path="config/styleguide.yaml",
    glossary_path="config/glossary.csv",
    style_checks_path="config/style_checks.yaml"
)

# Evaluate text
text = "Your text to be evaluated..."
result = agent.agent_evaluate_text(text)

print(f"Overall Score: {result['scores']['overall']}/100")
print(f"Compliance: {'Publishable' if result['compliance_flag'] == 1 else 'Not Publishable'}")
```

## Output Format

The agent returns a comprehensive JSON object containing:

```json
{
  "errors": [
    {
      "error_text": "problematic text",
      "error_type": "Grammar/Spelling/Style",
      "category": "correctness/readability/formatting/style",
      "severity": 1-4,
      "reasoning": "explanation of the error",
      "suggestion": "proposed correction",
      "source": "tool that identified the error"
    }
  ],
  "scores": {
    "overall": 78.5,
    "correctness": 70.0,
    "readability": 85.0,
    "formatting": 95.0,
    "style": 80.0
  },
  "compliance_flag": 0,
  "total_errors": 3,
  "error_counts_by_severity": {...},
  "error_counts_by_category": {...},
  "readability_metrics": {...}
}
```

## Scoring System

- **Overall Score (0-100)**: Weighted average of all categories
  - Correctness: 40% weight
  - Readability: 25% weight  
  - Formatting: 15% weight
  - Style: 20% weight

- **Error Severity Levels**:
  1. **Critical**: Fatal error preventing understanding
  2. **Major**: Creates ambiguity but doesn't prevent understanding
  3. **Medium**: Damages credibility but is understandable
  4. **Minor**: Slight ambiguity or style issue

- **Compliance Flag**:
  - `1` = Publishable (no critical/major/medium errors in correctness or readability)
  - `0` = Not publishable (has severe errors)

## Customization

### Style Guide (styleguide.yaml)

Define organization-specific style rules:

```yaml
general_rules:
  abbreviations: "Use periods in most two-letter abbreviations: U.S., U.N."
  numbers: "Spell out numbers below 10; use figures for 10 and above."
punctuation:
  quotes: "Periods and commas go inside quotation marks."
  dashes: "Use an em dash (—) with no spaces to denote an abrupt change."
```

### Glossary (glossary.csv)

List preferred terminology:

```csv
term,preferred,notes
website,website,"Always one word, lowercase"
cyber security,cybersecurity,"One word, no hyphen"
e-mail,email,"No hyphen"
```

### Style Checks (style_checks.yaml)

Add patterns for style improvement:

```yaml
active_voice:
  description: "Prefer active voice over passive voice."
  regex_patterns:
    - "\bwas\s+\w+ed\b"
    - "\bwere\s+\w+ed\b"
conciseness:
  description: "Avoid wordiness and redundant phrases."
  phrases_to_avoid:
    - "in order to"
    - "due to the fact that"
```

## Architecture

The agent uses a multi-tool approach:

1. **LanguageTool**: Grammar and spelling checking
2. **SpaCy**: Syntax analysis and NLP processing  
3. **Custom Readability**: Flesch reading ease and other metrics
4. **OpenAI LLM**: Advanced analysis and orchestration
5. **Custom Checkers**: Formatting, glossary, and style compliance

## API Documentation

### GrammarEvaluationAgent Class

#### `__init__(api_key, styleguide_path=None, glossary_path=None, style_checks_path=None, api_base=None, proxy_config=None)`

Initialize the agent with API credentials and optional configuration files.

#### `agent_evaluate_text(text)` 

Main evaluation method that returns comprehensive analysis results.

#### Individual Check Methods

- `check_with_language_tool(text)`: Grammar and spelling errors
- `check_with_spacy(text)`: Syntax and entity analysis  
- `check_readability(text)`: Readability metrics
- `check_formatting(text)`: Formatting consistency
- `check_glossary_compliance(text)`: Terminology compliance
- `check_style_compliance(text)`: Style guideline adherence

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Troubleshooting

### Common Issues

1. **SSL Certificate Errors**: If you encounter SSL issues, ensure your network allows HTTPS connections to the OpenAI API.

2. **SpaCy Model Not Found**: Run `python -m spacy download en_core_web_sm` to download the required language model.

3. **API Rate Limits**: The agent respects OpenAI's rate limits. For high-volume usage, consider implementing request throttling.

4. **Proxy Configuration**: For enterprise environments, configure the proxy settings in your `.env` file.

### Performance Tips

- Use the `agent_evaluate_text()` method for the most comprehensive analysis
- For batch processing, consider implementing rate limiting
- Large texts (>10,000 words) may take longer to process

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the example usage in `example_usage.py`
