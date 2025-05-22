"""
Example usage of the Grammar Evaluation Agent
"""

import os
import json
from dotenv import load_dotenv
from grammar_evaluation_agent import GrammarEvaluationAgent


def main():
    """Example usage of the GrammarEvaluationAgent."""
    # Load environment variables
    load_dotenv()
    
    # Initialize the agent
    agent = GrammarEvaluationAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_API_BASE"),
        styleguide_path="config/styleguide.yaml",
        glossary_path="config/glossary.csv",
        style_checks_path="config/style_checks.yaml"
    )
    
    # Sample text to evaluate
    sample_text = """
    Our company have been operating in the cyber security space for over 10 years. 
    We specialize in artificial intelligence solutions which help businesses protect there data 
    and systems from evolving threats. In order to achieve this goal, we leverage our core competencies 
    in machine-learning and deep-learning. The report was written by our research team and it's findings 
    were presented at the annual security confrence.
    
    Our CEO John smith said "we are proud of our achievements". Despite the competative market, 
    our revenue has grown by 20% last year and we plan to expand our operations to 5 new countrys.
    """
    
    # Evaluate the text
    print("Evaluating text...")
    evaluation_result = agent.agent_evaluate_text(sample_text)
    
    # Print the results
    print_evaluation_summary(evaluation_result)
    
    # Save results to JSON file
    with open("evaluation_results.json", "w") as f:
        json.dump(evaluation_result, f, indent=2)
    
    print("\nResults saved to evaluation_results.json")


def print_evaluation_summary(evaluation):
    """Print a formatted summary of the evaluation results."""
    print("\n" + "="*50)
    print("TEXT EVALUATION SUMMARY")
    print("="*50)
    
    # Print scores
    print("\nSCORES:")
    print(f"Overall Score: {evaluation['scores']['overall']}/100")
    print(f"Correctness: {evaluation['scores']['correctness']}/100")
    print(f"Readability: {evaluation['scores']['readability']}/100")
    print(f"Formatting: {evaluation['scores']['formatting']}/100")
    print(f"Style: {evaluation['scores']['style']}/100")
    
    # Print compliance status
    print("\nCOMPLIANCE STATUS:")
    if evaluation['compliance_flag'] == 1:
        print("✅ PUBLISHABLE: Text meets quality standards")
    else:
        print("❌ NOT PUBLISHABLE: Text requires corrections")
    
    # Print error counts
    print("\nERROR SUMMARY:")
    print(f"Total errors found: {evaluation['total_errors']}")
    
    if 'error_counts_by_category' in evaluation:
        print("\nErrors by category:")
        for category, count in evaluation['error_counts_by_category'].items():
            print(f"  - {category.capitalize()}: {count}")
    
    if 'error_counts_by_severity' in evaluation:
        print("\nErrors by severity:")
        severity_descriptions = {1: "Critical", 2: "Major", 3: "Medium", 4: "Minor"}
        for severity_key, count in evaluation['error_counts_by_severity'].items():
            severity_num = int(severity_key.split('_')[1])
            severity_desc = severity_descriptions.get(severity_num, "Unknown")
            print(f"  - {severity_desc} (Level {severity_num}): {count}")
    
    # Print readability metrics
    if 'readability_metrics' in evaluation:
        print("\nREADABILITY METRICS:")
        metrics = evaluation['readability_metrics']
        print(f"Flesch Reading Ease: {metrics.get('flesch_reading_ease', 'N/A')}")
        print(f"Average Words per Sentence: {metrics.get('avg_words_per_sentence', 'N/A')}")
        print(f"Complex Words: {metrics.get('complex_words', 'N/A')}")
    
    # Print top errors (up to 5 most severe)
    if 'errors' in evaluation and evaluation['errors']:
        print("\nTOP ISSUES TO ADDRESS:")
        sorted_errors = sorted(evaluation['errors'], key=lambda x: x.get('severity', 5))
        for i, error in enumerate(sorted_errors[:5], 1):
            severity_descriptions = {1: "Critical", 2: "Major", 3: "Medium", 4: "Minor"}
            severity_desc = severity_descriptions.get(error.get('severity', 4), "Minor")
            print(f"{i}. {severity_desc} {error.get('category', 'Unknown').capitalize()} issue: {error.get('error_type', 'Unknown')}")
            print(f"   Text: \"{error.get('error_text', '')}\"")
            print(f"   Reasoning: {error.get('reasoning', '')}")
            print(f"   Suggestion: {error.get('suggestion', '')}")
            print()


if __name__ == "__main__":
    main()