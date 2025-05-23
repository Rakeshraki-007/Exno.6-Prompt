# Exno.6-Prompt-Engg
# Date:23.05.2025
# Register no.212222090023
# Aim: Development of Python Code Compatible with Multiple AI Tools
AI tools
**Deepseek**
**ChatGpt**
**Gemini**


# Algorithm: Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights.
**DEEPSEEK**
AI-Powered API Integration and Analysis Tool
Below is a Python implementation that integrates with multiple AI tools to automate API interactions, compare outputs, and generate insights. This solution uses:

OpenAI's GPT for analysis

Anthropic's Claude (via API) as an alternative AI

LangChain for orchestration

Pandas for data comparison

python
import os
import requests
import pandas as pd
from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import json
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class APIAnalyzer:
    def __init__(self):
        # Initialize AI models
        self.openai = ChatOpenAI(model_name="gpt-4", temperature=0.5)
        self.claude = ChatAnthropic(model="claude-2", temperature=0.5)
        
        # Configure API timeout
        self.timeout = 30
        
        # Initialize results storage
        self.results = defaultdict(dict)
        
    def call_api(self, url: str, method: str = "GET", params: dict = None, 
                 headers: dict = None, data: dict = None) -> Dict[str, Any]:
        """
        Generic API caller with error handling
        """
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed: {e}")
            return {"error": str(e)}
    
    def analyze_with_ai(self, api_data: Dict[str, Any], model: str = "openai") -> Dict[str, Any]:
        """
        Analyze API response with specified AI model
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are an expert API analyst. Analyze this API response and provide insights."),
                HumanMessage(content=f"API Response: {json.dumps(api_data, indent=2)}\n\nProvide key insights and actionable recommendations.")
            ])
            
            if model.lower() == "openai":
                response = self.openai(prompt.format_messages())
            elif model.lower() == "claude":
                response = self.claude(prompt.format_messages())
            else:
                raise ValueError("Unsupported AI model specified")
                
            return {
                "model": model,
                "analysis": response.content,
                "success": True
            }
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {
                "model": model,
                "analysis": str(e),
                "success": False
            }
    
    def compare_responses(self, responses: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple API responses using pandas DataFrame
        """
        comparison_data = []
        
        for response in responses:
            if "error" in response:
                comparison_data.append({
                    "status": "error",
                    "content": response["error"]
                })
            else:
                # Flatten the response for comparison
                flat_response = self.flatten_dict(response)
                flat_response["status"] = "success"
                comparison_data.append(flat_response)
        
        return pd.DataFrame(comparison_data)
    
    def flatten_dict(self, data: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Flatten nested dictionary for easier comparison
        """
        items = {}
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self.flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = str(v)  # Convert all values to string for comparison
        return items
    
    def generate_comparison_report(self, df: pd.DataFrame) -> str:
        """
        Generate a human-readable comparison report
        """
        differences = []
        
        # Compare columns across responses
        for col in df.columns:
            unique_values = df[col].unique()
            if len(unique_values) > 1 and col != "status":
                differences.append({
                    "field": col,
                    "values": unique_values.tolist(),
                    "count": len(unique_values)
                })
        
        # Prepare report
        report = "API Response Comparison Report\n"
        report += "=" * 40 + "\n\n"
        report += f"Total responses compared: {len(df)}\n"
        report += f"Total differences found: {len(differences)}\n\n"
        
        if differences:
            report += "Key Differences:\n"
            for diff in sorted(differences, key=lambda x: x["count"], reverse=True)[:10]:
                report += f"- {diff['field']}: {diff['values']}\n"
        
        return report
    
    def run_analysis_pipeline(self, api_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Complete pipeline: call APIs, analyze with AI, compare results
        """
        final_results = {
            "api_responses": [],
            "ai_analyses": [],
            "comparison": None,
            "report": None
        }
        
        # Step 1: Call all APIs
        for config in api_configs:
            logger.info(f"Calling API: {config.get('name', 'Unnamed API')}")
            response = self.call_api(
                url=config["url"],
                method=config.get("method", "GET"),
                params=config.get("params"),
                headers=config.get("headers"),
                data=config.get("data")
            )
            final_results["api_responses"].append({
                "config": config,
                "response": response
            })
        
        # Step 2: Analyze each response with both AI models
        for api_result in final_results["api_responses"]:
            if "error" not in api_result["response"]:
                for model in ["openai", "claude"]:
                    logger.info(f"Analyzing with {model}")
                    analysis = self.analyze_with_ai(api_result["response"], model=model)
                    final_results["ai_analyses"].append({
                        "api_config": api_result["config"],
                        "model": model,
                        "analysis": analysis
                    })
        
        # Step 3: Compare responses
        comparison_df = self.compare_responses([r["response"] for r in final_results["api_responses"]])
        final_results["comparison"] = comparison_df.to_dict(orient="records")
        
        # Step 4: Generate report
        final_results["report"] = self.generate_comparison_report(comparison_df)
        
        return final_results


# Example Usage
if __name__ == "__main__":
    analyzer = APIAnalyzer()
    
    # Example API configurations
    api_configs = [
        {
            "name": "JSONPlaceholder Posts",
            "url": "https://jsonplaceholder.typicode.com/posts/1",
            "method": "GET"
        },
        {
            "name": "JSONPlaceholder Comments",
            "url": "https://jsonplaceholder.typicode.com/comments",
            "method": "GET",
            "params": {"postId": 1}
        },
        {
            "name": "Example Error API",
            "url": "https://jsonplaceholder.typicode.com/nonexistent",
            "method": "GET"
        }
    ]
    
    # Run the complete analysis
    results = analyzer.run_analysis_pipeline(api_configs)
    
    # Print the comparison report
    print("\nFinal Comparison Report:")
    print(results["report"])
    
    # Print AI analyses
    print("\nAI Analyses:")
    for analysis in results["ai_analyses"]:
        if analysis["analysis"]["success"]:
            print(f"\nAnalysis from {analysis['model']} for {analysis['api_config']['name']}:")
            print(analysis["analysis"]["analysis"])
Key Features of This Implementation:
Multi-API Integration:

Handles GET/POST requests with parameters, headers, and data

Robust error handling for API failures

Multi-AI Analysis:

Supports both OpenAI GPT-4 and Anthropic Claude

Standardized analysis prompts for consistent outputs

Comparison Engine:

Flattens nested API responses for easier comparison

Uses pandas DataFrames for structural analysis

Generates human-readable difference reports

Complete Pipeline:

Orchestrates API calls → AI analysis → comparison → reporting

Tracks all results in a structured format

Extensibility:

Easy to add more AI models (Bard, Llama, etc.)

Can be extended with more sophisticated comparison logic

Requirements:
Install required packages:

bash
pip install requests pandas langchain openai anthropic python-dotenv
Create a .env file with your API keys:

OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
This implementation provides a solid foundation for automating API interactions and analysis. You can extend it with additional features like:

Caching API responses

Adding more AI models

Implementing more sophisticated comparison algorithms

Generating visual comparison charts

Adding authentication handlers for different API types
**CHATGPT**
Here's a Python implementation that integrates with multiple AI tools via APIs, compares their outputs, and generates actionable insights. The example assumes you’re working with two AI language models (e.g., OpenAI GPT and Anthropic Claude), using them to generate summaries for a text input, and then comparing their outputs to extract key differences and common themes.

Features:
Interacts with multiple APIs.

Compares AI-generated outputs.

Generates a concise summary of similarities and differences.

Note: You'll need valid API keys and proper access to the respective AI tools.

✅ Step-by-step Python Code
python
Copy
Edit
import openai
import requests
import difflib
from typing import Dict, Tuple

# === Configuration ===
openai.api_key = 'your_openai_api_key'
anthropic_api_key = 'your_anthropic_api_key'

# === Input ===
input_text = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn. 
This field of computer science has grown significantly and is influencing various industries including healthcare, finance, and transportation.
"""

# === Helper Functions ===

def get_openai_response(prompt: str) -> str:
    """Query OpenAI GPT API."""
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[{"role": "user", "content": f"Summarize this text:\n{prompt}"}],
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

def get_anthropic_response(prompt: str) -> str:
    """Query Anthropic Claude API (hypothetical example using REST)."""
    headers = {
        "x-api-key": anthropic_api_key,
        "content-type": "application/json"
    }
    json_payload = {
        "model": "claude-3",
        "prompt": f"\n\nHuman: Summarize the following text:\n{prompt}\n\nAssistant:",
        "max_tokens_to_sample": 150
    }
    response = requests.post("https://api.anthropic.com/v1/complete", headers=headers, json=json_payload)
    return response.json()['completion'].strip()

def compare_outputs(output1: str, output2: str) -> Tuple[str, str, str]:
    """Compare outputs and extract commonalities and differences."""
    # Use difflib to compare
    differ = difflib.Differ()
    diff = list(differ.compare(output1.split(), output2.split()))

    common = ' '.join([word[2:] for word in diff if word.startswith('  ')])
    only_in_first = ' '.join([word[2:] for word in diff if word.startswith('- ')])
    only_in_second = ' '.join([word[2:] for word in diff if word.startswith('+ ')])

    return common, only_in_first, only_in_second

def generate_insights(common: str, only_in_first: str, only_in_second: str) -> Dict:
    """Generate human-readable insights."""
    return {
        "common_summary": common,
        "openai_unique_points": only_in_first,
        "anthropic_unique_points": only_in_second,
        "insight": (
            "The AI tools agree on these main points: "
            f"{common}. However, OpenAI adds: '{only_in_first}' while Claude adds: '{only_in_second}'."
        )
    }

# === Execution ===
if __name__ == "__main__":
    openai_output = get_openai_response(input_text)
    print("OpenAI Output:\n", openai_output)

    anthropic_output = get_anthropic_response(input_text)
    print("\nAnthropic Output:\n", anthropic_output)

    common, openai_diff, anthropic_diff = compare_outputs(openai_output, anthropic_output)

    insights = generate_insights(common, openai_diff, anthropic_diff)
    print("\nActionable Insights:")
    for key, value in insights.items():
        print(f"{key}:\n{value}\n")
🔧 Requirements
Install dependencies with:

bash
Copy
Edit
pip install openai requests
**GEMINI**
This is a fascinating and ambitious request! To truly "integrate with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights," we'll need to define the scope and the "AI tools" we're referring to.

For a practical and demonstrable example, I'll use the following interpretation:

API Interaction: Using Python's requests library.
Comparing Outputs: Using Python's json module for structured comparison and potentially a diffing library like deepdiff for more detailed analysis.
Generating Actionable Insights: This is where the "AI" comes in. I'll leverage a Large Language Model (LLM) (e.g., Google Gemini, OpenAI GPT) through their respective APIs to interpret the comparison results and suggest insights. Since I don't have direct access to external APIs for AI models in this environment, I'll simulate the LLM interaction with a placeholder function. In a real-world scenario, you would integrate with an actual LLM API (e.g., google-generativeai library for Gemini).
Scenario: We'll compare the output of two hypothetical "product catalog" APIs. Let's say one is a "current production" API and the other is a "new development" API. We want to identify differences in product data and get an AI-generated insight into what those differences might imply.

Libraries you'd typically need to install:

Bash

pip install requests deepdiff google-generativeai # (if using actual Gemini API)
Python Code Implementation:

Python

import requests
import json
from deepdiff import DeepDiff # For detailed JSON comparison
import os
# from google.generativeai import GenerativeModel # Uncomment and configure for actual Gemini API

# --- Configuration ---
# Replace with your actual API endpoints
API_ENDPOINT_1 = "https://api.example.com/products/v1" # Placeholder for production API
API_ENDPOINT_2 = "https://api.example.com/products/v2" # Placeholder for development API

# Replace with your actual API keys if required by the APIs
API_KEY_1 = os.getenv("API_KEY_1", "YOUR_API_KEY_1")
API_KEY_2 = os.getenv("API_KEY_2", "YOUR_API_KEY_2")

# Configure your AI model (e.g., Google Gemini)
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
# model = GenerativeModel('gemini-pro') # For actual Gemini integration

# --- Helper Functions ---

def fetch_api_data(url, api_key=None, params=None, headers=None):
    """
    Fetches data from a given API endpoint.
    """
    if headers is None:
        headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}' # Common for many APIs

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None

def compare_json_outputs(json1, json2):
    """
    Compares two JSON objects and returns the differences.
    Uses deepdiff for a comprehensive comparison.
    """
    if json1 is None or json2 is None:
        return {"error": "One or both JSON inputs are None for comparison."}

    differences = DeepDiff(json1, json2, ignore_order=True, report_repetition=True)
    return differences.to_json() # Return as JSON string for consistent handling

def generate_insight_with_ai(comparison_report):
    """
    Simulates interaction with an AI model to generate insights from a comparison report.
    In a real scenario, this would call an LLM API.
    """
    if not comparison_report:
        return "No significant differences found, or an error occurred during comparison. No insights generated."

    # In a real scenario, you would send `comparison_report` to your LLM API.
    # Example for Gemini (requires google-generativeai library and API key):
    # try:
    #     prompt = f"Analyze the following JSON comparison report and provide actionable insights:\n\n{comparison_report}\n\nWhat are the key differences, what might they imply, and what actions should be taken?"
    #     response = model.generate_content(prompt)
    #     return response.text
    # except Exception as e:
    #     return f"Error communicating with AI model: {e}. Raw comparison report:\n{comparison_report}"

    # For demonstration, we'll use a simple rule-based "AI" for insights:
    comparison_dict = json.loads(comparison_report)
    insights = []

    if "values_changed" in comparison_dict:
        insights.append(f"Detected changes in values for keys: {list(comparison_dict['values_changed'].keys())}. This could indicate data updates or discrepancies.")
        for key, details in comparison_dict['values_changed'].items():
            insights.append(f"  - Key: {key}, Old Value: {details['old_value']}, New Value: {details['new_value']}")
        insights.append("Action: Investigate these value changes. Are they expected? Do they align with release notes or data migration plans?")

    if "dictionary_item_added" in comparison_dict:
        insights.append(f"New dictionary items added: {comparison_dict['dictionary_item_added']}. This suggests new fields or products have been introduced.")
        insights.append("Action: Verify if these new items are intended and ensure backward compatibility if necessary. Update documentation.")

    if "dictionary_item_removed" in comparison_dict:
        insights.append(f"Dictionary items removed: {comparison_dict['dictionary_item_removed']}. This indicates data or fields have been deprecated or removed.")
        insights.append("Action: Assess the impact of these removals. Are they planned? Are there any downstream dependencies affected?")

    if not insights:
        insights.append("No significant differences identified that require specific actions based on this report.")

    return "\n".join(insights)

# --- Main Automation Logic ---

def automate_api_analysis(api1_url, api2_url, api1_key=None, api2_key=None):
    """
    Automates API interaction, comparison, and insight generation.
    """
    print(f"--- Fetching Data from API 1: {api1_url} ---")
    data1 = fetch_api_data(api1_url, api_key=api1_key)

    print(f"\n--- Fetching Data from API 2: {api2_url} ---")
    data2 = fetch_api_data(api2_url, api_key=api2_key)

    if data1 is None or data2 is None:
        print("Aborting analysis due to API data fetching errors.")
        return

    print("\n--- Comparing API Outputs ---")
    comparison_report = compare_json_outputs(data1, data2)
    print("Comparison Report (Raw JSON):\n", comparison_report)

    print("\n--- Generating Actionable Insights with AI ---")
    insights = generate_insight_with_ai(comparison_report)
    print("Actionable Insights:\n", insights)

# --- Example Usage (simulated API responses) ---
if __name__ == "__main__":
    # Simulate API responses for testing
    simulated_data_v1 = {
        "products": [
            {"id": "P001", "name": "Laptop Pro", "price": 1200, "category": "Electronics"},
            {"id": "P002", "name": "Mechanical Keyboard", "price": 150, "category": "Accessories"},
            {"id": "P003", "name": "Wireless Mouse", "price": 50, "category": "Accessories", "stock": 100}
        ],
        "metadata": {"version": "1.0", "timestamp": "2025-05-23T10:00:00Z"}
    }

    # Scenario 1: Minor changes
    simulated_data_v2_scenario1 = {
        "products": [
            {"id": "P001", "name": "Laptop Pro", "price": 1250, "category": "Electronics"}, # Price change
            {"id": "P002", "name": "Mechanical Keyboard", "price": 150, "category": "Accessories"},
            {"id": "P003", "name": "Wireless Mouse", "price": 50, "category": "Accessories", "stock": 95} # Stock change
        ],
        "metadata": {"version": "1.1", "timestamp": "2025-05-23T10:30:00Z"} # Timestamp change
    }

    # Scenario 2: Structural changes (added/removed items)
    simulated_data_v2_scenario2 = {
        "products": [
            {"id": "P001", "name": "Laptop Pro", "price": 1200, "category": "Electronics"},
            {"id": "P002", "name": "Mechanical Keyboard", "price": 150, "category": "Gaming Accessories"}, # Category change
            {"id": "P004", "name": "External Monitor", "price": 300, "category": "Electronics"} # New product
        ],
        "metadata": {"version": "2.0", "timestamp": "2025-05-23T11:00:00Z"}
    }

    # Override fetch_api_data to use simulated data for demonstration
    original_fetch_api_data = fetch_api_data

    def mock_fetch_api_data(url, api_key=None, params=None, headers=None):
        if url == API_ENDPOINT_1:
            print(f"(MOCK) Returning simulated data for {url}")
            return simulated_data_v1
        elif url == API_ENDPOINT_2 + "_scenario1":
            print(f"(MOCK) Returning simulated data for {url} (Scenario 1)")
            return simulated_data_v2_scenario1
        elif url == API_ENDPOINT_2 + "_scenario2":
            print(f"(MOCK) Returning simulated data for {url} (Scenario 2)")
            return simulated_data_v2_scenario2
        else:
            print(f"(MOCK) Unknown URL: {url}")
            return None

    fetch_api_data = mock_fetch_api_data

    print("\n\n############################################")
    print("### Running Scenario 1: Minor Changes ###")
    print("############################################")
    automate_api_analysis(API_ENDPOINT_1, API_ENDPOINT_2 + "_scenario1", API_KEY_1, API_KEY_2)

    print("\n\n############################################")
    print("### Running Scenario 2: Structural Changes ###")
    print("############################################")
    automate_api_analysis(API_ENDPOINT_1, API_ENDPOINT_2 + "_scenario2", API_KEY_1, API_KEY_2)

    # Restore original function to avoid interfering with other potential tests
    fetch_api_data = original_fetch_api_data
Explanation and How it Integrates AI:

API Interaction (Workspace_api_data):

Uses the requests library to make HTTP GET requests to the specified API endpoints.
Handles API keys for authentication (common for many APIs).
Includes basic error handling for network issues or HTTP error codes.
Parses the JSON response.
Comparing Outputs (compare_json_outputs):

Leverages the deepdiff library. This is a powerful tool for comparing complex Python objects (like dictionaries and lists, which JSON translates to) and provides detailed reports of additions, removals, and value changes, including nested structures.
ignore_order=True is crucial for JSON comparison as dictionary key order often doesn't matter, and list order might not always be significant depending on the API design.
report_repetition=True can be useful for identifying if items within lists have changed their order or count.
The output of DeepDiff is converted to JSON for consistent handling.
Generating Actionable Insights (generate_insight_with_ai):

The AI Integration Point: This is where you'd connect to a real AI service (like Google Gemini, OpenAI GPT, etc.).
Simulated AI (for demonstration): In the provided code, I've implemented a rule-based "AI" to demonstrate the concept without requiring actual API keys and external calls during execution. It parses the deepdiff report and generates human-readable insights and suggested actions based on predefined rules (e.g., if values_changed exists, suggest investigating data updates).
Real AI (commented out): The commented-out section shows how you would typically use google-generativeai to send the comparison_report to an LLM and get a more nuanced, context-aware analysis. The LLM can then:
Summarize the differences.
Interpret the meaning of the differences (e.g., "a price increase on 'Laptop Pro' might indicate a new model or market adjustment").
Suggest concrete actionable insights (e.g., "Verify this price change with the product team," "Update downstream systems reliant on this product data," "Review security implications of new fields").
Identify patterns that a simple deepdiff might not immediately highlight (e.g., a consistent shift in prices across many products).
Main Automation Logic (automate_api_analysis):

Orchestrates the entire process: fetches data from both APIs, compares them, and then passes the comparison report to the insight generation function.
How to make it truly robust and "AI-driven":

Actual LLM Integration: Replace the simulated generate_insight_with_ai with calls to a real LLM API. This is the core of the "AI" aspect for insight generation.
Dynamic API Discovery: For more advanced automation, you could use AI to read API documentation (e.g., OpenAPI/Swagger specs) to dynamically understand endpoints, parameters, and expected responses, rather than hardcoding URLs. Libraries like langchain or llama-index could help with this.
Anomaly Detection: Instead of just comparing, you could train an ML model (e.g., using scikit-learn or TensorFlow/PyTorch) to detect "anomalies" in API responses over time or across different environments. For example, a sudden, large increase in specific error codes or a deviation in the data structure that hasn't been explicitly defined.
Automated Remediation (Advanced): In highly mature systems, the actionable insights could trigger further automated actions, such as:
Opening a JIRA ticket with the identified differences.
Alerting a Slack channel.
Running automated tests on the new API version specifically for the changed areas.
Automatically updating documentation or configuration files (with human oversight).
Feedback Loop: Implement a mechanism for human feedback on the AI's insights. This feedback can then be used to fine-tune the LLM prompts or even retrain a custom AI model for better insight generation over time.
This comprehensive approach showcases how Python can be a central hub for integrating various tools (API clients, diffing libraries, and AI models) to create a powerful automation pipeline for API analysis and insight generation.
# Result: The corresponding Prompt is executed successfully
