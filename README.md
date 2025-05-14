# Exno.6 - Prompt Engineering
### Date: 
### Register No: 212222060191

## Aim:
Development of Python Code Compatible with Multiple AI Tools

---

## Algorithm:
1. **Write Python Code**: Develop a Python script that integrates with multiple AI tools via their APIs. Ensure the code is flexible enough to work with different platforms.
   
2. **Interact with APIs**: The script should send requests to various AI tool APIs (e.g., ChatGPT, Claude, Bard) with a test input prompt.
   
3. **Compare Outputs**: Once responses are received from each tool, compare the outputs for accuracy, quality, and relevance.

4. **Generate Insights**: Based on the comparison, generate actionable insights (e.g., which AI tool performed best for specific tasks).

5. **Output Result**: Provide the results of the comparison in a structured format, highlighting the strengths and weaknesses of each tool.

---

## Example Python Code (Pseudo):

```python
import openai
import requests
import json

# Function to interact with ChatGPT (OpenAI API)
def get_chatgpt_response(prompt):
    openai.api_key = "your_openai_api_key"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Function to interact with Cohere API
def get_cohere_response(prompt):
    api_url = "https://api.cohere.ai/v1/generate"
    headers = {"Authorization": "Bearer your_cohere_api_key"}
    data = {
        "model": "command-xlarge-2021-11-08",
        "prompt": prompt,
        "max_tokens": 100
    }
    response = requests.post(api_url, headers=headers, json=data)
    return response.json()["text"]

# Example usage
prompt = "What is the capital of France?"

# Get responses from multiple AI tools
chatgpt_output = get_chatgpt_response(prompt)
cohere_output = get_cohere_response(prompt)

# Compare and print outputs
print("ChatGPT Output:", chatgpt_output)
print("Cohere Output:", cohere_output)

# Here you can compare the outputs and generate actionable insights

```

## Output
The corresponding Prompt is executed successfully.
