"""
Traditional Few-Shot Learning: Email Classification

SCENARIO:
Classify customer support emails into categories (Billing, Technical, Refund, General)
to route them to appropriate teams automatically.

TRADITIONAL APPROACH:
- Manual prompt engineering with static examples
- Hand-crafted few-shot examples in system prompt
- No learning or optimization from training data
- Examples must be manually curated and maintained
"""

from openai import AzureOpenAI
from pydantic import BaseModel
import os


client = AzureOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_MODEL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)


class EmailClassification(BaseModel):
    category: str
    confidence: int
    reasoning: str


few_shot_prompt = """You are a customer support email classifier.
Classify each email into exactly one of these categories:
- Billing
- Technical
- Refund
- General

Return JSON matching the schema: {"category": string, "confidence": int, "reasoning": string}
Confidence is an integer from 0 to 10.

Examples:
Email: "My card was charged twice for the same invoice and I need the duplicate charge removed."
Label: {"category": "Billing", "confidence": 9, "reasoning": "Duplicate charge and invoice issue are billing-related."}

Email: "The app crashes on startup after the latest update, and reinstalling did not help."
Label: {"category": "Technical", "confidence": 9, "reasoning": "App crash and update issues are technical problems."}

Email: "I want a refund because the service did not meet expectations. Please cancel my plan."
Label: {"category": "Refund", "confidence": 8, "reasoning": "Customer explicitly requests a refund and cancellation."}

Email: "Can you tell me your business hours and how to reset my password?"
Label: {"category": "General", "confidence": 6, "reasoning": "Mixed inquiry but no billing, refund, or technical error described."}
"""


user_email = (
    "Hi team, my monthly subscription increased without notice. "
    "Please explain the new charge and adjust it back."
)

completion = client.chat.completions.parse(
    model=os.getenv("AZURE_OPENAI_MODEL"),
    messages=[
        {"role": "system", "content": few_shot_prompt},
        {"role": "user", "content": f"Email: \"{user_email}\""},
    ],
    response_format=EmailClassification,
)

result = completion.choices[0].message.parsed

print("=" * 50)
print(f"Category: {result.category}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
