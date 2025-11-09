from openai import AzureOpenAI
from pydantic import BaseModel
import os

client = AzureOpenAI(
    azure_deployment=os.getenv('AZURE_OPENAI_MODEL'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
)

class SentimentClassifier(BaseModel):
    text: str
    sentiment: int

completion  = client.chat.completions.parse(
    model=os.getenv('AZURE_OPENAI_MODEL'),
    messages=[
        {
            "role": "system",
            "content": "You are a sentiment classifier. the sentiment, the higher the more positive is an integer between 0 and 10.",
        },
        {"role": "user", "content": "I am feeling pretty happy!"},
    ],
    response_format=SentimentClassifier,
)

sentiment_classifier = completion.choices[0].message.parsed

print(f"The sentiment is: {sentiment_classifier.sentiment}")