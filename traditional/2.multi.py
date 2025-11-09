from openai import AzureOpenAI
from pydantic import BaseModel
import os

client = AzureOpenAI(
    azure_deployment=os.getenv('AZURE_OPENAI_MODEL'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
)

class BusinessAnalysis(BaseModel):
    market_analysis: str
    customer_analysis: str
    strategic_recommendations: str

completion = client.chat.completions.parse(
    model=os.getenv('AZURE_OPENAI_MODEL'),
    messages=[
        {
            "role": "system",
            "content": """You are a business analyst. Analyze the business opportunity and provide three key analyses:
1. Market Analysis: Size, growth potential, competitors, and revenue opportunity
2. Customer Analysis: Pain points, needs, and willingness to pay
3. Strategic Recommendations: Top recommendations with go-to-market strategy

Return the analysis in the specified structured format.""",
        },
        {
            "role": "user", 
            "content": f"""
Analyze this business opportunity:

TOPIC: AI-powered customer support tool

MARKET DATA:
- Market growing at 25% annually
- Current market size: $2B  
- Main competitors: Zendesk, Intercom, Freshdesk
- Regulatory environment: Moderate

CUSTOMER FEEDBACK:
- Users want faster response times
- Hate answering repetitive queries
- Willing to pay premium for quality support
- Want better integration with existing tools

Provide comprehensive analysis in the three required sections.
"""
        }
    ],
    response_format=BusinessAnalysis,
)

business_analysis = completion.choices[0].message.parsed

print("="*50)
print(f"Market Analysis: {business_analysis.market_analysis}")
print(f"Customer Analysis: {business_analysis.customer_analysis}")
print(f"Strategic Recommendations: {business_analysis.strategic_recommendations}")