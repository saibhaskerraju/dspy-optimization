import dspy
import os
from typing import List
import mlflow
mlflow.dspy.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Create a unique name for your experiment.
mlflow.set_experiment("DspyMultiStage")

# logging.basicConfig(level=logging.DEBUG)
os.environ['DSPY_DEBUG'] = '1'

key = os.getenv("AZURE_OPENAI_API_KEY")
llm = dspy.LM(model=f"azure/{os.getenv('AZURE_OPENAI_MODEL')}", api_key=key, api_base=os.getenv(
    "AZURE_OPENAI_ENDPOINT"), api_version=os.getenv("AZURE_OPENAI_API_VERSION"))

dspy.settings.configure(lm=llm, trace=[])


class MarketAnalysis(dspy.Signature):
    topic: str = dspy.InputField(desc="Business topic to analyze")
    market_data: str = dspy.InputField(desc="Available market information")
    market_size: str = dspy.OutputField(desc="Market size and growth potential")
    competitors: List[str] = dspy.OutputField(desc="Key competitors and differentiation")
    revenue_opportunity: str = dspy.OutputField(desc="Revenue opportunity estimate")

class CustomerAnalysis(dspy.Signature):
    topic: str = dspy.InputField(desc="Business topic to analyze")
    customer_feedback: str = dspy.InputField(desc="Customer feedback data")
    pain_points: List[str] = dspy.OutputField(desc="Key customer pain points")
    customer_needs: List[str] = dspy.OutputField(desc="Customer needs and preferences")
    willingness_to_pay: str = dspy.OutputField(desc="Willingness to pay indicators")

class StrategicRecommendations(dspy.Signature):
    topic: str = dspy.InputField(desc="Business topic to analyze")
    market_analysis: MarketAnalysis = dspy.InputField(desc="Market analysis results")
    customer_analysis: CustomerAnalysis = dspy.InputField(desc="Customer analysis results")
    recommendations: List[str] = dspy.OutputField(desc="Top 3 strategic recommendations")
    go_to_market: str = dspy.OutputField(desc="Go-to-market strategy")
    success_metrics: List[str] = dspy.OutputField(desc="Key success metrics")

class BusinessAnalysisPipeline(dspy.Module):
    def __init__(self):
        self.analyze_market = dspy.ChainOfThought(MarketAnalysis)
        self.analyze_customers = dspy.ChainOfThought(CustomerAnalysis)
        self.generate_strategy = dspy.ChainOfThought(StrategicRecommendations)
    
    def forward(self, topic: str, market_data: str, customer_feedback: str) -> dspy.Prediction:
        market = self.analyze_market(topic=topic, market_data=market_data)
        customers = self.analyze_customers(topic=topic, customer_feedback=customer_feedback)
        strategy = self.generate_strategy(
            topic=topic,
            market_analysis=market,
            customer_analysis=customers
        )
        
        return dspy.Prediction(
            market_analysis=market,
            customer_analysis=customers,
            recommendations=strategy
        )

# Usage
pipeline = BusinessAnalysisPipeline()

result = pipeline(
    topic="AI-powered customer support tool",
    market_data="Market growing at 25% annually, current size $2B, competitors: Zendesk, Intercom",
    customer_feedback="Users want faster response times, hate repetitive queries, willing to pay premium"
)

# Clean structured access
print(f"Market Size: {result.market_analysis.market_size}")
print(f"Pain Points: {result.customer_analysis.pain_points}")
print(f"Recommendations: {result.recommendations.recommendations}")