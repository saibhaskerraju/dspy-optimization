"""
DSPy MIPRO v2: Multi-Stage Instruction and Proposal Optimization

SCENARIO:
Email sentiment analysis - classify customer emails as Positive, Negative, or Neutral
with confidence scores and detailed reasoning using advanced MIPRO v2 optimization.

MIPRO v2 APPROACH:
- Multi-stage optimization combining instruction and few-shot examples
- Automatic bootstrapping of high-quality examples
- Instruction optimization with proposal generation
- Advanced evaluation metrics for comprehensive optimization
- Comparison with base performance
"""

import os
import dspy
import mlflow
from dspy.evaluate import Evaluate


mlflow.dspy.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("dspyfewshotmipro")

key = os.getenv("AZURE_OPENAI_API_KEY")
llm = dspy.LM(
    model=f"azure/{os.getenv('AZURE_OPENAI_MODEL')}",
    api_key=key,
    api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

dspy.settings.configure(lm=llm, trace=[])


class SentimentSignature(dspy.Signature):
    """Classify sentiment with confidence and reasoning."""

    email = dspy.InputField(desc="Customer email text")
    sentiment = dspy.OutputField(desc="Positive, Negative, or Neutral")
    confidence = dspy.OutputField(desc="Integer from 0 to 100")
    reasoning = dspy.OutputField(desc="Short justification")


class SentimentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(SentimentSignature)

    def forward(self, email: str) -> dspy.Prediction:
        return self.classify(email=email)


def parse_confidence(value) -> int | None:
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    try:
        return int(text)
    except ValueError:
        return None


def evaluate_sentiment_analysis(example, pred, trace=None) -> float:
    match = example.sentiment.lower() == pred.sentiment.lower().strip().rstrip(".")
    confidence = parse_confidence(pred.confidence)
    confidence_ok = confidence is not None and 0 <= confidence <= 100
    score = 1.0 if match else 0.0
    if confidence_ok:
        score += 0.1
    if not match:
        print(f"❌ Mismatch: {example.sentiment} vs {pred.sentiment}")
    return score



trainset = [
    dspy.Example(
        email="I love the new dashboard. It saved us hours today!",
        sentiment="Positive",
        confidence="90",
        reasoning="Clear praise and satisfaction with the new feature",
    ).with_inputs("email"),
    dspy.Example(
        email="The app keeps crashing after the update. This is unacceptable.",
        sentiment="Negative",
        confidence="95",
        reasoning="Strong dissatisfaction due to crashes",
    ).with_inputs("email"),
    dspy.Example(
        email="Can you confirm our renewal date and pricing?",
        sentiment="Neutral",
        confidence="70",
        reasoning="Information request without positive or negative sentiment",
    ).with_inputs("email"),
    dspy.Example(
        email="Thanks for the fast support, we are back online.",
        sentiment="Positive",
        confidence="88",
        reasoning="Gratitude for support indicates positive sentiment",
    ).with_inputs("email"),
    dspy.Example(
        email="Your billing error caused a disruption and wasted our time.",
        sentiment="Negative",
        confidence="85",
        reasoning="Complaint about error and disruption",
    ).with_inputs("email"),
    dspy.Example(
        email="I am evaluating the product and need a demo next week.",
        sentiment="Neutral",
        confidence="65",
        reasoning="Evaluation request without sentiment cues",
    ).with_inputs("email"),
]

devset = [
    dspy.Example(
        email="The migration went smoothly. Appreciate the guidance.",
        sentiment="Positive",
        confidence="82",
        reasoning="Positive outcome and appreciation",
    ).with_inputs("email"),
    dspy.Example(
        email="We are frustrated with the slow response times lately.",
        sentiment="Negative",
        confidence="80",
        reasoning="Explicit frustration about performance",
    ).with_inputs("email"),
    dspy.Example(
        email="Please send the SOC2 report and security docs.",
        sentiment="Neutral",
        confidence="60",
        reasoning="Compliance request without sentiment",
    ).with_inputs("email"),
    dspy.Example(
        email="Amazing work by your team, keep it up!",
        sentiment="Positive",
        confidence="92",
        reasoning="Strong praise and positive tone",
    ).with_inputs("email"),
    dspy.Example(
        email="This outage cost us revenue. I am very disappointed.",
        sentiment="Negative",
        confidence="90",
        reasoning="Disappointment and impact statement",
    ).with_inputs("email"),
    dspy.Example(
        email="Can you update my contact email on the account?",
        sentiment="Neutral",
        confidence="65",
        reasoning="Administrative request without sentiment cues",
    ).with_inputs("email"),
    dspy.Example(
        email="Your support team was incredibly helpful today.",
        sentiment="Positive",
        confidence="85",
        reasoning="Compliment for support team",
    ).with_inputs("email"),
    dspy.Example(
        email="I appreciate the refund, but the process was a nightmare.",
        sentiment="Negative",
        confidence="70",
        reasoning="Overall negative experience despite the positive outcome",
    ).with_inputs("email"),
    dspy.Example(
        email="I'm not sure if this is what I need, it's a bit confusing.",
        sentiment="Negative",
        confidence="60",
        reasoning="Expression of confusion and uncertainty",
    ).with_inputs("email"),
    dspy.Example(
        email="Is there a discount for non-profits?",
        sentiment="Neutral",
        confidence="75",
        reasoning="Pure inquiry about pricing policy",
    ).with_inputs("email"),
]

testset = [
    dspy.Example(
        email="The setup was confusing and took hours.",
        sentiment="Negative",
        confidence="80",
        reasoning="Negative experience and frustration",
    ).with_inputs("email"),
    dspy.Example(
        email="Thank you for resolving my issue so quickly.",
        sentiment="Positive",
        confidence="88",
        reasoning="Gratitude for quick resolution",
    ).with_inputs("email"),
    dspy.Example(
        email="Please send a copy of the invoice.",
        sentiment="Neutral",
        confidence="60",
        reasoning="Simple request without sentiment",
    ).with_inputs("email"),
]


print("🚀 DSPy MIPRO v2 Demo: Email Sentiment Analysis")
print("=" * 55)

baseline_model = SentimentClassifier()

baseline_evaluator = Evaluate(
    devset=devset,
    metric=evaluate_sentiment_analysis,
    num_threads=2,
    display_progress=True,
)

baseline_score = baseline_evaluator(baseline_model)
print(f"\n📊 Baseline Score: {baseline_score}")

mipro_optimizer = dspy.MIPROv2(
    metric=evaluate_sentiment_analysis,
    auto="light",
    num_threads=24,
    max_bootstrapped_demos=4,
    max_labeled_demos=6,
)

optimized_model = mipro_optimizer.compile(
    student=SentimentClassifier(),
    trainset=trainset,
    valset=devset,
)

optimized_evaluator = Evaluate(
    devset=devset,
    metric=evaluate_sentiment_analysis,
    num_threads=2,
    display_progress=True,
)

optimized_score = optimized_evaluator(optimized_model)
print(f"📈 Optimized Score: {optimized_score}")

print("\n✅ Running inference on test emails:")
for example in testset:
    pred = optimized_model(example.email)
    print("-" * 50)
    print(f"Email: {example.email}")
    print(f"Sentiment: {pred.sentiment}")
    print(f"Confidence: {pred.confidence}")
    print(f"Reasoning: {pred.reasoning}")

optimized_model.save("dspy/sentiment_mipro_v2.json")
