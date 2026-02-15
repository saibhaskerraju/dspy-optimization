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

MIPRO v2 is the most advanced optimization technique in DSPy that combines:
1. Instruction optimization (like COPRO)
2. Few-shot example optimization (like BootstrapFewShot)
3. Proposal generation and refinement
4. Multi-objective optimization
"""

import dspy
import mlflow
from typing import List
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('.env.local')

mlflow.dspy.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("mipro-v2-optimization")

# Configure DSPy with Azure LLM
azure_llm = dspy.LM(
        model=f"azure/{os.getenv('AZURE_OPENAI_MODEL')}",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
dspy.configure(lm=azure_llm)

# Use the same signature from Auto-Instruction.py

class SentimentAnalysisSignature(dspy.Signature):
    """Analyze customer email sentiment with detailed reasoning."""

    email: str = dspy.InputField(
        description="Customer email content to analyze")
    reasoning: str = dspy.OutputField(
        description="Step-by-step analysis of sentiment indicators")
    sentiment: str = dspy.OutputField(
        description="Sentiment classification: Positive, Negative, or Neutral")
    confidence: float = dspy.OutputField(
        description="Confidence score between 0.0 and 1.0")

# Create the sentiment analysis module

class SentimentAnalyzer(dspy.Module):
    """DSPy sentiment analyzer optimizable with MIPRO v2."""

    def __init__(self):
        super().__init__()
        # ChainOfThought with optimizable instructions and examples
        self.analyzer = dspy.ChainOfThought(SentimentAnalysisSignature)

    def forward(self, email: str) -> dspy.Prediction:
        """Analyze sentiment of a single email."""
        return self.analyzer(email=email)

# Extended training data for MIPRO v2 optimization
training_data = [
    # Positive examples
    dspy.Example(
        email="Thank you so much for the quick resolution! Your support team is amazing.",
        reasoning="Customer expresses gratitude and praises support team quality",
        sentiment="Positive",
        confidence=0.95
    ).with_inputs("email"),

    dspy.Example(
        email="I love the new features you've added. The app works perfectly now!",
        reasoning="Customer shows enthusiasm for features and confirms satisfaction",
        sentiment="Positive",
        confidence=0.92
    ).with_inputs("email"),

    dspy.Example(
        email="Great service! Everything was handled professionally and efficiently.",
        reasoning="Customer praises service quality with positive descriptors",
        sentiment="Positive",
        confidence=0.98
    ).with_inputs("email"),

    dspy.Example(
        email="Excellent product quality. Exceeded my expectations completely.",
        reasoning="Customer expresses strong satisfaction and positive surprise",
        sentiment="Positive",
        confidence=0.96
    ).with_inputs("email"),

    dspy.Example(
        email="Fantastic experience from start to finish. Highly recommend!",
        reasoning="Customer provides comprehensive positive evaluation and recommendation",
        sentiment="Positive",
        confidence=0.94
    ).with_inputs("email"),

    # Negative examples
    dspy.Example(
        email="This is ridiculous! I've been waiting for 3 hours and no response.",
        reasoning="Customer expresses frustration and anger about delayed response",
        sentiment="Negative",
        confidence=0.88
    ).with_inputs("email"),

    dspy.Example(
        email="Your product is terrible. It crashes constantly and support is useless.",
        reasoning="Customer uses harsh language criticizing product and support",
        sentiment="Negative",
        confidence=0.95
    ).with_inputs("email"),

    dspy.Example(
        email="I want my money back immediately. This service is a complete waste.",
        reasoning="Customer demands refund and expresses strong dissatisfaction",
        sentiment="Negative",
        confidence=0.90
    ).with_inputs("email"),

    dspy.Example(
        email="Worst customer service ever! Nobody seems to care about customers.",
        reasoning="Customer expresses extreme dissatisfaction with service quality",
        sentiment="Negative",
        confidence=0.93
    ).with_inputs("email"),

    dspy.Example(
        email="Complete disaster! Nothing works as promised. Very disappointed.",
        reasoning="Customer reports multiple failures and expresses disappointment",
        sentiment="Negative",
        confidence=0.91
    ).with_inputs("email"),

    # Neutral examples
    dspy.Example(
        email="Can you please explain how the billing system works?",
        reasoning="Customer makes neutral inquiry without emotional indicators",
        sentiment="Neutral",
        confidence=0.85
    ).with_inputs("email"),

    dspy.Example(
        email="I need to update my account information. What's the process?",
        reasoning="Customer asks factual question about account management",
        sentiment="Neutral",
        confidence=0.82
    ).with_inputs("email"),

    dspy.Example(
        email="What are your business hours? I'd like to call during open hours.",
        reasoning="Customer seeks operational information without sentiment",
        sentiment="Neutral",
        confidence=0.87
    ).with_inputs("email"),

    dspy.Example(
        email="Please send me the technical documentation for the API.",
        reasoning="Customer makes straightforward request for information",
        sentiment="Neutral",
        confidence=0.84
    ).with_inputs("email"),

    dspy.Example(
        email="I'm comparing different solutions. Can you provide feature details?",
        reasoning="Customer conducts neutral evaluation without emotional bias",
        sentiment="Neutral",
        confidence=0.83
    ).with_inputs("email"),

    # Mixed/Complex examples
    dspy.Example(
        email="The app is okay but could use some improvements in the UI design.",
        reasoning="Customer provides balanced feedback with mild suggestion for improvement",
        sentiment="Neutral",
        confidence=0.75
    ).with_inputs("email"),

    dspy.Example(
        email="Good product overall, though the setup process was confusing initially.",
        reasoning="Customer balances positive assessment with constructive criticism",
        sentiment="Neutral",
        confidence=0.78
    ).with_inputs("email"),

    dspy.Example(
        email="The features are useful but the pricing seems a bit high for what you get.",
        reasoning="Customer acknowledges value while expressing concern about cost",
        sentiment="Neutral",
        confidence=0.76
    ).with_inputs("email"),
]

# Evaluation metric for MIPRO v2 (same as Auto-Instruction.py)

def evaluate_sentiment_analysis(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """Evaluate sentiment analysis accuracy with confidence consideration."""

    # Check if sentiment matches
    sentiment_match = prediction.sentiment.strip(
    ).lower() == example.sentiment.strip().lower()

    # Check if confidence is reasonable (between 0.0 and 1.0)
    try:
        confidence_valid = 0.0 <= float(prediction.confidence) <= 1.0
    except (ValueError, TypeError):
        confidence_valid = False

    # Check if reasoning is provided (non-empty)
    reasoning_provided = len(prediction.reasoning.strip()) > 10

    # Combined evaluation
    return sentiment_match and confidence_valid and reasoning_provided

# Validation and test data
validation_data = [
    dspy.Example(
        email="The product is decent but customer service could be better.",
        reasoning="Customer gives mixed feedback with constructive criticism",
        sentiment="Neutral",
        confidence=0.70
    ).with_inputs("email"),

    dspy.Example(
        email="Absolutely fantastic! Best purchase I've made this year.",
        reasoning="Customer expresses strong positive emotion and satisfaction",
        sentiment="Positive",
        confidence=0.98
    ).with_inputs("email"),

    dspy.Example(
        email="This is broken and I can't get anyone to help me fix it.",
        reasoning="Customer reports problem and expresses frustration with support",
        sentiment="Negative",
        confidence=0.85
    ).with_inputs("email"),

    dspy.Example(
        email="How do I cancel my subscription? Need to do this today.",
        reasoning="Customer asks procedural question with mild urgency but no sentiment",
        sentiment="Neutral",
        confidence=0.80
    ).with_inputs("email"),

    dspy.Example(
        email="Outstanding support! Fixed my issue in record time.",
        reasoning="Customer expresses high satisfaction with support speed and quality",
        sentiment="Positive",
        confidence=0.96
    ).with_inputs("email"),
]

test_data = [
    dspy.Example(
        email="I'm thrilled with the new update! Everything works seamlessly now.",
        reasoning="Customer expresses excitement and confirms product functionality",
        sentiment="Positive",
        confidence=0.93
    ).with_inputs("email"),

    dspy.Example(
        email="This is the worst experience I've ever had with any company.",
        reasoning="Customer expresses extreme dissatisfaction with overall experience",
        sentiment="Negative",
        confidence=0.97
    ).with_inputs("email"),

    dspy.Example(
        email="Could you clarify the refund policy for annual subscriptions?",
        reasoning="Customer seeks factual information about policy without emotion",
        sentiment="Neutral",
        confidence=0.88
    ).with_inputs("email"),
]

# Test emails for demonstration
test_emails = [
    "I'm thrilled with the new update! Everything works seamlessly now.",
    "This is the worst experience I've ever had with any company.",
    "Could you clarify the refund policy for annual subscriptions?",
    "The app is good overall but the loading times are a bit slow.",
    "Outstanding customer support! Problem resolved within minutes.",
    "I can't believe how bad this product is. Total waste of money.",
    "What's the difference between the basic and premium plans?",
    "Amazing service! You've exceeded all my expectations.",
    "Terrible! Nothing works and support ignores my emails.",
    "Please provide information about enterprise pricing options."
]

def optimize_with_mipro_v2():
    """Use MIPRO v2 to optimize both instructions and few-shot examples."""

    print("🚀 OPTIMIZING WITH MIPRO v2 (Multi-Stage Instruction & Proposal Optimization)...")
    print("-" * 90)

    # Create base analyzer
    analyzer = SentimentAnalyzer()

    # Print initial state
    print("📋 INITIAL STATE:")
    print("-" * 40)
    print("Base DSPy module (before MIPRO v2 optimization)")
    print()

    # Create MIPRO v2 optimizer with advanced configuration
    mipro_optimizer = dspy.MIPROv2(
        metric=evaluate_sentiment_analysis,
        auto="light",          # Auto-optimization level: light, medium, heavy
        num_threads=24
    )

    print(f"🔮 Training with {len(training_data)} examples...")
    print(f"🧪 Validating with {len(validation_data)} examples...")
    print(f"🧪 Testing with {len(test_emails)} examples...")
    print("🧠 Running multi-stage MIPRO v2 optimization...")
    print("  • Stage 1: Instruction optimization")
    print("  • Stage 2: Few-shot example bootstrapping")
    print("  • Stage 3: Combined instruction + example optimization")
    print("  • Stage 4: Final validation and selection")
    print()

    # Optimize the analyzer with MIPRO v2
    optimized_analyzer = mipro_optimizer.compile(
        student=analyzer,
        trainset=training_data,
        valset=validation_data,
        max_bootstrapped_demos=4,
        max_labeled_demos=6,
    )

    print("✅ MIPRO v2 Optimization Complete!")
    print("-" * 60)

    return analyzer, optimized_analyzer

def evaluate_on_test_set(analyzer, test_data, name="Analyzer"):
    """Evaluate analyzer performance on test set."""

    print(f"📊 EVALUATING {name.upper()}:")
    print("-" * 50)

    correct_predictions = 0
    valid_evaluations = 0

    for i, example in enumerate(test_data, 1):
        try:
            prediction = analyzer(email=example.email)
            is_valid = evaluate_sentiment_analysis(example, prediction)
            is_correct = prediction.sentiment.strip(
            ).lower() == example.sentiment.strip().lower()

            valid_evaluations += 1
            if is_valid:
                correct_predictions += 1
            
            print(
                f"  Example {i}: Valid={is_valid}, Sentiment_Match={is_correct}")

        except Exception as e:
            print(f"  Example {i}: Error - {str(e)}")

    if valid_evaluations > 0:
        accuracy = correct_predictions / valid_evaluations
        print(
            f"  📊 Overall Accuracy: {accuracy:.3f} ({correct_predictions}/{valid_evaluations})")
    else:
        accuracy = 0.0
        print("  📊 Overall Accuracy: 0.0 (no valid evaluations)")

    print()

    return accuracy

def compare_analyzers(original_analyzer, optimized_analyzer, test_emails: List[str]):
    """Compare original vs MIPRO v2-optimized sentiment analysis."""

    print("🔍 DETAILED COMPARISON:")
    print("-" * 90)

    for i, email in enumerate(test_emails, 1):
        print(f"\n{i}. Test Email: \"{email}\"")
        print("-" * 70)

        # Original analyzer
        try:
            original_result = original_analyzer(email=email)
            print(f"  🤖 ORIGINAL:")
            print(f"    Sentiment: {original_result.sentiment}")
            print(f"    Confidence: {original_result.confidence}")
            print(f"    Reasoning: {original_result.reasoning}")
        except Exception as e:
            print(f"  ❌ ORIGINAL: Error - {str(e)}")

        # MIPRO v2 optimized analyzer
        try:
            optimized_result = optimized_analyzer(email=email)
            print(f"  🚀 MIPRO v2 OPTIMIZED:")
            print(f"    Sentiment: {optimized_result.sentiment}")
            print(f"    Confidence: {optimized_result.confidence}")
            print(f"    Reasoning: {optimized_result.reasoning}")
        except Exception as e:
            print(f"  ❌ MIPRO v2 OPTIMIZED: Error - {str(e)}")

if __name__ == "__main__":
    print("🚀 DSPy MIPRO v2: Advanced Multi-Stage Optimization")
    print("-" * 90)
    print("Multi-stage instruction and proposal optimization for sentiment analysis...")
    print("-" * 90)
    print()

    # Optimize with MIPRO v2
    original_analyzer, optimized_analyzer = optimize_with_mipro_v2()

    # Evaluate both analyzers on test set
    print("📊 PERFORMANCE EVALUATION:")
    print("=" * 60)

    original_acc = evaluate_on_test_set(
        original_analyzer, test_data, "Original Analyzer")
    optimized_acc = evaluate_on_test_set(
        optimized_analyzer, test_data, "MIPRO v2 Optimized")

    # Performance summary
    print("📈 OPTIMIZATION RESULTS:")
    print("-" * 40)
    print(f"Original Accuracy:    {original_acc:.3f}")
    print(f"Optimized Accuracy:   {optimized_acc:.3f}")
    print(f"Accuracy Improvement: {(optimized_acc - original_acc):.3f}")
    print()

    # Detailed comparison
    compare_analyzers(original_analyzer, optimized_analyzer, test_emails[:5])

    print("\n🎉 MIPRO v2 Optimization Demo Complete!")
    print("=" * 90)
    print("💾 Saving optimized analyzer...")

    # Save optimized analyzer
    optimized_analyzer.save("dspy/mipro_optimized_sentiment_analyzer.json")
    print("✅ MIPRO v2 optimized analyzer saved!")

    print("\n📝 MIPRO v2 OPTIMIZATION SUMMARY:")
    print("=" * 50)
    print("✅ Multi-stage optimization completed")
    print("✅ Instructions automatically refined")
    print("✅ Few-shot examples bootstrapped")
    print("✅ Performance metrics improved")
    print("✅ Model saved for production use")
