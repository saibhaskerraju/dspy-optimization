"""
Traditional Few-Shot Learning (DSPy): Email Classification

SCENARIO:
Classify customer support emails into categories (Billing, Technical, Refund, General)
to route them to appropriate teams automatically.

APPROACH:
- DSPy BootstrapFewShot for demo selection
- Hand-curated training examples
- No external training data optimization beyond bootstrapped selection
"""

import os
import dspy
import mlflow
from dspy.teleprompt import BootstrapFewShot


mlflow.dspy.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("dspyfewshot")

key = os.getenv("AZURE_OPENAI_API_KEY")
llm = dspy.LM(
    model=f"azure/{os.getenv('AZURE_OPENAI_MODEL')}",
    api_key=key,
    api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

dspy.settings.configure(lm=llm, trace=[])


class EmailClassifier(dspy.Signature):
    """Classify a support email into a single category."""

    email = dspy.InputField(desc="Customer support email text")
    category = dspy.OutputField(desc="Billing, Technical, Refund, or General")
    confidence = dspy.OutputField(desc="Integer from 0 to 10")
    reasoning = dspy.OutputField(desc="Short justification")


class EmailRouter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(EmailClassifier)

    def forward(self, email: str) -> dspy.Prediction:
        return self.classify(email=email)


def evaluate_classification(example, pred, trace=None) -> bool:
    return example.category.lower() == pred.category.lower()


training_data = [
    dspy.Example(
        email="How do I delete my account permanently?",
        category="Billing",
        reasoning="Customer asking about account management procedures",
    ).with_inputs("email"),
    dspy.Example(
        email="I was charged twice this month for my subscription. Can you help?",
        category="Billing",
        reasoning="Customer reporting duplicate billing charge",
    ).with_inputs("email"),
    dspy.Example(
        email="My credit card was declined but the bank says it is fine.",
        category="Billing",
        reasoning="Payment failure and card decline are billing-related",
    ).with_inputs("email"),
    dspy.Example(
        email="The app crashes every time I try to open it after the update.",
        category="Technical",
        reasoning="App crash after update is a technical issue",
    ).with_inputs("email"),
    dspy.Example(
        email="I cannot reset my password; the reset link keeps expiring.",
        category="Technical",
        reasoning="Password reset failure is a technical problem",
    ).with_inputs("email"),
    dspy.Example(
        email="Please cancel my plan and issue a refund for this month.",
        category="Refund",
        reasoning="Explicit refund request and cancellation",
    ).with_inputs("email"),
    dspy.Example(
        email="The service did not meet expectations; I want my money back.",
        category="Refund",
        reasoning="Dissatisfaction with service and refund request",
    ).with_inputs("email"),
    dspy.Example(
        email="What are your support hours and where can I find the docs?",
        category="General",
        reasoning="General information request",
    ).with_inputs("email"),
    dspy.Example(
        email="How do I change my company address on the account?",
        category="General",
        reasoning="Account profile update question",
    ).with_inputs("email"),
    dspy.Example(
        email="My invoice shows the wrong tax ID and needs correction.",
        category="Billing",
        reasoning="Invoice correction is billing-related",
    ).with_inputs("email"),
]


print("🚀 DSPy Few-Shot Demo: Email Classification")
print("=" * 50)

baseline_router = EmailRouter()

user_email = (
    "Hi team, my monthly subscription increased without notice. "
    "Please explain the new charge and adjust it back."
)

baseline_pred = baseline_router(user_email)
print("\nBaseline (Zero-Shot) Prediction:")
print(f"Category: {baseline_pred.category}")
print(f"Confidence: {baseline_pred.confidence}")
print(f"Reasoning: {baseline_pred.reasoning}")


bootstrap_optimizer = dspy.BootstrapFewShot(
    metric=evaluate_classification,
    max_bootstrapped_demos=5,
    max_labeled_demos=len(training_data),
    max_rounds=3,
)

optimized_router = bootstrap_optimizer.compile(
    EmailRouter(),
    trainset=training_data,
)

# optimized_pred = optimized_router(user_email)
# print("\nOptimized (BootstrapFewShot) Prediction:")
# print(f"Category: {optimized_pred.category}")
# print(f"Confidence: {optimized_pred.confidence}")
# print(f"Reasoning: {optimized_pred.reasoning}")
optimized_router.save("dspy/email_router_bootstrapfewshot.json")
