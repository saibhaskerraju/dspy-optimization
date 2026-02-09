"""DSPy Few-Shot Usage: Load optimized JSON and run inference."""

import os
import dspy


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


email_router = EmailRouter()
email_router.load("dspy/email_router_bootstrapfewshot.json")

user_email = (
    "Hi team, my monthly subscription increased without notice. "
    "Please explain the new charge and adjust it back."
)

prediction = email_router(user_email)

print("=" * 50)
print(f"Category: {prediction.category}")
print(f"Confidence: {prediction.confidence}")
print(f"Reasoning: {prediction.reasoning}")
