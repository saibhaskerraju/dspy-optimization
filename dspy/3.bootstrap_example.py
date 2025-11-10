import dspy
import os
from dspy.evaluate import Evaluate
#import logging
import mlflow
from dspy.teleprompt import BootstrapFewShot
mlflow.dspy.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Create a unique name for your experiment.
mlflow.set_experiment("dspyfewshot")

key = os.getenv("AZURE_OPENAI_API_KEY")
llm = dspy.LM(model=f"azure/{os.getenv('AZURE_OPENAI_MODEL')}", api_key=key, api_base=os.getenv("AZURE_OPENAI_ENDPOINT"), api_version=os.getenv("AZURE_OPENAI_API_VERSION"))

dspy.settings.configure(lm=llm, trace=[])

# Define signature
class RouteTicket(dspy.Signature):
    """Route support tickets to appropriate teams."""
    ticket_text = dspy.InputField(desc="Customer support ticket content")
    team = dspy.OutputField(desc="One of: Billing, Technical, Sales, Urgent, General")

# Training Data (20 examples)
trainset = [
    dspy.Example(ticket_text="I was double charged this month, need refund", team="Billing").with_inputs('ticket_text'),
    dspy.Example(ticket_text="API giving 500 errors since morning", team="Technical").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Website completely down, customers complaining", team="Urgent").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Want to upgrade to enterprise plan", team="Sales").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Where to download invoice?", team="Billing").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Database connection timeout errors", team="Technical").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Payment failed but money deducted", team="Billing").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Can't login, password reset not working", team="Urgent").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Need pricing for 100 users", team="Sales").with_inputs('ticket_text'),
    dspy.Example(ticket_text="How to export user data?", team="General").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Subscription cancelled but still charged", team="Billing").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Mobile app crashing on iOS", team="Technical").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Server not responding, complete outage", team="Urgent").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Interested in API partnership", team="Sales").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Where is my order confirmation?", team="General").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Tax ID missing from invoice", team="Billing").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Slow response times from API", team="Technical").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Security breach suspected", team="Urgent").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Volume discount for startup?", team="Sales").with_inputs('ticket_text'),
    dspy.Example(ticket_text="How to change company address?", team="General").with_inputs('ticket_text'),
]

# Test Data (10 examples)
testset = [
    dspy.Example(ticket_text="Credit card declined but it should work", team="Billing").with_inputs('ticket_text'),
    dspy.Example(ticket_text="SSL certificate expired error", team="Technical").with_inputs('ticket_text'),
    dspy.Example(ticket_text="All services down across regions", team="Urgent").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Want to discuss custom solution", team="Sales").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Where to find documentation?", team="General").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Invoice amount doesn't match agreement", team="Billing").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Database backup failing", team="Technical").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Unauthorized charges on my account", team="Billing").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Can't process any payments", team="Urgent").with_inputs('ticket_text'),
    dspy.Example(ticket_text="Feature comparison between plans", team="Sales").with_inputs('ticket_text'),
]

# Build the program
class TicketRouter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.router = dspy.ChainOfThought(RouteTicket)
    
    def forward(self, ticket_text):
        return self.router(ticket_text=ticket_text)

def exact_match(example, pred, trace=None):
    return example.team.lower() == pred.team.lower()

def business_critical_match(example, pred, trace=None):
    """Urgent tickets must be correctly identified"""
    if example.team == "Urgent":
        return pred.team == "Urgent"
    return example.team.lower() == pred.team.lower()

def team_wise_accuracy(example, pred, trace=None):
    """Returns team name if correct, else 'wrong' for analysis"""
    return example.team if example.team.lower() == pred.team.lower() else "wrong"

# DEMO TIME!
print("🚀 DSPy Optimization Demo: Ticket Routing System")
print("=" * 50)

# 1. Baseline Evaluation with Built-in Evaluator
print("\n1️⃣ BASELINE EVALUATION (Zero-Shot):")
baseline_router = TicketRouter()

# Use built-in evaluator for baseline
baseline_evaluator = Evaluate(
    devset=testset, 
    metric=exact_match, 
    num_threads=4, 
    display_progress=True
)
baseline_accuracy = baseline_evaluator(baseline_router)
print(f"📊 Baseline Exact Match Accuracy: {baseline_accuracy}")

# Optimize with BootstrapFewShot
print("=" * 40)
print("2️⃣ OPTIMIZING with BootstrapFewShot...")

teleprompter = BootstrapFewShot(
    metric=exact_match,
    max_bootstrapped_demos=5,
    max_rounds=2
)

optimized_router = teleprompter.compile(TicketRouter(), trainset=trainset)

# Test optimized version
print("3️⃣ OPTIMIZED - Few-Shot Performance:")
optimized_evaluator = Evaluate(
    devset=testset, 
    metric=exact_match, 
    num_threads=4, 
    display_progress=True
)
optimized_accuracy = optimized_evaluator(optimized_router)
print(f"📊 Optimized Exact Match Accuracy: {optimized_accuracy}")

# 4. COMPREHENSIVE COMPARISON METRICS
print("=" * 40)
print("\n4️⃣ COMPREHENSIVE COMPARISON METRICS")

# Evaluate both with multiple metrics
metrics = {
    "Exact Match": exact_match,
    "Business Critical (Urgent Caught)": business_critical_match
}

comparison_results = {}

for metric_name, metric_func in metrics.items():
    print(f"\n📈 Evaluating: {metric_name}")
    
    # Evaluate baseline
    baseline_eval = Evaluate(devset=testset, metric=metric_func, num_threads=4)
    baseline_score = baseline_eval(baseline_router)
    
    # Evaluate optimized
    optimized_eval = Evaluate(devset=testset, metric=metric_func, num_threads=4)
    optimized_score = optimized_eval(optimized_router)
    
    comparison_results[metric_name] = {
        'baseline': baseline_score["score"],
        'optimized': optimized_score["score"],
        'improvement': optimized_score['score'] - baseline_score['score']
    }
    
print("\n✅ Summary of Results:")
for metric_name, results in comparison_results.items():
    print(f"• {metric_name}: Baseline = {results['baseline']}, Optimized = {results['optimized']}, Improvement = {results['improvement']}")


# saving both models
baseline_router.save("dspy/baseline_ticket_router.json")
optimized_router.save("dspy/optimized_ticket_router.json")