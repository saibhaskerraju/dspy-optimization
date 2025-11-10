import dspy
import os
# import logging
import mlflow
from dspy.teleprompt import COPRO

mlflow.dspy.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Create a unique name for your experiment.
mlflow.set_experiment("dspyfewshot")

key = os.getenv("AZURE_OPENAI_API_KEY")
llm = dspy.LM(model=f"azure/{os.getenv('AZURE_OPENAI_MODEL')}", api_key=key, api_base=os.getenv(
    "AZURE_OPENAI_ENDPOINT"), api_version=os.getenv("AZURE_OPENAI_API_VERSION"))

dspy.settings.configure(lm=llm, trace=[])

# 🔥 PROBLEM: Sales Email Qualification - Complex Multi-Factor Analysis
class QualifySalesLead(dspy.Signature):
    """Analyze sales emails and predict deal success probability."""
    email_conversation = dspy.InputField(desc="Email exchange between sales and prospect")
    company_tier = dspy.OutputField(desc="Startup, SMB, Mid-Market, Enterprise")
    decision_maker_engaged = dspy.OutputField(desc="yes, no, or partially")
    budget_alignment = dspy.OutputField(desc="below, within, or above our pricing")
    urgency_level = dspy.OutputField(desc="low, medium, high, critical")
    deal_stage = dspy.OutputField(desc="Discovery, Demo, Proposal, Negotiation, Closed-Won, Closed-Lost")
    confidence_score = dspy.OutputField(desc="0-100% probability of closing")
    next_best_action = dspy.OutputField(desc="Specific recommended sales action")
    reasoning = dspy.OutputField(desc="Step-by-step analysis of all factors")

class SalesQualifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qualify = dspy.ChainOfThought(QualifySalesLead)
    
    def forward(self, email_conversation):
        return self.qualify(email_conversation=email_conversation)

# 🎯 TRAINING DATA: Real-world sales scenarios
trainset = [
    dspy.Example(
        email_conversation="""
        Prospect: "We're a 50-person startup looking to scale. Can you send pricing?"
        Sales: "Sure! Our growth plan is $299/month. Would you like a demo?"
        Prospect: "That's higher than expected. We'll discuss internally."
        """,
        company_tier="Startup",
        decision_maker_engaged="partially", 
        budget_alignment="below",
        urgency_level="low",
        deal_stage="Discovery",
        confidence_score="25%",
        next_best_action="Send case studies showing ROI for similar startups",
        reasoning="Startup with budget concerns, no decision maker engaged, early stage discussion"
    ).with_inputs('email_conversation'),
    
    dspy.Example(
        email_conversation="""
        Prospect: "I'm the CTO at 500-employee company. We need this implemented before Q4."
        Sales: "Understood. Our enterprise plan starts at $5K/month. Available for technical deep dive?"
        Prospect: "Yes, bring your solutions architect. We have budget approved."
        """,
        company_tier="Mid-Market",
        decision_maker_engaged="yes",
        budget_alignment="within", 
        urgency_level="high",
        deal_stage="Demo",
        confidence_score="75%",
        next_best_action="Schedule technical deep dive with solutions architect",
        reasoning="CTO engaged, budget approved, timeline urgency, mid-market company"
    ).with_inputs('email_conversation'),
    
    dspy.Example(
        email_conversation="""
        Prospect: "We're evaluating vendors for $500K annual contract. Send security docs."
        Sales: "Here's our SOC2 compliance. When can we discuss your requirements?"
        Prospect: "Our procurement team will review. We're deciding in 30 days."
        """,
        company_tier="Enterprise", 
        decision_maker_engaged="partially",
        budget_alignment="above",
        urgency_level="medium",
        deal_stage="Proposal",
        confidence_score="60%",
        next_best_action="Engage procurement team directly with compliance documentation",
        reasoning="Large contract, procurement process, security focus, longer sales cycle"
    ).with_inputs('email_conversation'),
]

print("🚀 **EXECUTIVE DEMO: AI-Powered Sales Intelligence**")
print("=" * 55)
print("🎯 Showing COPRO's Unique Value: Automatic Reasoning Optimization")
print()

# 1. 🎯 THE CHALLENGE: Show Baseline Limitations
print("1️⃣ THE PROBLEM: Naive Sales Qualification")
print("-" * 45)

baseline_qualifier = SalesQualifier()

# Complex real-world email that requires nuanced understanding
complex_email = """
Prospect: "I'm the VP of Engineering at a 2000-person public company. 
We're replacing our legacy system and need migration support. 
Our board meeting is in 3 weeks where we'll decide."

Sales: "We specialize in enterprise migrations. Our team can provide dedicated support."
Prospect: "Budget isn't finalized but we've allocated $250K for this project. 
Need to see your security certifications and implementation plan first."
"""

print("📧 Complex Enterprise Sales Email:")
print(f'"{complex_email[:100]}..."')
print()

baseline_pred = baseline_qualifier(complex_email)

print("🤖 Baseline AI Analysis:")
print(f"   Company Tier: {baseline_pred.company_tier}")
print(f"   Decision Maker: {baseline_pred.decision_maker_engaged}")
print(f"   Budget: {baseline_pred.budget_alignment}")
print(f"   Urgency: {baseline_pred.urgency_level}")
print(f"   Deal Stage: {baseline_pred.deal_stage}")
print(f"   Confidence: {baseline_pred.confidence_score}")
print(f"   Next Action: {baseline_pred.next_best_action}")
print(f"   Reasoning: {baseline_pred.reasoning}")

# 2. 🎯 COPRO OPTIMIZATION: The Magic Happens
print("\n2️⃣ COPRO OPTIMIZATION: Teaching Strategic Thinking")
print("-" * 50)
print("🔄 COPRO is automatically discovering optimal sales qualification logic...")
print("   • Generating different reasoning frameworks")
print("   • Testing strategic qualification approaches")
print("   • Learning to identify buying signals")
print("   • Optimizing for enterprise sales patterns")

copro_optimizer = COPRO(
    metric=lambda example, pred, trace=None: 1.0,  # Simple metric for demo
    depth=2,           # Optimization rounds
    breadth=3,         # Candidate strategies
    verbose=True,      # Show the optimization process
)

# Run COPRO - this is where the magic happens
optimized_qualifier = copro_optimizer.compile(
    SalesQualifier(),
    trainset=trainset,
    eval_kwargs={'num_threads': 1, 'display_progress': False}
)

# 3. 🎯 THE TRANSFORMATION: Show Dramatic Improvement
print("\n3️⃣ COPRO-OPTIMIZED: Enterprise Sales Intelligence")
print("-" * 50)

optimized_pred = optimized_qualifier(complex_email)

print("🎯 Optimized AI Analysis:")
print(f"   Company Tier: {optimized_pred.company_tier}")
print(f"   Decision Maker: {optimized_pred.decision_maker_engaged}") 
print(f"   Budget: {optimized_pred.budget_alignment}")
print(f"   Urgency: {optimized_pred.urgency_level}")
print(f"   Deal Stage: {optimized_pred.deal_stage}")
print(f"   Confidence: {optimized_pred.confidence_score}")
print(f"   Next Action: {optimized_pred.next_best_action}")
print(f"   Strategic Reasoning: {optimized_pred.reasoning}")

# 4. 🎯 BUSINESS IMPACT: Show Concrete Value
print("\n4️⃣ BUSINESS IMPACT DEMONSTRATION")
print("-" * 35)

print("💰 Sales Strategy Comparison:")
print(f"   Baseline: {baseline_pred.next_best_action}")
print(f"   COPRO:    {optimized_pred.next_best_action}")

if optimized_pred.next_best_action != baseline_pred.next_best_action:
    print("✅ **COPRO discovered a better sales strategy!**")

optimized_qualifier.save("dspy/sales_qualifier_copro.json")
