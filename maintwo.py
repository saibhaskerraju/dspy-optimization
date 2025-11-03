import dspy
import os
import json
from typing import Literal, List
import mlflow
import random


mlflow.dspy.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Create a unique name for your experiment.
mlflow.set_experiment("DSPyToday")

key = os.getenv("AZURE_OPENAI_API_KEY")
# llm = dspy.LM(model=f"azure/{os.getenv('AZURE_OPENAI_MODEL')}", api_key=key, api_base=os.getenv("AZURE_OPENAI_ENDPOINT"), api_version=os.getenv("AZURE_OPENAI_API_VERSION"))
llm = dspy.LM('ollama_chat/llama3.2:1b', api_base='http://host.docker.internal:11434', api_key='')

dspy.settings.configure(lm=llm, trace=[])
# updating config
dspy.configure_cache(enable_memory_cache=False, enable_disk_cache=False)

dspy.configure(adapter=dspy.JSONAdapter())

with open('data/nps_comments.json', 'r') as f:
    nps_data = json.loads(f.read())
    
topics = set()

for rec in nps_data: 
    for t in rec['topics']: 
        topics.add(t)

print(list(topics))



class NPSTopic(dspy.Signature):
    """Classify NPS topics"""

    comment: str = dspy.InputField()
    answer: List[Literal['Slow or Unreliable Shipping', 'Inaccurate Product Descriptions or Photos', 'Limited Size or Shade Availability', 
                    'Unresponsive or Generic Customer Support', 'Website or App Bugs', 'Confusing Loyalty or Discount Systems', 
                    'Complicated Returns or Exchanges', 'Customs and Import Charges', 'Difficult Product Discovery', 
                    'Damaged or Incorrect Items']] = dspy.OutputField()

#print(nps_data[0])
nps_topic_model = dspy.ChainOfThought(NPSTopic)

response = nps_topic_model(comment = "Absolutely frustrated! Every time I find something I love, it's sold out in my size. What's the point of having a wishlist if nothing is ever available?")
print("*************** response: ********************", response)

dspy.inspect_history(n = 1)

trainset = []
valset = []
for rec in nps_data: 
    if random.random() <= 0.5:
        trainset.append(
            dspy.Example(
                comment = rec['comment'],
                answer = rec['topics']
            ).with_inputs('comment')
        )
    else: 
        valset.append(
            dspy.Example(
                comment = rec['comment'],
                answer = rec['topics']
            ).with_inputs('comment')
        )
        
def list_exact_match(example, pred, trace=None):
    """Custom metric for comparing lists of topics"""
    try:
        pred_answer = pred.answer
        expected_answer = example.answer
        
        # Convert to sets for order-independent comparison
        if isinstance(pred_answer, list) and isinstance(expected_answer, list):
            return set(pred_answer) == set(expected_answer)
        else:
            return pred_answer == expected_answer
    except Exception as e:
        print(f"Error in metric: {e}")
        return False
    
tp = dspy.MIPROv2(metric=list_exact_match, auto="light", num_threads=24)

opt_nps_topic_model =  tp.compile(
    nps_topic_model, 
    trainset=trainset, 
    valset=valset,
    requires_permission_to_run = False, provide_traceback=True)

opt_nps_topic_model(comment = "Absolutely frustrated! Every time I find something I love, it's sold out in my size. What's the point of having a wishlist if nothing is ever available?"
)

dspy.inspect_history(n = 1)