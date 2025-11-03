import dspy
import os
import json
#import logging
import mlflow
mlflow.dspy.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Create a unique name for your experiment.
mlflow.set_experiment("DSPyToday")

# logging.basicConfig(level=logging.DEBUG)
# os.environ['DSPY_DEBUG'] = '1'

key = os.getenv("AZURE_OPENAI_API_KEY")
llm = dspy.LM(model=f"azure/{os.getenv('AZURE_OPENAI_MODEL')}", api_key=key, api_base=os.getenv("AZURE_OPENAI_ENDPOINT"), api_version=os.getenv("AZURE_OPENAI_API_VERSION"))

dspy.settings.configure(lm=llm, trace=[])

# class SentimentClassifier(dspy.Signature):
#     """Classify the sentiment of a text."""

#     text: str = dspy.InputField(desc="input text to classify sentiment")
#     sentiment: int = dspy.OutputField(
#         desc="sentiment, the higher the more positive", ge=0, le=10
#     )

# predict = dspy.Predict(SentimentClassifier) 

# output = predict(text="I am feeling pretty happy!")
# print(output)

# print(f"The sentiment is: {output.sentiment}")


# dspy.inspect_history(n=1)

def search_wikipedia(query: str) -> list[str]:
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
    return [x["text"] for x in results]

react = dspy.ReAct("question -> answer", tools=[search_wikipedia])


# Load trainset
trainset = []
with open("trainset.jsonl", "r") as f:
    for line in f:
        trainset.append(dspy.Example(**json.loads(line)).with_inputs("question"))

# Load valset
valset = []
with open("valset.jsonl", "r") as f:
    for line in f:
        valset.append(dspy.Example(**json.loads(line)).with_inputs("question"))
        
# Overview of the dataset.
print(trainset[0])

tp = dspy.MIPROv2(
    metric=dspy.evaluate.answer_exact_match,
    auto="light",
    num_threads=16
)

dspy.cache.load_memory_cache("./memory_cache.pkl")

optimized_react = tp.compile(
    react,
    trainset=trainset,
    valset=valset,
    requires_permission_to_run=False,
)

optimized_react.react.signature
optimized_react.react.demos

evaluator = dspy.Evaluate(
    metric=dspy.evaluate.answer_exact_match,
    devset=valset,
    display_table=True,
    display_progress=True,
    num_threads=24,
)
original_score = evaluator(react)
print(f"Original score: {original_score}")

optimized_score = evaluator(optimized_react)
print(f"Optimized score: {optimized_score}")