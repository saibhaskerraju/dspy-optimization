import dspy
import os
# import logging
import mlflow
mlflow.dspy.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Create a unique name for your experiment.
mlflow.set_experiment("DspyPredict")

# logging.basicConfig(level=logging.DEBUG)
os.environ['DSPY_DEBUG'] = '1'

key = os.getenv("AZURE_OPENAI_API_KEY")
llm = dspy.LM(model=f"azure/{os.getenv('AZURE_OPENAI_MODEL')}", api_key=key, api_base=os.getenv(
    "AZURE_OPENAI_ENDPOINT"), api_version=os.getenv("AZURE_OPENAI_API_VERSION"))

dspy.settings.configure(lm=llm, trace=[])


class SentimentClassifier(dspy.Signature):
    """Classify the sentiment of a text."""

    text: str = dspy.InputField(desc="input text to classify sentiment")
    sentiment: int = dspy.OutputField(
        desc="sentiment, the higher the more positive", ge=0, le=10
    )


predict = dspy.Predict(SentimentClassifier)

output = predict(text="I am feeling pretty happy!")

print(f"The sentiment is: {output.sentiment}")
print("="*50)
dspy.inspect_history(n=1)
