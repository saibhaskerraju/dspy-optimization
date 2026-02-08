"""
Sample: DSPy ReAct tool calling with two local tools.

This script mirrors the traditional OpenAI tool-calling example, but uses
DSPy ReAct to let the model decide when to call tools. Two tools are provided:
  1) list_files(directory): browse directory entries with metadata
  2) run_command(command): execute safe shell commands with constraints

How it works:
  - You provide a question (for example, disk usage or directory listing).
  - ReAct decides whether to call a tool and uses it as needed.
  - The final answer is returned in the `answer` field.

Requirements:
  - Python 3.8+
  - dspy installed
  - LLM configured for DSPy (env or config per your workflow)

Notes:
  - Tools are read-only and use an allowlist for safety.
  - Adjust the allowed commands list to fit your environment.
"""


import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Dict, List
import mlflow
import dspy
mlflow.dspy.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Create a unique name for your experiment.
mlflow.set_experiment("DsPyToolCalling")

# logging.basicConfig(level=logging.DEBUG)
os.environ['DSPY_DEBUG'] = '1'

key = os.getenv("AZURE_OPENAI_API_KEY")
llm = dspy.LM(model=f"azure/{os.getenv('AZURE_OPENAI_MODEL')}", api_key=key, api_base=os.getenv(
    "AZURE_OPENAI_ENDPOINT"), api_version=os.getenv("AZURE_OPENAI_API_VERSION"))

dspy.settings.configure(lm=llm, trace=[])


class ToolCallingSignature(dspy.Signature):
    """Answer a question using tools when helpful."""

    question: str = dspy.InputField(desc="User question")
    answer: str = dspy.OutputField(desc="Final response")


def list_files(directory: str) -> Dict[str, Any]:
    """Browse directory contents with basic metadata."""
    target = Path(directory).expanduser().resolve()
    if not target.exists():
        return {"error": f"Path does not exist: {target}"}
    if not target.is_dir():
        return {"error": f"Path is not a directory: {target}"}

    entries = []
    for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
        try:
            stat = child.stat()
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "type": "dir" if child.is_dir() else "file",
                            "size_bytes": stat.st_size,
                            "modified_ts": int(stat.st_mtime),
                }
            )
        except OSError as exc:
            entries.append(
                {"name": child.name, "path": str(child), "error": str(exc)})

    return {"path": str(target), "entries": entries}


def run_command(command: str) -> Dict[str, Any]:
    """Execute safe shell commands with strict allowlist and no shell."""
    if not command or not command.strip():
        return {"error": "Command is empty."}

    allowed = {"ls", "pwd", "whoami", "date",
               "cat", "head", "tail", "wc", "du"}
    try:
        args = shlex.split(command)
    except ValueError as exc:
        return {"error": f"Failed to parse command: {exc}"}

    if not args:
        return {"error": "Command is empty."}
    if args[0] not in allowed:
        return {"error": f"Command not allowed: {args[0]}"}

    disallowed_tokens = {"|", ";", "&&", "||", ">", ">>", "<"}
    if any(token in disallowed_tokens for token in args):
        return {"error": "Pipes and redirection are not allowed."}

    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"Command failed: {exc}"}

    return {
        "command": command,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main() -> None:
    react = dspy.ReAct(signature=ToolCallingSignature,
                       tools=[list_files, run_command])

    question = (
        "What is the disk usage of /workspace and which files or folders are largest?"
    )
    pred = react(question=question)
    print(pred.answer)


if __name__ == "__main__":
    main()
