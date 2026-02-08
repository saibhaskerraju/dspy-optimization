"""
Sample: OpenAI tool calling with two file system tools.

This script demonstrates a traditional OpenAI client flow (not DSPy) where the
model can call tools to inspect the local workspace and run safe shell commands.
The model is given two tools:
    1) list_files: browse directory entries (files and subfolders) with metadata
    2) run_command: execute safe shell commands with security constraints

How it works:
    - You provide a user question, e.g. "Show me the files in /workspace/traditional"
    - The model decides whether to call a tool.
    - When a tool call is returned, the script executes the tool locally.
    - The tool result is sent back to the model for a final response.

Requirements:
  - Python 3.8+
  - openai Python package
  - OPENAI_API_KEY environment variable set

Notes:
  - This is a minimal, single-file example for clarity.
  - Tools are intentionally small and safe (read-only).
  - Adjust the model name if your account requires a specific one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Dict, List

from openai import AzureOpenAI, OpenAI


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
                {"name": child.name, "path": str(child), "error": str(exc)}
            )

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


def run_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "list_files":
        return list_files(arguments["directory"])
    if tool_name == "run_command":
        return run_command(arguments["command"])
    return {"error": f"Unknown tool: {tool_name}"}


def main() -> None:
    client = AzureOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_MODEL"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "Browse directory contents with metadata.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Directory path",
                        }
                    },
                    "required": ["directory"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Execute safe shell commands with security constraints.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command (allowlisted)",
                        }
                    },
                    "required": ["command"],
                },
            },
        },
    ]

    user_prompt = (
        "whats the disk usuage of current directory and what files are taking up more space?"
    )

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can call tools.",
        },
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_MODEL"),
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message
    if message.tool_calls:
        messages.append(message)
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")
            tool_result = run_tool(tool_name, arguments)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result),
                }
            )

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL"),
            messages=messages,
        )

    final_text = response.choices[0].message.content
    print(final_text)


if __name__ == "__main__":
    main()
