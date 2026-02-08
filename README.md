# DsPy - Declarative Self-improving Python 
DSPy is a framework for building and optimizing LLM programs using declarative components. This repo provides concise, hands-on examples that show why DSPy outperforms prompt-only approaches in real projects.

- DSPy website: https://dspy.ai/
- DSPy GitHub: https://github.com/stanfordnlp/dspy

# Why DSPy Beats Prompt-Only Systems

- 🧩 **Programs, Not Prompts**: Build typed modules and pipelines instead of brittle string templates.
- 📊 **Data-Driven Optimization**: Learn the best prompts and examples from real data, automatically.
- 🧠 **Reasoning You Can Trust**: Compose chain-of-thought steps without manual prompt tinkering.
- ✅ **Eval-First Workflow**: Measure quality with built-in evaluations and track regressions.
- 🚀 **Ready for Production**: Versioned configs, repeatable runs, and vendor-agnostic deployment.
- 🔁 **Continuous Improvement**: Feedback loops make the system better as usage grows.

**Result?** AI systems that stay reliable and improve as new data arrives.

# Documentation

To understand the impoortance of DsPy framework. we need to understand the potential pain points of prompt engineering.

This repository provides practical examples comparing trditional manual prompt engineering with DSPy-based declarative programming.

- [1. Prompt Engineering Problems](docs/1.Prompt_Engineering.md) - Explore common challenges in prompt engineering and how DSPy addresses them.

# Local setup

- Open this solution in dev container mode run the files mentioned in the [Documentation Hierarchy](#documentation-hierarchy)

# Enable ML-Flow for local debugging

- Run the following command
```sh
mlflow server --backend-store-uri sqlite:///mydb.sqlite
```

# Documentation Hierarchy

- [Prompt Engineering Problems](docs/1.Prompt_Engineering.md)
- [Optimizers Used](docs/3.Optimization_Solution.md)