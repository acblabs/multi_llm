# 🚀 Multi-Model Agent  
### Google ADK • Vertex AI Agent Engine • Multi-LLM Ensemble System

<p align="center">
  <b>Designing systems of models — not just using models</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Framework-Google%20ADK-blue" />
  <img src="https://img.shields.io/badge/Platform-Vertex%20AI-orange" />
  <img src="https://img.shields.io/badge/Architecture-Multi--Agent-green" />
  <img src="https://img.shields.io/badge/Pattern-Ensemble%20LLM-purple" />
  <img src="https://img.shields.io/badge/Focus-Orchestration-critical" />
</p>

---

## 🧠 Overview

This project implements a **multi-model AI agent system** using the Google Agent Development Kit (ADK), designed to explore:

> **How multiple LLMs can collaborate to outperform any single model**

Instead of relying on one model, this system:

- Executes **multiple LLMs in parallel**
- Uses **Gemini as a meta-learner (synthesizer + contributor)**
- Applies **ensemble + hybrid orchestration patterns**
- Implements **retry, fallback, and reliability layers**
- Supports **session memory + tracing**
- Deploys to **Vertex AI Agent Engine**

---

## 🔥 Core Insight

```text
Better outputs come from diverse reasoning, not repeated reasoning
🧬 Architecture
🔥 Primary Pattern (Best Performing)
User Prompt
      ↓
┌──────────────┬──────────────┬──────────────┐
│   OpenAI     │    Claude    │     Grok     │
│ (Structure)  │ (Reasoning)  │ (Creativity) │
└──────┬───────┴──────┬───────┴──────┬───────┘
       │              │              │
       └──────────────┴──────────────┘
                      ↓
               ┌──────────────┐
               │   Gemini     │
               │ Synth + Build│
               └──────┬───────┘
                      ↓
               Final Output
🧩 System Components
Component	Role
Gemini	Orchestrator, synthesizer, and contributor
OpenAI	Structured implementation and code quality
Claude	Deep reasoning and robustness
Grok	Exploration and alternative approaches
LiteLLM	Unified API layer across providers
Tools Layer	Execution interface for models
Retry/Fallback	Reliability + fault tolerance
Session Memory	Context persistence
Tracing/Monitoring	Observability + debugging
🧪 Experimental Results
🏆 Code Quality Ranking
Rank	Architecture
🥇	Multi-LLM Parallel + Gemini
🥈	Multi-LLM Sequential
🥉	Single-LLM Parallel
❌	Single-LLM Sequential
📊 Key Findings
⚡ ~55% fewer tokens vs sequential pipelines
🧠 Higher quality outputs with parallel reasoning
🔥 Multi-model > temperature-based variation
🤖 Gemini performs best as meta-learner + builder
🧠 Orchestration Patterns
❌ Sequential (Weak)
Model A → Model B → Model C
Anchoring bias
Context bloat
Diminishing returns
✅ Parallel (Best)
Model A + Model B + Model C → Gemini
Independent reasoning
High diversity
Strong synthesis
🔥 Hybrid (Optimal)
Parallel → Synthesis → Single refinement
Best balance of cost + quality
📁 Project Structure
multi_model_agent/
├── __init__.py
├── agent.py          # Gemini orchestrator
├── tools.py          # LLM tool wrappers
├── config.py         # Model configuration
├── metrics.py        # Token + cost tracking
├── requirements.txt
└── .env.example
🔑 Features
✅ Multi-LLM Ensemble
Parallel execution across models
True reasoning diversity (not temperature tricks)
✅ Reliability Layer
Retry with exponential backoff
Error classification (retry / fallback / fail)
Cross-model fallback chains
✅ Observability
Token usage tracking
Cost estimation
Debug-friendly architecture
✅ Session Memory
Persistent conversation state
Native Agent Engine support
✅ Production Deployment
Vertex AI Agent Engine
Scalable, managed runtime
🚀 Getting Started
1. Install dependencies
pip install -r requirements.txt
2. Configure environment
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
XAI_API_KEY=your_key

GOOGLE_GENAI_USE_VERTEXAI=TRUE
3. Run locally
adk web
4. Deploy to Vertex AI
adk deploy agent_engine \
  --project=YOUR_PROJECT \
  --region=us-central1 \
  --display_name="Multi-Model Agent" \
  multi_model_agent
☁️ Deployment

This agent runs on Vertex AI Agent Engine, providing:

Managed infrastructure
Session handling
Observability + logging
Scalable execution
🧠 Design Philosophy
Traditional Approach

Pick the best model

This System

Design the best system of models

🔮 Future Work
Weighted ensemble scoring
Automatic model routing
Compile/test validation loops
Skill-based context loading
External execution (Cloud Run validation)
⚠️ Security
Do NOT expose API keys in .env in production
Use GCP Secret Manager
Avoid plaintext keys in Agent Engine UI
🤝 Contributing

Contributions and experiments are welcome.
