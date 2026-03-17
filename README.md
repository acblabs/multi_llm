# 🚀 Multi-LLM Agent on Vertex AI (Gemini + OpenAI + Claude + Grok)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![GCP](https://img.shields.io/badge/GCP-Vertex%20AI-orange)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-green)

> A production-ready **multi-LLM agent** using **Vertex AI Agent Builder (ADK)** with **Gemini as the orchestrator** and support for **OpenAI, Claude, and Grok** — including **real-time token + cost tracking**.

---

## 🔥 Why this project?

Most examples show **single-model agents**.

This project demonstrates a **real-world multi-model architecture**:

* 🧠 Use **Gemini (Vertex AI)** as the default (fast + cost-efficient)
* 🔌 Call **OpenAI / Claude / Grok** only when needed
* 💰 Track **token usage + cost per call**
* ☁️ Deploy to **Vertex AI Agent Engine**
* 🖥️ Develop locally with **ADK Dev UI**

👉 This is a practical production pattern for modern AI systems.

---

## 🧠 Architecture

```text
User
 ↓
Vertex Agent Engine
 ↓
Gemini (primary orchestrator)
 ↓
Tool routing layer
 ├── OpenAI
 ├── Claude
 └── Grok
 ↓
Response + cost tracking
```

---

## ✨ Features

* ✅ Multi-LLM orchestration (Gemini + OpenAI + Claude + Grok)
* ✅ Tool-based routing
* ✅ Per-call token + cost tracking
* ✅ Local development UI (`adk web`)
* ✅ Vertex Agent Engine deployment
* ✅ Clean, extensible Python structure

---

## 💡 Example Output

```text
Here is your rewritten email...

[Claude: 812 tokens | $0.0021]
```

---

## 📦 Project Structure

```text
multi_llm/
├── __init__.py
├── agent.py            # shim → app.agent
├── app/
│   ├── agent.py
│   ├── tools.py
│   └── config.py
├── requirements.txt
└── .env
```

---

## ⚙️ Prerequisites

* Python **3.10+**
* Google Cloud project with billing enabled
* Vertex AI + Cloud Storage APIs enabled
* GCS staging bucket (e.g. `gs://my-gcp-bucket`) 
* Authenticated locally:

```bash
gcloud auth application-default login
```

* API keys with **available credits / billing enabled** for:

  * OpenAI
  * Anthropic (Claude)
  * xAI (Grok)

---

## 🔐 GCP Permissions / IAM

To deploy and run this project, the following roles are required:

### For your user (deployment)

* `roles/aiplatform.user` (Vertex AI access)
* `roles/storage.admin` on the staging bucket (e.g. `gs://ai_fnol`)

If you need to create the bucket:

* `roles/storage.admin` at the project level (temporary is fine)

---

### For runtime (Agent Engine)

By default, Vertex uses a managed service agent:

```
service-PROJECT_NUMBER@gcp-sa-aiplatform-re.iam.gserviceaccount.com
```

This works out of the box.

For production use, you should migrate to a **custom service account** with least-privilege access.

---

## 🔧 Installation

### 1. Create virtual environment (outside repo recommended)

```bash
python -m venv ~/multi_llm_venv
source ~/multi_llm_venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Setup

```bash
cp .env.example .env
```

Fill in:

```env
GOOGLE_CLOUD_PROJECT=PROJECT_NAME
GOOGLE_CLOUD_LOCATION=GCP_REGION
GOOGLE_CLOUD_STAGING_BUCKET=gs://GCP_BUCKET

OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
XAI_API_KEY=...
```

---

## 🧪 Local Development

```bash
adk web multi_llm
```

Open:

```
http://127.0.0.1:8000/dev-ui
```

---

## ☁️ Deploy to Vertex AI

```bash
adk deploy agent_engine \
  --project="PROJECT_NAME" \
  --region="GCP_REGION" \
  --display_name="multi_llm" \
  --staging_bucket="gs://GCP_BUCKET" \
  multi_llm
```

---

## 🔍 Test Deployed Agent

```python
import vertexai
from vertexai import agent_engines

vertexai.init(project="PROJECT_NAME", location="GCP_REGION")

agent = agent_engines.get(
    "projects/PROJECT_NAME/locations/GCP_REGION/reasoningEngines/Id"
)

print(agent.query(input="Compare Gemini vs OpenAI vs Claude"))
```

---

## 💰 Usage & Cost Tracking

Each external model call returns:

```text
[OpenAI: 496 tokens | $0.0042]
```

Also available via:

```text
get_usage_summary()
```

---

## ⚠️ Notes

* Gemini is used as the **default model**
* External models are used **selectively**
* Keep `.venv` **outside the repo** to avoid deployment issues

---

## 🗺️ Roadmap

* [ ] Session memory (conversation state)
* [ ] Long-term memory (user preferences)
* [ ] Cost-aware routing
* [ ] RAG (retrieval)
* [ ] Secret Manager integration
* [ ] Frontend / API integrations
* [ ] Migrate deployment and runtime authentication to a dedicated **least-privilege service account** instead of the local user IAM account

---

## 📌 Status

**Active / Iterating toward production-grade system**

---
