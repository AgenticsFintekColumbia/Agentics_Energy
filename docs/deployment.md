# Deployment

## 1. 📊 Local Streamlit

See the [Quickstart](quickstart.md) page for local setup instructions.



## 2. 🐳 Docker

### 📦 Build the Image
From the project root:

```bash
docker build -t agenticsenergy-streamlit .
```

### ▶️ Run the Container

```bash
docker run --rm \
  -p 8501:8501 \
  -e GRB_WLSACCESSID="your-wls-id" \
  -e GRB_WLSSECRET="your-wls-secret" \
  -e GRB_LICENSEID="your-license-id" \
  -e GEMINI_API_KEY="your-gemini-api-key" \
  -e GEMINI_MODEL_ID="gemini/gemini-2.0-flash" \
  agenticsenergy-streamlit
```

Then open : 
```bash
http://localhost:8501
```

## 3. 🐙 Github Pages (Docs)

Documentation is built with **MkDocs + Material** and deployed from this repo.

To build docs locally:
```bash
pip install mkdocs-material
mkdocs serve
```

Then visit:
``` text
http://127.0.0.1:8000
```