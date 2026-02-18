# Deploy NANIN app on Streamlit (without touching existing app)

This repo already has a working root app.  
Use this new entrypoint instead:

- `Nanin/streamlit_nanin_app.py`

## 1) Local run

```powershell
cd C:\Users\Usuario\OneDrive\Desktop\Codex\Nanin
streamlit run streamlit_nanin_app.py
```

## 2) API key options

Option A (UI input each run):
- Paste key in sidebar field.

Option B (recommended for deployment):
- In Streamlit Cloud app settings, set secret:
  - `OPENAI_API_KEY = "sk-..."`

Local secrets file (optional):
- `.streamlit/secrets.toml`
```toml
OPENAI_API_KEY = "sk-..."
```

## 3) Streamlit Cloud deploy

When creating the app in Streamlit:
1. Repository: `nanosep/Codex`
2. Branch: `main`
3. Main file path: `Nanin/streamlit_nanin_app.py`
4. Add secret `OPENAI_API_KEY` in app settings.

This avoids changing the existing root `streamlit_app.py`.
