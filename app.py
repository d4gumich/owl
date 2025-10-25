# Author: Sara Haptonstall
# Date: 2025-10-25 - revised to remove lang chanin
# AI power Q&A bot that uses 2406 pdf files ReliefWeb (https://apidoc.reliefweb.int/) reports to create a RAG model.

# Author: Sara Haptonstall
# Owl 1.0 – Streamlit app using google-generativeai SDK (no LangChain)

import os
import json
import time
import requests
import streamlit as st
import google.generativeai as genai

# ----------------------------
# Page config & Header
# ----------------------------
st.set_page_config(page_title="Owl 1.0", page_icon="🦉", layout="wide")
st.title("Owl Q&A")
st.subheader("_Unlock Insights with AI-Powered Assistance_", divider=True)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.image("owl_logo.jpg", caption="Owl 1.0", use_column_width=True)
st.sidebar.header("Settings")
st.sidebar.markdown("[🌐 Visit Data for Good](https://data4good.center/)", unsafe_allow_html=True)

MODEL_OPTIONS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.0-flash-lite",
]
selected_model = st.sidebar.selectbox("LLM model", MODEL_OPTIONS, index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5, 0.05)
k = st.sidebar.slider("Similar documents (k)", 1, 10, 5, 1)
max_context_chars = st.sidebar.number_input("Max context size (chars)", 2000, 50000, 12000, 1000)
debug = st.sidebar.toggle("Show debug info", value=False)

# ----------------------------
# Secrets / Env
# ----------------------------
def _get_secret(group: str, key: str, env_fallback: str | None = None) -> str | None:
    try:
        if group in st.secrets:
            v = st.secrets[group].get(key)
            if v:
                return str(v)
    except Exception:
        pass
    return os.getenv(env_fallback) if env_fallback else None

SIMILARITY_API_URL = _get_secret("general", "SIMILARITY_API", "SIMILARITY_API") 
GOOGLE_API_KEY = _get_secret("general", "GOOGLE_API_KEY", "GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Add it to Streamlit secrets or an environment variable.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# ----------------------------
# System Prompt
# ----------------------------
SYSTEM_PROMPT = (
    "You are a Q&A assistant dedicated to providing accurate, up-to-date information "
    "from ReliefWeb, a humanitarian platform managed by OCHA. Use the provided context documents "
    "to answer the user’s question. If you cannot find the answer or are not sure, say that you do not know. "
    "Keep your answer to ten sentences maximum, be clear and concise. Always end by inviting the user to ask more!"
)

# ----------------------------
# Input
# ----------------------------
st.write("### Ask a question about nonprofit/humanitarian reports:")
query = st.text_input("Your question", "")
submit = st.button("Submit")

# ----------------------------
# Helpers
# ----------------------------
def trim_context(text: str, limit_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= limit_chars:
        return text
    return text[:limit_chars] + "\n\n[...context truncated to fit model limits...]"

def build_prompt(system_prompt: str, context: str, question: str) -> str:
    return f"{system_prompt}\n\n### Context:\n{context}\n\n### User Question:\n{question}"

def _requests_error_msg(e: requests.exceptions.RequestException) -> str:
    if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
        status = e.response.status_code
        try:
            body = e.response.json()
        except Exception:
            body = e.response.text
        if status == 429:
            return f"Rate limit / quota exceeded (HTTP 429). Details: {body}"
        if status >= 500:
            return f"Server error (HTTP {status}). Details: {body}"
        return f"HTTP {status} error. Details: {body}"
    if isinstance(e, requests.exceptions.ConnectTimeout):
        return "Connection timed out contacting the Similarity API."
    if isinstance(e, requests.exceptions.ReadTimeout):
        return "Similarity API took too long to respond."
    if isinstance(e, requests.exceptions.ConnectionError):
        return "Network error reaching the Similarity API."
    return f"Request error: {str(e)}"

@st.cache_data(show_spinner=False, ttl=300)
def fetch_similar(api_url: str, text: str, top_k: int):
    payload = {"text": text, "k": int(top_k)}
    backoffs = [0.5, 1.0, 2.0]
    for i, pause in enumerate(backoffs):
        try:
            r = requests.post(api_url, json=payload, timeout=30)
            r.raise_for_status()
            return {"ok": True, "json": r.json()}
        except requests.exceptions.RequestException as e:
            if i < len(backoffs) - 1:
                time.sleep(pause)
            else:
                return {"ok": False, "error": _requests_error_msg(e)}
    return {"ok": False, "error": "Unknown error"}

# ----------------------------
# Main
# ----------------------------
if submit:
    if not query.strip():
        st.warning("Please enter a question to continue.")
        st.stop()

    st.subheader("📚 Retrieving Similar Documents")
    with st.spinner("Finding relevant documents..."):
        out = fetch_similar(SIMILARITY_API_URL, query, k)
        if not out.get("ok"):
            st.error(f"❌ Error calling Similarity API: {out.get('error', 'Unknown error')}")
            if debug:
                st.code(json.dumps(out, indent=2))
            st.stop()

        similar_docs = out["json"].get("results", []) or []
        if debug:
            st.caption("Raw Similarity API response (truncated):")
            st.code(json.dumps(out, indent=2)[:4000])

    if not similar_docs:
        st.info("No similar documents were retrieved. Try rephrasing your question or increasing **k**.")
        st.stop()

    # Build/trim context
    context_details = "\n\n".join([d.get("combined_details") or d.get("document", "") for d in similar_docs])
    context_details = trim_context(context_details, max_context_chars)

    # Generate answer with Gemini
    st.subheader("🤖 Generating Final Answer")
    with st.spinner("Creating final answer..."):
        try:
            model = genai.GenerativeModel(selected_model)
            full_prompt = build_prompt(SYSTEM_PROMPT, context_details, query)
            resp = model.generate_content(full_prompt)

            # Prefer resp.text; fallback to candidates/parts if needed
            final_answer = (getattr(resp, "text", "") or "").strip()
            if not final_answer:
                parts = []
                for c in getattr(resp, "candidates", []) or []:
                    content = getattr(c, "content", None)
                    for p in getattr(content, "parts", []) or []:
                        txt = getattr(p, "text", "")
                        if txt:
                            parts.append(txt)
                final_answer = "\n".join(parts).strip() or "⚠️ No response received from Gemini."

            st.subheader("🧠 Agent Response")
            st.write(final_answer)

            if debug:
                st.caption("Raw Gemini object (repr, truncated):")
                st.code(repr(resp)[:1500])

        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower():
                st.error("❌ Gemini returned a quota/rate-limiting error. Try Flash/Lite, reduce k/context, or check quota.")
            else:
                st.error(f"❌ Error generating response with Gemini: {e}")
            st.stop()

    # Retrieved docs display
    st.subheader("📑 Retrieved Documents")
    for i, doc in enumerate(similar_docs, 1):
        with st.expander(f"Document {i}: {doc.get('title', 'Untitled')}", expanded=(i == 1)):
            left, right = st.columns([2.2, 1])
            with left:
                st.write(f"**Source:** {doc.get('source', 'Unknown')}")
                st.write(f"**Page:** {doc.get('page_label', '—')}")
                url = doc.get("URL") or doc.get("url") or ""
                if url:
                    st.write(f"**URL:** {url}")
                preview = (doc.get("document") or doc.get("combined_details") or "")[:800]
                st.write(f"**Content Preview:**\n\n{preview}…")
            with right:
                st.code((doc.get("combined_details") or doc.get("document") or "")[:1800], language="markdown")


