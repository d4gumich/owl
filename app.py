# Author: Sara Haptonstall
# Date: 2025-02-06
# AI-powered Q&A bot using ReliefWeb reports to create a RAG model.
# Rewritten to use google-generativeai SDK (no LangChain dependency)

import os
import streamlit as st
import requests
import google.generativeai as genai

# ----------------------------
# Configure Streamlit page
# ----------------------------
st.set_page_config(page_title="Owl 1.0", page_icon="🦉")
st.title("Owl Q&A")
st.subheader("_Unlock Insights with AI-Powered Assistance_", divider=True)

# ----------------------------
# Sidebar settings
# ----------------------------
st.sidebar.image("owl_logo.jpg", caption="Owl 1.0")
st.sidebar.header("Settings")
st.sidebar.markdown(
    "[🌐 Visit Data for Good](https://data4good.center/)", unsafe_allow_html=True
)

# ----------------------------
# Model selection and configuration
# ----------------------------
model_options = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
selected_model = st.sidebar.selectbox(
    "Select LLM Model for Answer", model_options, index=0
)
temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.5, 0.05)
k = st.sidebar.slider("Number of Similar Documents (k)", 1, 10, 5, 1)

# ----------------------------
# User input
# ----------------------------
st.write("### Ask a question about nonprofit reports:")
query = st.text_input("Your question", "")
submit = st.button("Submit")

# ----------------------------
# Load API secrets
# ----------------------------
try:
    SIMILARITY_API_URL = st.secrets["general"].get("SIMILARITY_API")
    GOOGLE_API_KEY = st.secrets["general"].get("GOOGLE_API_KEY")
except Exception:
    SIMILARITY_API_URL = os.getenv("SIMILARITY_API")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ Google API key not found. Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# ----------------------------
# Main Logic
# ----------------------------
if submit and query.strip():
    ##### Step 1: Call Similarity API #####
    st.subheader("📚 Retrieving Similar Documents")
    with st.spinner("Finding relevant documents..."):
        payload = {"text": query, "k": k}
        try:
            response = requests.post(SIMILARITY_API_URL, json=payload, timeout=30)
            response.raise_for_status()
            similar_docs = response.json().get("results", [])
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error calling Similarity API: {e}")
            st.stop()

    ##### Step 2: Prepare context from documents #####
    if similar_docs:
        context_details = "\n\n".join(
            [doc.get("combined_details", "No details available") for doc in similar_docs]
        )

        ##### Step 3: Generate Final Answer Using Gemini #####
        st.subheader("🤖 Generating Final Answer")

        system_prompt = (
            "You are a Q&A assistant dedicated to providing accurate, up-to-date information "
            "from ReliefWeb, a humanitarian platform managed by OCHA. Use the provided context documents "
            "to answer the user’s question. If you cannot find the answer or are not sure, say that you do not know. "
            "Keep your answer to ten sentences maximum, be clear and concise. Always end by inviting the user to ask more!"
        )

        retrieval_prompt = (
            f"{system_prompt}\n\n### Context:\n{context_details}\n\n### User question:\n{query}"
        )

        # Create the model
        model = genai.GenerativeModel(selected_model)

        with st.spinner("Creating final answer..."):
            try:
                response = model.generate_content(
                    retrieval_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=1024,
                    ),
                )

                # Extract final answer text
                final_answer = getattr(response, "text", None)
                if not final_answer:
                    # Fallback if response.text is missing
                    parts = []
                    for c in getattr(response, "candidates", []) or []:
                        content = getattr(c, "content", None)
                        for p in getattr(content, "parts", []) or []:
                            txt = getattr(p, "text", "")
                            if txt:
                                parts.append(txt)
                    final_answer = "\n".join(parts).strip() or "⚠️ No response received from Gemini."

                # Display the agent response
                st.subheader("🧠 Agent Response")
                st.write(final_answer)

                ##### Step 4: Display Retrieved Documents #####
                st.subheader("📑 Retrieved Documents")
                for i, doc in enumerate(similar_docs, start=1):
                    st.markdown(f"### **Document {i}**")
                    st.write(f"📌 **Title:** {doc.get('title', 'No title available')}")
                    st.write(f"🔹 **Source:** {doc.get('source', 'Unknown source')}")
                    st.write(f"🔹 **Page:** {doc.get('page_label', 'N/A')}")
                    st.write(f"🌍 **URL:** [Click here]({doc.get('URL')})")
                    preview = doc.get("document", "No details available")[:500]
                    st.write(f"📝 **Content Preview:** {preview}...")

            except Exception as e:
                msg = str(e)
                if "429" in msg or "quota" in msg.lower():
                    st.error(
                        "❌ Gemini returned a quota or rate-limiting error. "
                        "Try using Flash/Lite, reduce k, or check your Google Cloud quota."
                    )
                    st.code(msg, language="text")  # 👈 Show actual error text
                else:
                    st.error("❌ Error generating response with Gemini:")
                    st.code(msg, language="text")  # 👈 Display the real error message for visibility


    else:
        st.warning("⚠️ No similar documents found. Try another query or adjust 'k'.")

