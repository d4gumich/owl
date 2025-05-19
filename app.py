# Author: Sara Haptonstall
# Date: 2025-02-06
# AI power Q&A bot that uses 2406 pdf files ReliefWeb (https://apidoc.reliefweb.int/) reports to create a RAG model.




import os
import streamlit as st
import requests
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure Streamlit page
st.set_page_config(page_title="Owl 1.0", page_icon=":owl:")
st.title("Owl Q&A")
st.subheader("_Unlock Insights with AI-Powered Assistance_", divider=True)

# Sidebar settings
st.sidebar.image("owl_logo.jpg")
st.sidebar.header("Settings")
st.sidebar.markdown("[🌐 Visit Data for Good](https://data4good.center/)", unsafe_allow_html=True)

# Model selection and configuration
model_options = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]
selected_model = st.sidebar.selectbox("Select LLM Model for Answer", model_options, index=0)
temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.5, 0.05)
k = st.sidebar.slider("Number of Similar Documents (k)", 1, 10, 5, 1)

# User input
st.write("### Ask a question about nonprofits reports:")
query = st.text_input("Your question", "")
submit = st.button("Submit")

# Load API secrets
SIMILARITY_API_URL = st.secrets["general"].get("SIMILARITY_API")
GOOGLE_API_KEY = st.secrets["general"].get("GOOGLE_API_KEY")

if submit and query.strip():
    if not GOOGLE_API_KEY:
        st.error("❌ Google API key not found. Please set the GOOGLE_API_KEY in your secrets.")
    else:
        ##### Step 1: Call Similarity API #####
        st.subheader("📚 Retrieving Similar Documents")
        with st.spinner("Finding relevant documents..."):
            payload = {"text": query, "k": k}
            try:
                response = requests.post(SIMILARITY_API_URL, json=payload)
                response.raise_for_status()
                similar_docs = response.json().get("results", [])

                # # Debug: Show raw API response
                # st.write("**🔍 Raw API Response:**", similar_docs)

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error calling similarity API: {e}")
                st.stop()

        ##### Step 2: Extract "combined_details" from API response #####
        if similar_docs:
            context_details = "\n\n".join([doc.get("combined_details", "No details available") for doc in similar_docs])

            ##### Step 3: Generate Final Answer Using Gemini #####
            st.subheader("🤖 Generating Final Answer")
            
            # System prompt for Gemini
            system_prompt = (
                "You are a Q&A assistant dedicated to providing accurate, up-to-date information "
                "from ReliefWeb, a humanitarian platform managed by OCHA. Use the provided context documents "
                "to answer the user’s question. If you cannot find the answer or are not sure, say that you do not know. "
                "Keep your answer to ten sentences maximum, be clear and concise. Always end by inviting the user to ask more!"
            )

            # Combine system prompt and context details
            retrieval_prompt = f"{system_prompt}\n\n### Context:\n{context_details}\n\n### User question:\n{query}"

            # Use Gemini to generate the response
            final_llm = ChatGoogleGenerativeAI(model=selected_model, temperature=temperature, api_key=GOOGLE_API_KEY)
            with st.spinner("Creating final answer..."):
                try:
                    final_response = final_llm.invoke(retrieval_prompt)
                    
                    # Extract content from AIMessage object
                    if hasattr(final_response, "content"):
                        final_answer = final_response.content
                    else:
                        final_answer = "⚠️ No response received from Gemini."
            
                    # Display the agent response
                    st.subheader("🧠 Agent Response")
                    st.write(final_answer)
            
                    # # Debug: Show the raw response
                    # st.subheader("🔍 Debugging Raw LLM Response")
                    # st.write(final_response)
            
                    ##### Step 4: Display Retrieved Documents #####
                    st.subheader("📑 Retrieved Documents")
                    for i, doc in enumerate(similar_docs, start=1):
                        st.markdown(f"### **Document {i}**")
                        st.write(f"📌 **Title:** {doc.get('title', 'No title available')}")
                        st.write(f"🔹 **Source:** {doc.get('source', 'Unknown source')}")
                        st.write(f"🔹 **Page:** {doc.get('page_label', 'Unknown source')}")
                        st.write(f"🌍 **URL:** [Click here]({doc.get('URL')})")
                        st.write(f"📝 **Content Preview:** {doc.get('document', 'No details available')[:500]}...")  
            
                except Exception as e:
                    st.error(f"❌ Error generating response with Gemini: {e}")

