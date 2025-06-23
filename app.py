import streamlit as st
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
NUM_IMAGES_PER_ROW = 3

# Import chatbot engine functions
from chatbot import build_vectorstore_from_pdf, generate_insurance_answer

# Initialize session state
if "reset_done" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False
    st.session_state.pdf_uploaded = False
    st.session_state.vector_store = None
    st.session_state.reset_done = True

# Title
st.title("📄🧠 SmartPolicy Chat")

# User Manual
with st.expander("📘 User Manual: Insurance Policy QA Chatbot", expanded=False):
    st.markdown("""
Welcome to your AI Insurance Assistant — a smart tool that helps you understand and compare insurance policies with ease. Just upload your policy PDF, and let the chatbot do the heavy lifting.

---

✅ **What This Assistant Can Do**
- 💬 Answer questions about your policy
- 📝 Summarize the document
- 🆚 Compare with other policies
- 🌐 Fill in gaps via the web
- ⚡ Remembers previous queries

🎯 Example:  
• What’s the waiting period?  
• Is maternity covered?  
• Compare this with my previous policy  
• Summarize this policy  
""")

# Upload PDF
uploaded_pdf = st.file_uploader("📤 Upload your insurance policy (PDF)", type="pdf")

if uploaded_pdf and not st.session_state.pdf_uploaded:
    with st.spinner("🔄 Processing your PDF..."):
        temp_path = os.path.join("temp", uploaded_pdf.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_pdf.read())

        try:
            vector_store = build_vectorstore_from_pdf(temp_path)
            st.session_state.vector_store = vector_store
            st.session_state.pdf_uploaded = True
            st.success("✅ PDF uploaded and processed successfully!")
        except Exception as e:
            st.error(f"❌ Failed to process the PDF: {e}")
            st.stop()

# Display chat history
def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "images" in message:
                for i in range(0, len(message["images"]), NUM_IMAGES_PER_ROW):
                    cols = st.columns(NUM_IMAGES_PER_ROW)
                    for j in range(NUM_IMAGES_PER_ROW):
                        if i + j < len(message["images"]):
                            cols[j].image(message["images"][i + j], width=200)

# Main Chat
if st.session_state.pdf_uploaded:
    if not st.session_state.greetings:
        greeting = "👋 Hi! I'm your Insurance Assistant. Ask me anything about your uploaded policy."
        st.session_state.messages.append({"role": "assistant", "content": greeting})
        st.session_state.greetings = True

    display_chat_messages()

    example_prompts = [
        "Give me an overview of the policy",
        "Show a comparison with other policies",
    ]

    col1, col2 = st.columns(2)
    if col1.button(example_prompts[0]):
        prompt = example_prompts[0]
    elif col2.button(example_prompts[1]):
        prompt = example_prompts[1]
    else:
        prompt = st.chat_input("Ask about your policy...")

    if prompt:
        # Display user input
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate model response
        with st.spinner("🤖 Thinking..."):
            try:
                response = generate_insurance_answer(prompt, st.session_state.vector_store)
            except Exception as e:
                response = f"⚠️ Sorry, an error occurred: {e}"

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("📌 Please upload your policy document to get started.")
