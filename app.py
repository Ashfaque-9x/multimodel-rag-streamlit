import streamlit as st
import tempfile
import base64
from multimodel_rag import build_multimodal_rag

st.set_page_config(page_title="Multimodal RAG Chat")
st.title("📄 Multi-Modal RAG by Virtual Techbox")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Initialize session state once
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    # Chat history state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Build RAG only once
    if st.session_state.rag_chain is None:
        with st.spinner("Processing PDF..."):
            st.session_state.rag_chain = build_multimodal_rag(pdf_path)

    # ---- Display previous chat history ----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("images"):
                st.write("### Retrieved Images")
                for img_b64 in msg["images"]:
                    try:
                        image_bytes = base64.b64decode(img_b64)
                        st.image(image_bytes)
                    except Exception:
                        st.warning("Could not render image")

    # Chat input
    user_question = st.chat_input("Ask a question")

    if user_question:

        # Store & show user message
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.chat_message("user").write(user_question)

        # Get response
        response = st.session_state.rag_chain.invoke(user_question)

        answer = response.get("answer", "")
        images = response.get("context", {}).get("images", [])

        # Store assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "images": images
        })

        # Show assistant text answer
        st.chat_message("assistant").write(answer)

        # Show images if any
        if images:
            st.write("### Retrieved Images")
            for img_b64 in images:
                try:
                    image_bytes = base64.b64decode(img_b64)
                    st.image(image_bytes)
                except Exception:
                    st.warning("Could not render image")
