# multimodel_rag.py

import os
import uuid
import base64
import io
from dotenv import load_dotenv
from PIL import Image

from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.stores import InMemoryStore
from langchain_community.vectorstores import FAISS

load_dotenv()


def normalize_image_b64(image_b64: str):
    if not image_b64:
        return None
    try:
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except:
        return None


def build_multimodal_rag(pdf_path):

    chunks = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    texts, tables, images = [], [], []

    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk)
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images.append(el.metadata.image_base64)

        if "Table" in str(type(chunk)):
            tables.append(chunk)

    # ---- Summaries ----
    summarizer_llm = ChatOpenAI(model="gpt-4.1-mini")

    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize the following content concisely:\n{element}"
    )

    summarize_chain = (
        {"element": lambda x: x}
        | summary_prompt
        | summarizer_llm
        | StrOutputParser()
    )

    text_summaries = summarize_chain.batch(texts)

    tables_html = [t.metadata.text_as_html for t in tables]
    table_summaries = summarize_chain.batch(tables_html)

    # ---- Image Summaries ----
    normalized_images = []
    for img in images:
        norm = normalize_image_b64(img)
        if norm:
            normalized_images.append(norm)

    image_prompt = ChatPromptTemplate.from_messages([
        ("user", [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,{image}"}}
        ])
    ])

    image_chain = image_prompt | ChatOpenAI(model="gpt-4.1-mini") | StrOutputParser()
    image_summaries = image_chain.batch(normalized_images)

    # ---- Build Vectorstore ----
    embedding = OpenAIEmbeddings()

    summary_docs = []
    store = InMemoryStore()

    all_summaries = text_summaries + table_summaries + image_summaries
    all_originals = texts + tables_html + normalized_images

    ids = [str(uuid.uuid4()) for _ in all_summaries]

    for i, summary in enumerate(all_summaries):
        summary_docs.append(
            Document(page_content=summary, metadata={"doc_id": ids[i]})
        )

    vectorstore = FAISS.from_documents(summary_docs, embedding)
    store.mset(list(zip(ids, all_originals)))

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="doc_id",
    )

    # ---- RAG Chain ----
    def parse_docs(docs):
        images_b64, texts = [], []
        for d in docs:
            try:
                base64.b64decode(d)
                images_b64.append(d)
            except:
                texts.append(d)
        return {"images": images_b64, "texts": texts}

    def build_prompt(inputs):
        context = inputs["context"]
        question = inputs["question"]

        combined_text = ""
        for t in context["texts"]:
            combined_text += t.text + "\n"

        content = [{
            "type": "text",
            "text": f"Answer using only this context:\n{combined_text}\nQuestion:{question}"
        }]

        for img in context["images"]:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img}"}
            })

        return ChatPromptTemplate.from_messages([("user", content)])

    rag_chain = {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        answer=(
            RunnableLambda(build_prompt)
            | ChatOpenAI(model="gpt-4.1-mini")
            | StrOutputParser()
        )
    )

    return rag_chain
