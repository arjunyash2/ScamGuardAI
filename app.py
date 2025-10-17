import os
import pandas as pd
import streamlit as st
from typing import List
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================
# 1. Define Schema
# ============================
class SpamClassification(BaseModel):
    label: str  # "Spam" or "Not Spam"
    risk_score: float
    reasons: List[str]
    red_flags: List[str]
    suggested_action: str


# ============================
# 2. Build Chain
# ============================
@st.cache_resource
def get_chain():
    """Builds and caches the LangChain classification pipeline."""
    parser = PydanticOutputParser(pydantic_object=SpamClassification)
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are ScamGuard, an expert spam detector. Return ONLY a JSON object."),
        ("system", "{format_instructions}"),
        ("human", "Classify this message:\n\"{message}\"")
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0
    )

    chain = prompt | llm | parser
    return chain, format_instructions


# ============================
# 3. Single Classification
# ============================
def classify_message(message: str) -> SpamClassification:
    chain, format_instructions = get_chain()
    return chain.invoke({
        "format_instructions": format_instructions,
        "message": message
    })


# ============================
# 4. Batch Classification
# ============================
def classify_batch(messages: List[str]) -> pd.DataFrame:
    chain, format_instructions = get_chain()

    inputs = [
        {"format_instructions": format_instructions, "message": msg}
        for msg in messages
    ]

    outputs: List[SpamClassification] = chain.batch(inputs)

    results = []
    for i, (msg, out) in enumerate(zip(messages, outputs), start=1):
        results.append({
            "Message #": i,
            "Message": msg,
            "Label": out.label,
            "Risk Score": out.risk_score,
            "Reasons": "; ".join(out.reasons),
            "Red Flags": "; ".join(out.red_flags),
            "Suggested Action": out.suggested_action
        })

    return pd.DataFrame(results)


# ============================
# 5. Streamlit UI
# ============================

def display_result(result):
    st.subheader("🔍 ScamGuard Result")

    # Status Badge
    if result.label.lower() == "spam":
        st.markdown("<h3 style='color:red;'>🚨 Spam Detected</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:green;'>✅ Not Spam</h3>", unsafe_allow_html=True)

    # Risk Score Progress Bar
    st.write("**Risk Score**")
    st.progress(min(max(int(result.risk_score * 100), 0), 100))

    # Details in Expanders
    with st.expander("📌 Reasons"):
        st.write("\n".join([f"- {r}" for r in result.reasons]))

    with st.expander("🚩 Red Flags"):
        st.write("\n".join([f"- {rf}" for rf in result.red_flags]))

    with st.expander("🛠 Suggested Action"):
        st.write(result.suggested_action)


st.set_page_config(page_title="ScamGuard", page_icon="📧", layout="wide")
st.title("📧 ScamGuard - Fraud/Spam Email Classifier")

mode = st.sidebar.radio("Choose Input Mode:", ["Paste Email", "Upload CSV"])

if mode == "Paste Email":
    message = st.text_area("📥 Paste your email content here:", height=200, placeholder="Enter email text...")
    if st.button("🔎 Classify Email"):
        if message.strip():
            with st.spinner("Analyzing email..."):
                result = classify_message(message)
            display_result(result)
        else:
            st.warning("⚠️ Please enter some email text.")

elif mode == "Upload CSV":
    file = st.file_uploader("📂 Upload CSV with a column named 'email'", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if "email" in df.columns:
            with st.spinner("Classifying emails in batch..."):
                results_df = classify_batch(df["email"].tolist())
            st.subheader("📊 Batch Results")
            st.dataframe(results_df)

            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results as CSV",
                data=csv,
                file_name="classified_emails.csv",
                mime="text/csv"
            )
        else:
            st.error("❌ CSV must contain an 'email' column.")
