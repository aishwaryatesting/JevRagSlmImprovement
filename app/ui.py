import streamlit as st
from .pipeline import run_pipeline

st.set_page_config(page_title="SLM + Jev RAG Verifier", layout="wide")
st.title("SLM + Jev RAG Verifier")
st.caption("Offline by default. Set VERIFIER_MODE=jev and TYPESAFE_API_KEY for live Jev.")

question = st.text_input(
    "Ask a policy question",
    "Can I return opened headphones if they are not defective?"
)

if st.button("Run verification", type="primary"):
    result = run_pipeline(question)

    st.subheader("Retrieved evidence")
    for item in result["retrieved_evidence"]:
        st.info(item)

    st.subheader("Draft answer")
    st.write(result["draft_answer"])

    st.subheader("Claim checks")
    for check in result["checks"]:
        st.json(check)

    st.subheader("Final answer")
    st.success(result["final_answer"])
