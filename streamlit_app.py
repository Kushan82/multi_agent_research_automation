import streamlit as st
import os
import sys

sys.path.append(os.path.abspath("src"))
from workflow.research_flow import research_workflow

st.set_page_config(page_title="ECYPD Research Assistant", layout= "wide")

st.title("🧠 Multi-Agent Research Automation")
st.markdown("Give me a topic and I'll search, analyze, and generate a full research summary using autonomous agents.")

# Input form
with st.form("query_form"):
    query = st.text_input("🔍 Enter your research topic:", "")
    submitted = st.form_submit_button("Run Agents")

# Results placeholder
if submitted and query.strip():
    with st.spinner("Running agents and generating report..."):
        result = research_workflow.invoke({"query": query.strip()})

    st.success("✅ Done!")

    # Display Final Report
    st.subheader("📄 Final Research Report")
    st.write("DEBUG full state:", result)
    st.markdown(result["final_report"])

    # Optional: Expand intermediate outputs
    with st.expander("🔍 Search Agent Output"):
        st.markdown(result["search_output"])

    with st.expander("🧠 Analysis Agent Output"):
        st.markdown(result["analysis_output"])
else:
    st.info("Enter a topic above and click 'Run Agents' to start.")
