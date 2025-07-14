import streamlit as st
import os
import sys
import io

sys.path.insert(0,os.path.abspath("src"))
from workflow.research_flow import research_workflow
from agent.memory_agent import MemoryAgent

if "memory_agent" not in st.session_state:
    st.session_state.memory_agent = MemoryAgent()

st.set_page_config(page_title="ECYPD Research Assistant", layout= "wide")

# Sidebar memory viewer
with st.sidebar:
    st.header("🧠 Memory")
    past = st.session_state.memory_agent.get_all()
    
    if past:
        for entry in past:
            with st.expander(f"📌 {entry['query']} ({entry['timestamp'].split('T')[0]})"):
                st.markdown(entry["final_report"])
    else:
        st.markdown("No memory yet. Run a query to store it here.")
    #debug toggle
    debug_mode = st.checkbox("🔧 Show Debug Logs", value=False)

# Main UI
st.title("🧠 Multi-Agent Research Automation")
st.markdown("Give me a topic and I'll search, analyze, and generate a full research summary using autonomous agents.")



# Input form
with st.form("query_form"):
    query = st.text_input("🔍 Enter your research topic:", "")
    submitted = st.form_submit_button("Run Agents")

# Results placeholder
if submitted and query.strip():
    with st.spinner("Running agents and generating report..."):
        result = research_workflow.invoke({"query": query.strip(),"debug":debug_mode})

    st.success("✅ Done!")

    # Store in memory
    st.session_state.memory_agent.store(query, result["final_report"])

    # Display Final Report
    st.subheader("📄 Final Research Report")
    #st.write("DEBUG full state:", result)
    st.markdown(result["final_report"])

    # Tool Agent Output
    with st.expander("🧰 Tool Agent Output"):
        st.markdown(result.get("tool_output", "No tool data available."))

    st.markdown("#### 📁 Export this report:")
    # Prepare exportable content
    report_text = result["final_report"]
    query_slug = query.strip().replace(" ", "_")[:50]
    # Markdown download
    st.download_button(
        label="📥 Download as .md",
        data=report_text,
        file_name=f"{query_slug}.md",
        mime="text/markdown",)
    # Text download
    st.download_button(
        label="📥 Download as .txt",
        data=report_text,
        file_name=f"{query_slug}.txt",
        mime="text/plain",)


    # Optional: Expand intermediate outputs
    with st.expander("🔍 Search Agent Output"):
        st.markdown(result["search_output"])

    with st.expander("🧠 Analysis Agent Output"):
        st.markdown(result["analysis_output"])

    # 🪵 Debug Section
    if debug_mode:
        st.subheader("🪵 Debug Logs")

        with st.expander("📝 Search Agent Prompt"):
            st.code(result.get("search_debug", {}).get("prompt", "No debug data"))

        with st.expander("📝 Analysis Agent Prompt"):
            st.code(result.get("analysis_debug", {}).get("prompt", "No debug data"))

        with st.expander("📝 Generation Agent Prompt"):
            st.code(result.get("generation_debug", {}).get("prompt", "No debug data"))
        with st.expander("📤 Analysis Input to Generation Agent"):
            debug_input = result.get("analysis_debug", {}).get("input", "")

            if isinstance(debug_input, str):
                st.code(debug_input[:2000])
            else:
                st.warning("⚠️ Could not display analysis input (not a string).")
        with st.expander("📤 Generation Input (Raw Text)"):
            st.code(result.get("generation_debug", {}).get("input", "")[:2000])



else:
    st.info("Enter a topic above and click 'Run Agents' to start.")


