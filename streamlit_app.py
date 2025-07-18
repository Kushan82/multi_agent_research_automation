import streamlit as st
import os
import sys
import io
import tempfile
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0,os.path.abspath("src"))
from workflow.research_flow import research_workflow
from agent.memory_agent import MemoryAgent
from agent.rag_agent import RAGAgent

# Initialize session state
if "memory_agent" not in st.session_state:
    st.session_state.memory_agent = MemoryAgent()

if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = RAGAgent()

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

st.set_page_config(page_title="ECYPD Research Assistant", layout="wide")

# Sidebar for document management and memory
with st.sidebar:
    st.header("📚 Document Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents for RAG",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx']
    )
    
    if uploaded_files:
        if st.button("📥 Ingest Documents"):
            with st.spinner("Processing documents..."):
                temp_files = []
                try:
                    # Save uploaded files temporarily
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_files.append(tmp_file.name)
                    
                    # Ingest documents
                    result = st.session_state.rag_agent.ingest_documents(temp_files)
                    
                    if result["success"]:
                        st.success(f"✅ Ingested {result['total_chunks']} chunks from {len(result['processed_files'])} files")
                        st.session_state.ingested_files.extend([f.name for f in uploaded_files])
                    else:
                        st.error(f"❌ Failed to ingest documents: {result.get('error', 'Unknown error')}")
                        
                finally:
                    # Clean up temp files
                    for temp_file in temp_files:
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
    
    # Show ingested files
    if st.session_state.ingested_files:
        st.subheader("📄 Ingested Files")
        for file in st.session_state.ingested_files:
            st.text(f"• {file}")
    
    # Vector store stats
    if st.button("📊 Vector Store Stats"):
        stats = st.session_state.rag_agent.get_vector_store_stats()
        st.json(stats)
    
    # URL ingestion
    st.subheader("🌐 URL Ingestion")
    url_input = st.text_input("Enter URL to ingest:")
    if st.button("📥 Ingest URL") and url_input:
        with st.spinner("Processing URL..."):
            result = st.session_state.rag_agent.ingest_urls([url_input])
            if result["success"]:
                st.success(f"✅ Ingested {result['total_chunks']} chunks from URL")
            else:
                st.error(f"❌ Failed to ingest URL: {result.get('error', 'Unknown error')}")
    
    st.divider()
    
    # Memory section
    st.header("🧠 Memory")
    past = st.session_state.memory_agent.get_all()
    
    if past:
        for entry in past:
            with st.expander(f"📌 {entry['query']} ({entry['timestamp'].split('T')[0]})"):
                st.markdown(entry["final_report"])
    else:
        st.markdown("No memory yet. Run a query to store it here.")
    
    # Debug toggle
    debug_mode = st.checkbox("🔧 Show Debug Logs", value=False)

# Main UI
st.title("🧠 Multi-Agent Research Automation")
st.markdown("Give me a topic and I'll search, analyze, and generate a full research summary using autonomous agents.")

# RAG-only mode toggle
rag_only = st.checkbox("🔍 RAG-Only Mode (Search only uploaded documents)", value=False)

# Input form
with st.form("query_form"):
    query = st.text_input("🔍 Enter your research topic:", "")
    submitted = st.form_submit_button("Run Agents")

# Results placeholder
if submitted and query.strip():
    if rag_only:
        # RAG-only mode
        with st.spinner("Querying your documents..."):
            result = st.session_state.rag_agent.query_with_rag(query.strip(), debug=debug_mode)
        
        st.success("✅ Done!")
        
        # Display RAG response
        st.subheader("📄 RAG Response")
        st.markdown(result["output"])
        
        # Show context info
        if result.get("context_used"):
            st.success(f"✅ Found relevant context ({result['context_length']} characters)")
        else:
            st.warning("⚠️ No relevant documents found - showing fallback response")
        
        # Debug info
        if debug_mode and result.get("debug"):
            with st.expander("🔍 RAG Debug Info"):
                st.json(result["debug"])
    
    else:
        # Full workflow mode
        with st.spinner("Running agents and generating report..."):
            result = research_workflow.invoke({"query": query.strip(), "debug": debug_mode})

        st.success("✅ Done!")

        # Store in memory
        st.session_state.memory_agent.store(query, result["final_report"])

        # Display Final Report
        st.subheader("📄 Final Research Report")
        st.markdown(result["final_report"])

        # RAG Output (if available)
        if result.get("rag_output"):
            with st.expander("🧠 RAG Agent Output"):
                st.markdown(result["rag_output"])

        # Tool Agent Output
        with st.expander("🧰 Tool Agent Output"):
            st.markdown(result.get("tool_output", "No tool data available."))

        st.markdown("#### 📁 Export this report:")
        # Prepare exportable content
        report_text = result["final_report"]
        query_slug = query.strip().replace(" ", "_")[:50]
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download as .md",
                data=report_text,
                file_name=f"{query_slug}.md",
                mime="text/markdown"
            )
        with col2:
            st.download_button(
                label="📥 Download as .txt",
                data=report_text,
                file_name=f"{query_slug}.txt",
                mime="text/plain"
            )

        # Intermediate outputs
        with st.expander("🔍 Search Agent Output"):
            st.markdown(result["search_output"])

        with st.expander("🧠 Analysis Agent Output"):
            st.markdown(result["analysis_output"])

        # Debug Section
        if debug_mode:
            st.subheader("🪵 Debug Logs")

            with st.expander("📝 Search Agent Prompt"):
                st.code(result.get("search_debug", {}).get("prompt", "No debug data"))

            with st.expander("📝 Analysis Agent Prompt"):
                st.code(result.get("analysis_debug", {}).get("prompt", "No debug data"))

            with st.expander("📝 Generation Agent Prompt"):
                st.code(result.get("generation_debug", {}).get("prompt", "No debug data"))
                
            if result.get("rag_debug"):
                with st.expander("🧠 RAG Debug Info"):
                    st.json(result["rag_debug"])

else:
    st.info("Enter a topic above and click 'Run Agents' to start.")
    if not st.session_state.ingested_files:
        st.info("💡 Tip: Upload documents in the sidebar to enable RAG functionality!")