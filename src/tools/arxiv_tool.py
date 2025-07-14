from langchain_community.utilities.arxiv import ArxivAPIWrapper

arxiv = ArxivAPIWrapper(load_max_docs=3)

def search_arxiv(query: str) -> str:
    try:
        results = arxiv.run(query)
        return f"📄 Arxiv Results:\n{results}"
    except Exception as e:
        return f"❌ Arxiv failed: {str(e)}"