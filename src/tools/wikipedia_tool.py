import wikipedia

def search_wikipedia(query: str) -> str:
    try:
        summary = wikipedia.summary(query, sentences=3)
        return f"📚 Wikipedia:\n{summary}"
    except Exception as e:
        return f"❌ Wikipedia search failed: {str(e)}"
