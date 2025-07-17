import wikipediaapi

def search_wikipedia(query: str) -> str:
    try:
        wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='MultiAgentResearchBot/1.0'
        )
        page = wiki.page(query)
        if not page.exists():
            return f"Wikipedia: No page found for '{query}'"
        summary = page.summary[:500] + "..."  # Trim for brevity
        return f"Wikipedia:\n{summary}"
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"
