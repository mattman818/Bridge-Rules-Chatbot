"""
Bridge Laws Chatbot - FastAPI Backend
Enhanced for complex, multi-dimensional bridge questions with RAG.
"""

import os
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import anthropic

from document_processor import (
    get_all_chunks,
    search_chunks,
    get_follow_up_suggestions,
    get_law_relationships,
    LawChunk
)

# Load environment variables
load_dotenv()

# Global state
chunks: list[LawChunk] = []
client: Optional[anthropic.Anthropic] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global chunks, client

    # Load document chunks - check same directory first, then parent
    laws_file = Path(__file__).parent / "laws-of-duplicate-bridge.txt"
    if not laws_file.exists():
        laws_file = Path(__file__).parent.parent / "laws-of-duplicate-bridge.txt"
    if not laws_file.exists():
        raise FileNotFoundError(f"Laws file not found")

    print(f"Loading laws from {laws_file}...")
    chunks = get_all_chunks(str(laws_file))
    print(f"Loaded {len(chunks)} document chunks")

    # Initialize Anthropic client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        client = anthropic.Anthropic(api_key=api_key)
        print("Anthropic client initialized")
    else:
        print("WARNING: No ANTHROPIC_API_KEY found. API calls will fail.")

    yield

    # Cleanup (if needed)
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Bridge Laws Chatbot",
    description="AI-powered chatbot for the Laws of Duplicate Bridge 2017 - Enhanced for complex, multi-dimensional questions",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    conversation_history: list[dict] = []


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    sources: list[dict]
    follow_up_suggestions: list[str] = []


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str
    top_k: int = 5


SYSTEM_PROMPT = """You are an expert tournament director for duplicate bridge. Give CONCISE, ACCURATE answers about the Laws of Duplicate Bridge 2017 (ACBL).

## Response Style
- **Be brief**: Get to the point quickly. Players need clear guidance, not essays.
- **Lead with the answer**: State what happens/what to do first, then explain why.
- **Use bullet points** for multiple steps or options.
- **Cite laws precisely** (e.g., "Law 64A") but don't quote lengthy passages.
- **One paragraph per concept** maximum.

## For Complex Situations
When multiple laws interact, briefly explain the chain of events:
1. What's the irregularity?
2. What's the rectification?
3. Any penalties or restrictions?

Don't list every edge case unless asked. Focus on the most likely scenario.

## Key Distinctions to Get Right
- Correctable vs established revokes (Law 62 vs 63)
- Major vs minor penalty cards (Law 50B)
- When UI "demonstrably suggests" an action (Law 16B)
- Comparable calls that avoid penalties (Law 23)

## Terminology
Use "may/should/shall/must" correctly per the Laws' definitions, but don't explain the hierarchy unless relevant.

Here is the relevant context from the Laws:

{context}

Give a clear, practical answer. If the situation is ambiguous, say so briefly and explain the key factor that would determine the ruling."""


def build_context(relevant_chunks: list[LawChunk]) -> str:
    """Build context string from relevant chunks."""
    if not relevant_chunks:
        return "No specific laws found matching this query."

    context_parts = []
    for chunk in relevant_chunks:
        context_parts.append(chunk.get_context_string())

    return "\n\n---\n\n".join(context_parts)


@app.get("/")
async def root():
    """Serve the main HTML page."""
    static_path = Path(__file__).parent / "static" / "index.html"
    if static_path.exists():
        return FileResponse(static_path)
    return {"message": "Bridge Laws Chatbot API", "docs": "/docs"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return a response with follow-up suggestions."""
    global chunks, client

    if not client:
        raise HTTPException(
            status_code=500,
            detail="Anthropic API key not configured. Please set ANTHROPIC_API_KEY environment variable."
        )

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Search for relevant chunks (increased for complex questions)
    relevant_chunks = search_chunks(request.message, chunks, top_k=10)

    # Build context
    context = build_context(relevant_chunks)

    # Build messages for Claude
    messages = []

    # Add conversation history
    for msg in request.conversation_history[-6:]:  # Keep last 6 messages for context
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })

    # Add current message
    messages.append({
        "role": "user",
        "content": request.message
    })

    try:
        # Call Claude API
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,  # Increased for complex answers
            system=SYSTEM_PROMPT.format(context=context),
            messages=messages
        )

        response_text = response.content[0].text

        # Extract sources for citation
        sources = []
        seen = set()
        for chunk in relevant_chunks:
            key = f"{chunk.law_number}-{chunk.section}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "law": chunk.law_number,
                    "title": chunk.title,
                    "section": chunk.section,
                    "category": chunk.category
                })

        # Generate follow-up suggestions
        follow_ups = get_follow_up_suggestions(relevant_chunks, request.message)

        return ChatResponse(
            response=response_text,
            sources=sources,
            follow_up_suggestions=follow_ups
        )

    except anthropic.APIError as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")


@app.post("/api/search")
async def search(request: SearchRequest):
    """Search the laws for relevant sections."""
    global chunks

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    results = search_chunks(request.query, chunks, top_k=request.top_k)

    return {
        "results": [
            {
                "law": chunk.law_number,
                "title": chunk.title,
                "section": chunk.section,
                "category": chunk.category,
                "cross_references": chunk.cross_references,
                "content": chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content
            }
            for chunk in results
        ]
    }


@app.get("/api/laws")
async def list_laws():
    """List all available laws with categories."""
    global chunks

    laws = {}
    for chunk in chunks:
        if chunk.law_number not in laws:
            laws[chunk.law_number] = {
                "number": chunk.law_number,
                "title": chunk.title,
                "category": chunk.category,
                "sections": []
            }
        if chunk.section:
            laws[chunk.law_number]["sections"].append(chunk.section)

    return {"laws": list(laws.values())}


@app.get("/api/law/{law_number}")
async def get_law(law_number: str):
    """Get full content of a specific law."""
    global chunks

    law_chunks = [c for c in chunks if c.law_number == law_number]

    if not law_chunks:
        raise HTTPException(status_code=404, detail=f"Law {law_number} not found")

    return {
        "law_number": law_number,
        "title": law_chunks[0].title,
        "category": law_chunks[0].category,
        "sections": [
            {
                "section": chunk.section,
                "content": chunk.content,
                "cross_references": chunk.cross_references
            }
            for chunk in law_chunks
        ]
    }


@app.get("/api/relationships")
async def get_relationships():
    """Get law relationships and concept mappings for navigation."""
    return get_law_relationships()


@app.get("/api/related/{law_number}")
async def get_related_laws(law_number: str):
    """Get laws related to a specific law."""
    relationships = get_law_relationships()
    related = relationships["related_laws"].get(law_number, [])

    # Get details for related laws
    global chunks
    related_details = []
    for rel_law in related:
        law_chunks = [c for c in chunks if c.law_number == rel_law]
        if law_chunks:
            related_details.append({
                "law_number": rel_law,
                "title": law_chunks[0].title,
                "category": law_chunks[0].category
            })

    return {
        "law_number": law_number,
        "related_laws": related_details
    }


@app.get("/api/concept/{concept}")
async def get_concept_laws(concept: str):
    """Get laws related to a bridge concept."""
    relationships = get_law_relationships()
    concept_lower = concept.lower()

    matching_laws = []
    for key, laws in relationships["concepts"].items():
        if concept_lower in key.lower():
            matching_laws.extend(laws)

    matching_laws = list(set(matching_laws))

    # Get details for matching laws
    global chunks
    law_details = []
    for law_num in matching_laws:
        law_chunks = [c for c in chunks if c.law_number == law_num]
        if law_chunks:
            law_details.append({
                "law_number": law_num,
                "title": law_chunks[0].title,
                "category": law_chunks[0].category
            })

    return {
        "concept": concept,
        "laws": law_details
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "chunks_loaded": len(chunks),
        "api_configured": client is not None,
        "version": "2.0.0"
    }


# Mount static files (if directory exists)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
