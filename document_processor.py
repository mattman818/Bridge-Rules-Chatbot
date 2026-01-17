"""
Document processor for the Laws of Duplicate Bridge.
Enhanced with semantic chunking, cross-referencing, and multi-dimensional query support.
"""

import re
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class LawChunk:
    """A chunk of the laws document with enhanced metadata."""
    law_number: str
    title: str
    section: str
    content: str
    cross_references: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    category: str = ""

    def to_dict(self):
        return {
            "law_number": self.law_number,
            "title": self.title,
            "section": self.section,
            "content": self.content,
            "cross_references": self.cross_references,
            "keywords": self.keywords,
            "category": self.category
        }

    def get_context_string(self, max_chars: int = 800) -> str:
        """Return a formatted string for use as context, truncated if needed."""
        header = f"LAW {self.law_number}: {self.title}"
        if self.section:
            header += f" - {self.section}"

        # Truncate content if too long
        content = self.content
        if len(content) > max_chars:
            content = content[:max_chars] + "..."

        refs = ""
        if self.cross_references:
            refs = f"\n[See also: Law {', '.join(self.cross_references[:3])}]"

        return f"{header}\n{content}{refs}"


# Law categories for semantic grouping
LAW_CATEGORIES = {
    "setup": ["1", "2", "3", "4", "5", "6", "7", "8"],
    "general_procedure": ["9", "10", "11", "12", "13", "14", "15", "16"],
    "auction": ["17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40"],
    "play": ["41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60"],
    "revoke": ["61", "62", "63", "64"],
    "tricks_and_claims": ["65", "66", "67", "68", "69", "70", "71"],
    "proprieties": ["72", "73", "74", "75", "76"],
    "scoring": ["77", "78", "79"],
    "tournament": ["80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93"],
}

# Related law mappings - laws that commonly interact
RELATED_LAWS = {
    # Revoke-related laws
    "61": ["62", "63", "64", "44", "47"],
    "62": ["61", "63", "64", "47"],
    "63": ["61", "62", "64"],
    "64": ["61", "62", "63", "12"],

    # Penalty card related
    "49": ["50", "51", "52", "24"],
    "50": ["49", "51", "52", "56", "57"],
    "51": ["49", "50", "52"],
    "52": ["49", "50", "51"],

    # Insufficient bid related
    "27": ["23", "26", "16"],
    "23": ["27", "25", "26"],

    # Call out of rotation
    "28": ["29", "30", "31", "32"],
    "29": ["28", "30", "31", "32", "26"],
    "30": ["28", "29", "31", "32", "26"],
    "31": ["28", "29", "30", "32", "26"],
    "32": ["28", "29", "30", "31", "26"],

    # Lead out of turn
    "53": ["54", "55", "56", "50"],
    "54": ["53", "55", "56", "50", "48"],
    "55": ["53", "54", "56"],
    "56": ["53", "54", "55", "50"],

    # Claims and concessions
    "68": ["69", "70", "71"],
    "69": ["68", "70", "71"],
    "70": ["68", "69", "71"],
    "71": ["68", "69", "70"],

    # Unauthorized information
    "16": ["73", "75", "12"],
    "73": ["16", "74", "75"],
    "75": ["16", "20", "21", "73"],

    # Director's powers
    "12": ["10", "11", "81", "82", "84", "85"],
    "81": ["12", "82", "83", "84"],

    # Dummy related
    "42": ["43", "45", "46"],
    "43": ["42", "45", "9"],
}

# Concept mappings - bridge concepts to relevant laws
CONCEPT_TO_LAWS = {
    "revoke": ["61", "62", "63", "64"],
    "revoking": ["61", "62", "63", "64"],
    "renege": ["61", "62", "63", "64"],
    "failure to follow suit": ["61", "62", "63", "64"],

    "penalty card": ["49", "50", "51", "52"],
    "exposed card": ["24", "48", "49", "50"],
    "faced card": ["48", "49", "50"],

    "insufficient bid": ["27"],
    "illegal bid": ["27", "35", "36", "37", "38"],
    "out of rotation": ["28", "29", "30", "31", "32"],
    "bid out of turn": ["31"],
    "pass out of turn": ["30"],
    "double out of turn": ["32"],

    "lead out of turn": ["53", "54", "55", "56"],
    "opening lead": ["41", "47", "54"],
    "premature lead": ["57"],
    "premature play": ["57"],

    "claim": ["68", "69", "70"],
    "concession": ["68", "69", "70", "71"],
    "contested claim": ["70"],

    "unauthorized information": ["16", "73"],
    "ui": ["16", "73"],
    "hesitation": ["16", "73"],
    "slow pass": ["16", "73"],
    "tempo": ["73"],
    "break in tempo": ["16", "73"],

    "misinformation": ["20", "21", "75"],
    "wrong explanation": ["20", "21", "75"],
    "mistaken explanation": ["75"],
    "mistaken call": ["75"],
    "alert": ["20", "40", "75"],

    "dummy": ["42", "43", "45", "46"],
    "dummy's rights": ["42"],
    "dummy's limitations": ["43"],

    "director": ["9", "10", "11", "12", "81", "82", "83", "84", "85"],
    "appeal": ["83", "92", "93"],
    "adjusted score": ["12"],

    "comparable call": ["23"],
    "withdrawn call": ["25", "26"],
    "unintended call": ["25"],

    "tricks": ["65", "66", "67", "79"],
    "defective trick": ["67"],
    "inspection": ["66"],

    "scoring": ["77", "78", "79"],
    "vulnerability": ["77"],
    "matchpoint": ["78"],
    "imp": ["78"],

    "psychic": ["40", "73"],
    "psych": ["40", "73"],
}

# Question type indicators for multi-dimensional analysis
MULTI_DIMENSIONAL_INDICATORS = [
    "what happens", "what are all", "what if", "and then",
    "after that", "consequences", "options", "rights",
    "both", "multiple", "sequence", "chain",
    "interact", "combined", "together", "related"
]


def load_document(file_path: str) -> str:
    """Load the document from file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def clean_text(text: str) -> str:
    """Clean up extracted text."""
    # Remove page numbers and roman numerals
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[ivxlc]+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Fix common OCR/extraction issues
    text = text.replace('', '♠')
    text = text.replace('', '♥')
    text = text.replace('', '♦')
    text = text.replace('', '♣')

    # Remove excessive whitespace while preserving paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)

    return text.strip()


def extract_cross_references(content: str) -> list[str]:
    """Extract references to other laws from content."""
    # Pattern for "Law X", "Laws X and Y", "Law XY", etc.
    refs = set()

    # Single law references
    single_refs = re.findall(r'\bLaw\s+(\d+[A-Z]?\d*)\b', content, re.IGNORECASE)
    refs.update(single_refs)

    # Multiple law references like "Laws 61, 62, 63"
    multi_refs = re.findall(r'\bLaws?\s+([\d,\s]+(?:and\s+)?\d+)', content, re.IGNORECASE)
    for ref_group in multi_refs:
        numbers = re.findall(r'\d+', ref_group)
        refs.update(numbers)

    # Section references like "see A above" or "see Law 12C2"
    section_refs = re.findall(r'\bLaw\s+(\d+[A-Z]\d*)\b', content, re.IGNORECASE)
    refs.update(section_refs)

    return sorted(list(refs), key=lambda x: (int(re.match(r'\d+', x).group()), x))


def extract_keywords(content: str, title: str) -> list[str]:
    """Extract important bridge keywords from content."""
    keywords = []

    # Important bridge terms to look for
    bridge_terms = [
        'revoke', 'penalty', 'card', 'lead', 'bid', 'auction', 'play',
        'declarer', 'defender', 'dummy', 'trump', 'claim', 'concession',
        'irregularity', 'director', 'rectification', 'insufficient',
        'hesitation', 'unauthorized', 'alert', 'explanation', 'double',
        'redouble', 'pass', 'exposed', 'faced', 'turn', 'rotation',
        'misinformation', 'psychic', 'convention', 'agreement',
        'score', 'trick', 'contract', 'vulnerable', 'slam', 'game',
        'partscore', 'overtrick', 'undertrick', 'opening', 'suit',
        'notrump', 'major', 'minor', 'honor', 'establish', 'correction'
    ]

    content_lower = content.lower()
    title_lower = title.lower()

    for term in bridge_terms:
        if term in content_lower or term in title_lower:
            keywords.append(term)

    return keywords


def get_category_for_law(law_number: str) -> str:
    """Get the category for a law number."""
    base_law = re.match(r'\d+', law_number).group()

    for category, laws in LAW_CATEGORIES.items():
        if base_law in laws:
            return category
    return "general"


def parse_laws(text: str) -> list[LawChunk]:
    """Parse the laws document into chunks by law and section."""
    chunks = []

    # First, extract the definitions section
    definitions_match = re.search(
        r'Definitions\s*\n(.*?)(?=LAW\s+1\b)',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if definitions_match:
        definitions_content = clean_text(definitions_match.group(1))
        if definitions_content:
            chunks.append(LawChunk(
                law_number="DEFINITIONS",
                title="Definitions",
                section="",
                content=definitions_content,
                cross_references=[],
                keywords=["definition", "terminology", "meaning"],
                category="definitions"
            ))

    # Pattern to match law headers
    law_pattern = re.compile(
        r'LAW\s+(\d+)\s*[-–—]?\s*\n?\s*([A-Z][A-Z\s,\'\-&]+?)(?=\n)',
        re.MULTILINE
    )

    # Find all law positions
    law_matches = list(law_pattern.finditer(text))

    for i, match in enumerate(law_matches):
        law_number = match.group(1)
        law_title = match.group(2).strip()

        # Clean up title
        law_title = re.sub(r'\s+', ' ', law_title).strip()

        # Get content until next law
        start_pos = match.end()
        if i + 1 < len(law_matches):
            end_pos = law_matches[i + 1].start()
        else:
            end_pos = len(text)

        law_content = text[start_pos:end_pos]
        category = get_category_for_law(law_number)

        # Try to split by sections (A., B., C., etc.)
        section_pattern = re.compile(
            r'^([A-Z])\.\s+([^\n]+)',
            re.MULTILINE
        )
        section_matches = list(section_pattern.finditer(law_content))

        if section_matches:
            # Create a chunk for any content before the first section
            pre_section = law_content[:section_matches[0].start()].strip()
            if pre_section:
                pre_section_clean = clean_text(pre_section)
                chunks.append(LawChunk(
                    law_number=law_number,
                    title=law_title,
                    section="Introduction",
                    content=pre_section_clean,
                    cross_references=extract_cross_references(pre_section_clean),
                    keywords=extract_keywords(pre_section_clean, law_title),
                    category=category
                ))

            # Create chunks for each section
            for j, sec_match in enumerate(section_matches):
                section_letter = sec_match.group(1)
                section_title = sec_match.group(2).strip()

                sec_start = sec_match.start()
                if j + 1 < len(section_matches):
                    sec_end = section_matches[j + 1].start()
                else:
                    sec_end = len(law_content)

                section_content = law_content[sec_start:sec_end]
                section_content = clean_text(section_content)

                if section_content:
                    chunks.append(LawChunk(
                        law_number=law_number,
                        title=law_title,
                        section=f"{section_letter}. {section_title}",
                        content=section_content,
                        cross_references=extract_cross_references(section_content),
                        keywords=extract_keywords(section_content, law_title),
                        category=category
                    ))
        else:
            # No sections found, use the whole law as one chunk
            law_content = clean_text(law_content)
            if law_content:
                chunks.append(LawChunk(
                    law_number=law_number,
                    title=law_title,
                    section="",
                    content=law_content,
                    cross_references=extract_cross_references(law_content),
                    keywords=extract_keywords(law_content, law_title),
                    category=category
                ))

    return chunks


def decompose_query(query: str) -> list[str]:
    """
    Decompose a complex query into sub-queries for better retrieval.
    This helps with multi-dimensional questions.
    """
    sub_queries = [query]  # Always include the original
    query_lower = query.lower()

    # Check if this is a multi-dimensional question
    is_complex = any(indicator in query_lower for indicator in MULTI_DIMENSIONAL_INDICATORS)

    if is_complex:
        # Look for concepts mentioned and create sub-queries
        for concept, laws in CONCEPT_TO_LAWS.items():
            if concept in query_lower:
                # Add a query specifically about this concept
                sub_queries.append(f"Law {' '.join(laws)} {concept}")

        # Handle "and" conjunctions
        if " and " in query_lower:
            parts = query_lower.split(" and ")
            for part in parts:
                if len(part.strip()) > 10:
                    sub_queries.append(part.strip())

        # Handle "then" for sequence questions
        if " then " in query_lower or "after" in query_lower:
            # Look for sequential concepts
            if "revoke" in query_lower:
                sub_queries.append("revoke correction establishment")
                sub_queries.append("revoke penalty trick adjustment")
            if "penalty card" in query_lower:
                sub_queries.append("penalty card lead options")
                sub_queries.append("penalty card disposal")

    return list(set(sub_queries))


def get_related_laws_for_chunk(chunk: LawChunk) -> list[str]:
    """Get laws that are related to this chunk."""
    related = set()

    # Add explicitly cross-referenced laws
    for ref in chunk.cross_references:
        base_law = re.match(r'\d+', ref)
        if base_law:
            related.add(base_law.group())

    # Add laws from the related laws mapping
    base_law = re.match(r'\d+', chunk.law_number)
    if base_law and base_law.group() in RELATED_LAWS:
        related.update(RELATED_LAWS[base_law.group()])

    return sorted(list(related), key=lambda x: int(x))


def search_chunks(query: str, chunks: list[LawChunk], top_k: int = 8) -> list[LawChunk]:
    """
    Search for relevant chunks using enhanced multi-query search.
    Supports complex, multi-dimensional questions.
    """
    query_lower = query.lower()

    # Decompose query for better coverage
    sub_queries = decompose_query(query)

    # Aggregate scores across all sub-queries
    chunk_scores = defaultdict(float)

    for sub_query in sub_queries:
        sub_query_lower = sub_query.lower()
        query_words = set(re.findall(r'\b[a-z]{3,}\b', sub_query_lower))

        # Look for specific law references
        law_refs = re.findall(r'law\s*(\d+)', sub_query_lower)

        # Look for concept matches
        concept_laws = set()
        for concept, laws in CONCEPT_TO_LAWS.items():
            if concept in sub_query_lower:
                concept_laws.update(laws)

        for i, chunk in enumerate(chunks):
            score = 0
            text = f"{chunk.law_number} {chunk.title} {chunk.section} {chunk.content}".lower()

            # Exact law reference match (highest priority)
            for ref in law_refs:
                if chunk.law_number == ref:
                    score += 150
                elif ref in chunk.cross_references:
                    score += 30

            # Concept-based law match
            base_law = re.match(r'\d+', chunk.law_number)
            if base_law and base_law.group() in concept_laws:
                score += 80

            # Keyword matching with position weighting
            for word in query_words:
                if word in text:
                    count = text.count(word)
                    # Boost for title matches
                    if word in chunk.title.lower():
                        score += 15
                    # Boost for section title matches
                    if word in chunk.section.lower():
                        score += 10
                    # Content matches
                    score += min(count, 5)

            # Boost for keyword matches in chunk metadata
            for keyword in chunk.keywords:
                if keyword in sub_query_lower:
                    score += 8

            # Important bridge terms boost
            important_terms = [
                'revoke', 'penalty', 'card', 'lead', 'bid', 'auction', 'play',
                'declarer', 'defender', 'dummy', 'trump', 'claim', 'concession',
                'irregularity', 'director', 'rectification', 'insufficient',
                'hesitation', 'unauthorized', 'alert', 'explanation', 'double',
                'redouble', 'pass', 'exposed', 'faced', 'turn', 'rotation'
            ]
            for term in important_terms:
                if term in sub_query_lower and term in text:
                    score += 5

            chunk_scores[i] += score

    # Convert to sorted list
    scored_chunks = [(score, i) for i, score in chunk_scores.items()]
    scored_chunks.sort(reverse=True)

    # Select top chunks while ensuring diversity
    results = []
    seen_laws = defaultdict(int)
    related_laws_to_include = set()

    # First pass: get primary matches and collect related laws
    for score, idx in scored_chunks:
        if score > 0:
            chunk = chunks[idx]
            base_law = re.match(r'\d+', chunk.law_number)
            if base_law:
                related_laws_to_include.update(get_related_laws_for_chunk(chunk))

    # Second pass: select chunks with diversity control
    for score, idx in scored_chunks:
        if score > 0:
            chunk = chunks[idx]
            base_law = re.match(r'\d+', chunk.law_number)
            law_key = base_law.group() if base_law else chunk.law_number

            # Allow up to 3 chunks per law, but include related laws
            if seen_laws[law_key] < 3 or law_key in related_laws_to_include:
                seen_laws[law_key] += 1
                results.append(chunk)

            if len(results) >= top_k:
                break

    # If we have room and this is a complex query, add related law chunks
    if len(results) < top_k and len(sub_queries) > 1:
        for score, idx in scored_chunks:
            if chunks[idx] not in results:
                chunk = chunks[idx]
                base_law = re.match(r'\d+', chunk.law_number)
                if base_law and base_law.group() in related_laws_to_include:
                    results.append(chunk)
                    if len(results) >= top_k:
                        break

    return results


def get_follow_up_suggestions(chunks: list[LawChunk], original_query: str) -> list[str]:
    """Generate follow-up question suggestions based on retrieved chunks."""
    suggestions = []
    query_lower = original_query.lower()

    # Get all related laws from the chunks
    related_laws = set()
    for chunk in chunks:
        related_laws.update(get_related_laws_for_chunk(chunk))

    # Generate suggestions based on related concepts
    if any(l in ["61", "62", "63", "64"] for l in related_laws):
        if "penalty" not in query_lower:
            suggestions.append("What are the penalty provisions for an established revoke?")
        if "correct" not in query_lower:
            suggestions.append("When can a revoke be corrected vs when is it established?")

    if any(l in ["49", "50", "51", "52"] for l in related_laws):
        if "lead" not in query_lower:
            suggestions.append("What lead restrictions apply when there's a penalty card?")
        if "multiple" not in query_lower:
            suggestions.append("What happens if a defender has multiple penalty cards?")

    if any(l in ["16", "73"] for l in related_laws):
        if "demonstrably" not in query_lower:
            suggestions.append("What does 'demonstrably suggested' mean for unauthorized information?")

    if any(l in ["68", "69", "70"] for l in related_laws):
        if "contested" not in query_lower:
            suggestions.append("What happens if the opponents don't agree with a claim?")

    if any(l in ["27", "23"] for l in related_laws):
        if "comparable" not in query_lower:
            suggestions.append("What is a comparable call and when does it apply?")

    return suggestions[:3]  # Return top 3 suggestions


def get_all_chunks(file_path: str) -> list[LawChunk]:
    """Load document and return all chunks."""
    text = load_document(file_path)
    return parse_laws(text)


def get_law_relationships() -> dict:
    """Return the law relationships for the frontend."""
    return {
        "related_laws": RELATED_LAWS,
        "categories": LAW_CATEGORIES,
        "concepts": CONCEPT_TO_LAWS
    }


if __name__ == "__main__":
    # Test the document processor
    import sys

    # Fix encoding for Windows console
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    file_path = sys.argv[1] if len(sys.argv) > 1 else "../laws-of-duplicate-bridge.txt"

    print(f"Loading document from {file_path}...")
    chunks = get_all_chunks(file_path)

    print(f"\nFound {len(chunks)} chunks:")
    for chunk in chunks[:10]:
        print(f"  - Law {chunk.law_number}: {chunk.title}")
        if chunk.section:
            print(f"    Section: {chunk.section}")
        if chunk.cross_references:
            print(f"    Cross-refs: {', '.join(chunk.cross_references[:5])}")
        if chunk.keywords:
            print(f"    Keywords: {', '.join(chunk.keywords[:5])}")
        preview = chunk.content[:100].encode('ascii', 'replace').decode('ascii')
        print(f"    Content preview: {preview}...")
        print()

    # Test complex query search
    print("\n" + "="*60)
    print("Testing multi-dimensional search for 'what happens when a defender revokes and then tries to correct it':")
    results = search_chunks("what happens when a defender revokes and then tries to correct it", chunks)
    for chunk in results:
        print(f"  - Law {chunk.law_number}: {chunk.title} - {chunk.section}")

    print("\n" + "="*60)
    print("Testing follow-up suggestions:")
    suggestions = get_follow_up_suggestions(results, "what happens when a defender revokes")
    for suggestion in suggestions:
        print(f"  - {suggestion}")
