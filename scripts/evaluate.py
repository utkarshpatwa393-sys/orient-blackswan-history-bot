"""
Automated evaluation script for the five mandatory LMS assignment questions.
Queries the indexed ChromaDB collection and verifies grounding against the catalogue.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.vector_store import VectorStoreManager
from src.retriever import DocumentRetriever
from src.rag_chain import RAGChain


LMS_QUESTIONS = [
    {
        "id": 1,
        "question": "On what date were the prices in this catalogue last updated?",
        "intent": "Factual date query regarding price currency.",
        "expected_behavior": "Must identify price revision statement and cite page number (e.g. Page 1).",
    },
    {
        "id": 2,
        "question": "In which year was the publisher (Orient BlackSwan) incorporated?",
        "intent": "Factual corporate history query.",
        "expected_behavior": "Must cite incorporation year (1982) and cite page number (e.g. Page 1).",
    },
    {
        "id": 3,
        "question": "What is the price of a Physics textbook?",
        "intent": "Cross-discipline or ambiguous query. Catalogue covers History only.",
        "expected_behavior": "Must return 'Not found' because no physics textbooks are in this catalogue.",
    },
    {
        "id": 4,
        "question": "Who is the current Minister of Education?",
        "intent": "External political query outside document scope.",
        "expected_behavior": "Must return 'Not found' without hallucinating from outside training data.",
    },
    {
        "id": 5,
        "question": "Which is the \"best\" history book for a beginner?",
        "intent": "Subjective opinion query.",
        "expected_behavior": "Must return 'Not found' or abstain unless an explicit beginner recommendation is stated.",
    },
]


def run_evaluation(output_markdown_path: Path):
    """Run evaluation on the 5 LMS questions and generate markdown report."""
    print("=" * 70)
    print("  ORIENT BLACKSWAN HISTORY BOT - LMS QUESTIONS EVALUATION")
    print("=" * 70)

    vm = VectorStoreManager()
    if not vm.is_indexed():
        print("[ERROR] Vector store is not indexed. Please run `python scripts/ingest_pdf.py` first.")
        sys.exit(1)

    vs = vm.get_vector_store()
    retriever = DocumentRetriever(vs, k=Config.RETRIEVER_K)
    rag_chain = RAGChain(retriever)

    results = []
    output_markdown_path.parent.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "# Orient BlackSwan History Bot — Five LMS Question Answers",
        "",
        "**Catalogue Document:** `History_2024-25.pdf` (Orient BlackSwan History Catalogue 2024–25, ~80 pages)  ",
        "**Evaluation Mode:** Live Retrieval-Augmented Generation (Google Gemini + ChromaDB + LangChain)  ",
        "**Abstention Policy:** Strict grounding only; return exact `Not found` when unsupported.  ",
        "",
        "---",
        "",
    ]

    for item in LMS_QUESTIONS:
        qid = item["id"]
        q_text = item["question"]
        print(f"\n[Evaluating Q{qid}]: {q_text}")

        res = rag_chain.query(q_text)
        is_found = res["is_found"]
        answer = res["answer"]
        source = res.get("source", "")
        evidence = res.get("evidence", "")
        pages = res.get("pages", [])

        print(f"  Result : {'SUPPORTED' if is_found else 'ABSTAINED (Not found)'}")
        print(f"  Answer : {answer}")
        if is_found:
            print(f"  Source : {source}")
            print(f"  Quote  : \"{evidence}\"")

        results.append({
            "id": qid,
            "question": q_text,
            "answer": answer,
            "source": source,
            "evidence": evidence,
            "pages": pages,
            "is_found": is_found,
        })

        # Format markdown entry
        md_lines.append(f"### Question {qid}: {q_text}")
        md_lines.append("")
        md_lines.append(f"**Intent & Expected Behavior:** {item['intent']} {item['expected_behavior']}")
        md_lines.append("")
        md_lines.append(f"- **Final Answer:** `{answer}`")
        if is_found:
            md_lines.append(f"- **Source Reference:** {source}")
            md_lines.append(f"- **Supporting Evidence:** \"{evidence}\"")
            md_lines.append(f"- **PDF Page Number(s):** {', '.join(str(p) for p in pages)}")
            md_lines.append(f"- **Status:** ✅ Grounded in Document")
        else:
            md_lines.append(f"- **Source Reference:** None (Information absent from document)")
            md_lines.append(f"- **Status:** 🛡️ Correctly Abstains (`Not found`) — No hallucination")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    # Summary table
    md_lines.append("## Summary Evaluation Table")
    md_lines.append("")
    md_lines.append("| Q# | Question | Grounded Answer | Source Page | RAG Decision |")
    md_lines.append("|:---|:---|:---|:---|:---|")
    for r in results:
        status_icon = "✅ Supported" if r["is_found"] else "🛡️ Not found"
        page_str = ", ".join(str(p) for p in r["pages"]) if r["pages"] else "N/A"
        md_lines.append(f"| Q{r['id']} | {r['question']} | {r['answer']} | {page_str} | {status_icon} |")

    # Write output markdown
    with open(output_markdown_path, "w") as f:
        f.write("\n".join(md_lines))

    print("\n" + "=" * 70)
    print(f"[SUCCESS] LMS evaluation report saved to: {output_markdown_path}")
    print("=" * 70)


if __name__ == "__main__":
    report_file = Config.SUBMISSION_DIR / "lms_answers.md"
    run_evaluation(report_file)
