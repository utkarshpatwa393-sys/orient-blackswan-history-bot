"""
Generate clean, high-resolution architecture diagram for Orient BlackSwan History Bot.
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def create_architecture_diagram(output_path: Path):
    width, height = 1200, 950
    # Warm off-white background
    img = Image.new("RGB", (width, height), color=(251, 249, 245))
    draw = ImageDraw.Draw(img)

    # Colors
    navy = (20, 30, 48)
    burgundy = (136, 19, 55)
    light_burgundy = (251, 235, 240)
    card_bg = (255, 255, 255)
    border_col = (210, 205, 195)
    accent_green = (21, 128, 61)
    accent_red = (185, 28, 28)

    # Title
    draw.rectangle([(0, 0), (width, 85)], fill=navy)
    draw.text((40, 25), "ORIENT BLACKSWAN HISTORY BOT — SYSTEM RAG ARCHITECTURE", fill=(255, 255, 255))

    # Helper box drawer
    def draw_box(x, y, w, h, title, subtitle="", bg=card_bg, border=border_col, title_color=navy):
        draw.rounded_rectangle([(x, y), (x + w, y + h)], radius=10, fill=bg, outline=border, width=2)
        draw.text((x + 18, y + 14), title, fill=title_color)
        if subtitle:
            draw.text((x + 18, y + 36), subtitle, fill=(100, 110, 125))

    def draw_arrow(x1, y1, x2, y2, color=navy):
        draw.line([(x1, y1), (x2, y2)], fill=color, width=3)
        # Arrowhead pointing downwards
        if y2 > y1:
            draw.polygon([(x2 - 6, y2 - 10), (x2 + 6, y2 - 10), (x2, y2)], fill=color)
        elif x2 > x1:
            draw.polygon([(x2 - 10, y2 - 6), (x2 - 10, y2 + 6), (x2, y2)], fill=color)

    # Column 1: Ingestion Pipeline (Left)
    draw.text((60, 110), "1. PDF DOCUMENT INGESTION PIPELINE", fill=burgundy)
    
    draw_box(60, 140, 320, 60, "Source Catalogue PDF", "History_2024-25.pdf (~80 pages)", bg=(245, 240, 230))
    draw_arrow(220, 200, 220, 235)

    draw_box(60, 235, 320, 60, "PyMuPDF PDF Loader", "Page-by-page extraction & 1-based page metadata")
    draw_arrow(220, 295, 220, 330)

    draw_box(60, 330, 320, 60, "Recursive Character Splitter", "1,000 char chunk size | 150 char overlap")
    draw_arrow(220, 390, 220, 425)

    draw_box(60, 425, 320, 60, "Google Gemini Embeddings", "Model: gemini-embedding-001 (3072 dims)")
    draw_arrow(220, 485, 220, 520)

    draw_box(60, 520, 320, 75, "Persistent ChromaDB Store", "Local storage in chroma_db/ with SHA-256 cache", bg=light_burgundy, border=burgundy, title_color=burgundy)

    # Arrow from ChromaDB to Retriever
    draw.line([(380, 560), (470, 560)], fill=burgundy, width=3)
    draw_arrow(470, 560, 510, 560, color=burgundy)

    # Column 2: Retrieval & Generation Pipeline (Right)
    draw.text((510, 110), "2. QUERY RETRIEVAL & ANSWER GENERATION PIPELINE", fill=burgundy)

    draw_box(510, 140, 350, 60, "User Natural Language Query", "e.g., 'In which year was publisher incorporated?'", bg=(240, 245, 250))
    draw_arrow(685, 200, 685, 235)

    draw_box(510, 235, 350, 60, "Query Embedding", "Gemini embedding model conversion")
    draw_arrow(685, 295, 685, 330)

    draw_box(510, 330, 350, 60, "ChromaDB Similarity Search", "Retrieve top-k (k=4) chunks with L2 distance")
    draw_arrow(685, 390, 685, 425)

    draw_box(510, 425, 350, 60, "Relevance & Distance Filter", "Validate semantic proximity threshold")
    draw_arrow(685, 485, 685, 520)

    draw_box(510, 520, 350, 65, "LangChain + Google Gemini", "Strict prompt grounding on retrieved context", bg=(245, 240, 250))
    draw_arrow(685, 585, 685, 620)

    draw_box(510, 620, 350, 65, "Citation & Evidence Verifier", "Verify exact quotation matches source chunk text")

    # Decision Split: Grounded Answer vs Not Found
    draw_arrow(590, 685, 590, 725)
    draw_arrow(780, 685, 780, 725)

    draw_box(450, 725, 270, 65, "Grounded Answer + Citation", "Answer: ... | Source: PDF page X | Evidence: '...'", bg=(240, 253, 244), border=accent_green, title_color=accent_green)
    draw_box(740, 725, 220, 65, "Strict Abstention", "Output exactly: 'Not found'", bg=(254, 242, 242), border=accent_red, title_color=accent_red)

    # Convergence into Streamlit UI
    draw.line([(585, 790), (585, 825)], fill=navy, width=3)
    draw.line([(850, 790), (850, 825)], fill=navy, width=3)
    draw.line([(585, 825), (850, 825)], fill=navy, width=3)
    draw_arrow(717, 825, 717, 860)

    draw_box(510, 860, 415, 65, "Streamlit Academic UI Interface", "Interactive document viewer + History Bot chat", bg=navy, border=navy, title_color=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))
    print(f"[SUCCESS] Architecture diagram generated at: {output_path}")


if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent.parent / "screenshots" / "05_architecture.png"
    create_architecture_diagram(out_file)
