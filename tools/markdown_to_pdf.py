"""Render a project Markdown report to PDF (A4 landscape, wrapped tables, page numbers).

    python tools/markdown_to_pdf.py docs/training_recipes.md Resources/Training_Recipes_Reference_Papers.pdf

Supports the subset the reports use: # / ## headings, paragraphs, nested "-" bullets, "1." lists, pipe tables,
**bold**, `code`, [text](url). Uses Windows Arial (falls back to Helvetica) so symbols such as -> <= alpha render.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT_DIR = Path("C:/Windows/Fonts")


def register_fonts() -> tuple[str, str, str]:
    try:
        pdfmetrics.registerFont(TTFont("Arial", str(FONT_DIR / "arial.ttf")))
        pdfmetrics.registerFont(TTFont("Arial-Bold", str(FONT_DIR / "arialbd.ttf")))
        pdfmetrics.registerFont(TTFont("Arial-Italic", str(FONT_DIR / "ariali.ttf")))
        pdfmetrics.registerFont(TTFont("Consolas", str(FONT_DIR / "consola.ttf")))
        pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic",
                                      boldItalic="Arial-Bold")
        return "Arial", "Arial-Bold", "Consolas"
    except Exception:  # noqa: BLE001 - not on Windows: built-in fonts (some symbols may not render)
        return "Helvetica", "Helvetica-Bold", "Courier"


def inline_markup(text: str, code_font: str) -> str:
    text = text.replace("\u2011", "-")                                    # non-breaking hyphen: missing glyph
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)                # links -> their text
    text = re.sub(r"`([^`]+)`", rf'<font face="{code_font}">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def build_styles(body_font, bold_font):
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName=bold_font, fontSize=18, leading=22,
                                alignment=TA_LEFT, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName=bold_font, fontSize=13, leading=16,
                             spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#1c5cab")),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName=body_font, fontSize=9.5, leading=12.5,
                               spaceAfter=4),
        "cell": ParagraphStyle("cell", parent=base["BodyText"], fontName=body_font, fontSize=7.6, leading=9.4),
        "header": ParagraphStyle("header", parent=base["BodyText"], fontName=bold_font, fontSize=7.8, leading=9.6),
    }


def parse_table(lines, styles, code_font, available_width):
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    rows = [rows[0]] + [r for r in rows[1:] if not re.fullmatch(r"[\s:\-|]+", "|".join(r))]
    n_columns = len(rows[0])
    data = [[Paragraph(inline_markup(cell, code_font), styles["header" if i == 0 else "cell"])
             for cell in row + [""] * (n_columns - len(row))] for i, row in enumerate(rows)]
    lengths = [max(min(len(row[c]) if c < len(row) else 0, 70) for row in rows) + 6 for c in range(n_columns)]
    widths = [available_width * length / sum(lengths) for length in lengths]
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef6")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9c8c3")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafaf8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def parse_list(lines, styles, code_font):
    """Nested "-" / "1." list from lines (indent of 2+ spaces = one level deeper)."""
    items, index = [], 0
    numbered = bool(re.match(r"\s*\d+\.\s", lines[0]))
    base_indent = len(lines[0]) - len(lines[0].lstrip())
    while index < len(lines):
        line = lines[index]
        text = re.sub(r"^\s*(-|\d+\.)\s+", "", line)
        children = []
        index += 1
        while index < len(lines) and len(lines[index]) - len(lines[index].lstrip()) > base_indent:
            children.append(lines[index])
            index += 1
        flowables = [Paragraph(inline_markup(text, code_font), styles["body"])]
        if children:
            flowables.append(parse_list(children, styles, code_font))
        items.append(ListItem(flowables, leftIndent=12))
    return ListFlowable(items, bulletType="1" if numbered else "bullet", start="1" if numbered else "•",
                        leftIndent=12, bulletFontSize=8)


def markdown_to_story(markdown: str, styles, code_font, available_width):
    story, lines, i = [], markdown.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("# "):
            story.append(Paragraph(inline_markup(line[2:], code_font), styles["title"]))
            i += 1
        elif line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:], code_font), styles["h2"]))
            i += 1
        elif line.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            story += [parse_table(block, styles, code_font, available_width), Spacer(1, 6)]
        elif re.match(r"\s*(-|\d+\.)\s", line):
            block = []
            while i < len(lines) and lines[i].strip() and (re.match(r"\s*(-|\d+\.)\s", lines[i])
                                                            or lines[i].startswith("  ")):
                block.append(lines[i])
                i += 1
            story += [parse_list(block, styles, code_font), Spacer(1, 3)]
        else:
            block = []
            while i < len(lines) and lines[i].strip() and not re.match(r"(#|\s*\||\s*(-|\d+\.)\s)", lines[i]):
                block.append(lines[i].strip())
                i += 1
            story.append(Paragraph(inline_markup(" ".join(block), code_font), styles["body"]))
    return story


def convert(markdown_path: Path, pdf_path: Path):
    body_font, bold_font, code_font = register_fonts()
    styles = build_styles(body_font, bold_font)
    page_size = landscape(A4)
    margin = 1.4 * cm
    document = SimpleDocTemplate(str(pdf_path), pagesize=page_size, leftMargin=margin, rightMargin=margin,
                                 topMargin=margin, bottomMargin=margin,
                                 title=markdown_path.stem.replace("_", " ").title(), author="SET Lab, MIST")
    story = markdown_to_story(markdown_path.read_text(encoding="utf-8"), styles, code_font,
                              page_size[0] - 2 * margin)

    def footer(canvas, doc):
        canvas.setFont(body_font, 7.5)
        canvas.setFillColor(colors.HexColor("#52514e"))
        canvas.drawRightString(page_size[0] - margin, 0.8 * cm, f"page {doc.page}")
        canvas.drawString(margin, 0.8 * cm, f"generated from {markdown_path.as_posix()}")

    document.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python tools/markdown_to_pdf.py <input.md> <output.pdf>")
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print("wrote", sys.argv[2])
