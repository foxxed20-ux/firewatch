"""Build the FireWatch project passport and domain research as formal A4 PDFs.

The Markdown files remain the source of truth. This script only lays them out;
it neither enriches their claims nor changes their order.
"""

from __future__ import annotations

import hashlib
import html
import json
import posixpath
import re
import sys
from urllib.parse import quote
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
PRIVATE_DIR = ROOT / "artifacts" / "documents"
FONT_DIR = Path(r"C:\Windows\Fonts")
NAVY = colors.HexColor("#071C29")
SEA = colors.HexColor("#16313D")
ORANGE = colors.HexColor("#C9682D")
MIST = colors.HexColor("#EAF0ED")
RULE = colors.HexColor("#B8C6C1")
TEXT = colors.HexColor("#1E2D34")
GITHUB_BLOB = "https://github.com/foxxed20-ux/firewatch/blob/main/"


@dataclass(frozen=True)
class DocumentSpec:
    source: Path
    output: Path
    short_title: str
    expected_min: int
    expected_max: int


SPECS = (
    DocumentSpec(
        ROOT / "docs" / "PROJECT_PASSPORT.md",
        OUT_DIR / "FireWatch_Project_Passport.pdf",
        "Паспорт проекта",
        3,
        4,
    ),
    DocumentSpec(
        ROOT / "docs" / "DOMAIN_RESEARCH.md",
        OUT_DIR / "FireWatch_Domain_Research.pdf",
        "Исследование предметной области",
        5,
        8,
    ),
)


def register_fonts() -> tuple[str, str]:
    """Embed Arial with a Cyrillic-capable fallback already present in Windows."""
    regular = FONT_DIR / "arial.ttf"
    bold = FONT_DIR / "arialbd.ttf"
    if not regular.exists() or not bold.exists():
        raise FileNotFoundError("Arial regular/bold TTF files are required in C:\\Windows\\Fonts")
    pdfmetrics.registerFont(TTFont("FireWatchArial", str(regular)))
    pdfmetrics.registerFont(TTFont("FireWatchArialBold", str(bold)))
    return "FireWatchArial", "FireWatchArialBold"


def _link_target(target: str) -> str:
    if target.startswith(("https://", "http://", "mailto:", "#")):
        return target
    relative = posixpath.normpath(posixpath.join("docs", target.replace("\\", "/")))
    return GITHUB_BLOB + quote(relative, safe="/-_.~")


def inline(markdown: str) -> str:
    """Convert the small, explicit Markdown inline subset used by project docs."""
    escaped = html.escape(markdown.strip())
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)\s]+)\)",
        lambda match: f'<link href="{_link_target(html.unescape(match.group(2)))}" color="#C9682D">{match.group(1)}</link>',
        escaped,
    )
    escaped = re.sub(r"`([^`]+)`", r'<font name="FireWatchArial">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped


def styles(font: str, bold: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "FireWatchTitle", parent=base["Title"], fontName=bold, fontSize=22,
            leading=26, textColor=NAVY, alignment=TA_LEFT, spaceAfter=8 * mm,
        ),
        "subtitle": ParagraphStyle(
            "FireWatchSubtitle", parent=base["Normal"], fontName=font, fontSize=11,
            leading=15, textColor=SEA, spaceAfter=8 * mm,
        ),
        "h1": ParagraphStyle(
            "FireWatchH1", parent=base["Heading1"], fontName=bold, fontSize=15,
            leading=19, textColor=NAVY, spaceBefore=7 * mm, spaceAfter=3 * mm,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "FireWatchH2", parent=base["Heading2"], fontName=bold, fontSize=12,
            leading=15, textColor=SEA, spaceBefore=5 * mm, spaceAfter=2.5 * mm,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "FireWatchH3", parent=base["Heading3"], fontName=bold, fontSize=10.5,
            leading=13, textColor=SEA, spaceBefore=4 * mm, spaceAfter=2 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "FireWatchBody", parent=base["BodyText"], fontName=font, fontSize=10.4,
            leading=14.3, textColor=TEXT, alignment=TA_JUSTIFY, spaceAfter=3.2 * mm,
        ),
        "bullet": ParagraphStyle(
            "FireWatchBullet", parent=base["BodyText"], fontName=font, fontSize=10.1,
            leading=13.7, textColor=TEXT, leftIndent=0, spaceAfter=1.3 * mm,
        ),
        "table_head": ParagraphStyle(
            "FireWatchTableHead", parent=base["BodyText"], fontName=bold, fontSize=10.5,
            leading=13.1, textColor=colors.white,
        ),
        "table": ParagraphStyle(
            "FireWatchTable", parent=base["BodyText"], fontName=font, fontSize=10.5,
            leading=13.1, textColor=TEXT,
        ),
        "caption": ParagraphStyle(
            "FireWatchCaption", parent=base["BodyText"], fontName=font, fontSize=10.5,
            leading=13.1, textColor=SEA, alignment=TA_LEFT,
        ),
    }


def is_table_row(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")


def table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip()[1:-1].split("|")]


def is_separator(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def make_table(lines: list[str], st: dict[str, ParagraphStyle], usable_width: float) -> Table:
    rows = [table_cells(line) for line in lines]
    if len(rows) < 2 or not is_separator(rows[1]):
        raise ValueError("Malformed Markdown table")
    values = [rows[0], *rows[2:]]
    columns = max(len(row) for row in values)
    normalised = [row + [""] * (columns - len(row)) for row in values]
    data = [
        [Paragraph(inline(cell), st["table_head"] if index == 0 else st["table"]) for cell in row]
        for index, row in enumerate(normalised)
    ]
    col_width = usable_width / columns
    table = Table(data, colWidths=[col_width] * columns, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def is_list_item(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*+] |\d+[.)] )", line))


def parse_markdown(markdown: str, st: dict[str, ParagraphStyle], usable_width: float) -> list:
    lines = markdown.replace("\r\n", "\n").split("\n")
    flowables: list = []
    index = 0
    title_seen = False
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("#"):
            hashes, _, text = line.partition(" ")
            level = min(len(hashes), 3)
            if level == 1 and not title_seen:
                flowables.extend([
                    HRFlowable(width="28%", thickness=2.4, color=ORANGE, spaceAfter=4 * mm),
                    Paragraph(inline(text), st["title"]),
                    Paragraph("FireWatch · КосмоХакатон 2026", st["subtitle"]),
                ])
                title_seen = True
            else:
                flowables.append(Paragraph(inline(text), st[f"h{level}"]))
            index += 1
            continue
        if is_table_row(line):
            table_lines: list[str] = []
            while index < len(lines) and is_table_row(lines[index]):
                table_lines.append(lines[index])
                index += 1
            flowables.extend([make_table(table_lines, st, usable_width), Spacer(1, 4 * mm)])
            continue
        if is_list_item(line):
            ordered = bool(re.match(r"^\s*\d+[.)] ", line))
            items: list[ListItem] = []
            while index < len(lines) and is_list_item(lines[index]):
                item_text = re.sub(r"^\s*(?:[-*+] |\d+[.)] )", "", lines[index]).strip()
                items.append(ListItem(Paragraph(inline(item_text), st["bullet"])))
                index += 1
            flowables.append(ListFlowable(items, bulletType="1" if ordered else "bullet", leftIndent=13 * mm, bulletFontName=st["bullet"].fontName))
            flowables.append(Spacer(1, 2 * mm))
            continue
        paragraph_lines = [line.strip()]
        index += 1
        while index < len(lines) and lines[index].strip() and not lines[index].startswith("#") and not is_table_row(lines[index]) and not is_list_item(lines[index]):
            paragraph_lines.append(lines[index].strip())
            index += 1
        flowables.append(Paragraph(inline(" ".join(paragraph_lines)), st["body"]))
    return flowables


def header_footer(canvas, doc, short_title: str) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.45)
    canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
    canvas.setFont("FireWatchArialBold", 8.4)
    canvas.setFillColor(NAVY)
    canvas.drawString(18 * mm, height - 10 * mm, "FIREWATCH")
    canvas.setFont("FireWatchArial", 8.4)
    canvas.setFillColor(SEA)
    canvas.drawRightString(width - 18 * mm, height - 10 * mm, short_title)
    canvas.setStrokeColor(RULE)
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.setFont("FireWatchArial", 8)
    canvas.setFillColor(SEA)
    canvas.drawString(18 * mm, 8.5 * mm, "КосмоХакатон 2026")
    canvas.drawRightString(width - 18 * mm, 8.5 * mm, f"Страница {doc.page}")
    canvas.restoreState()


def build(spec: DocumentSpec, font: str, bold: str) -> dict:
    markdown = spec.source.read_text(encoding="utf-8")
    document = SimpleDocTemplate(
        str(spec.output), pagesize=A4,
        leftMargin=19 * mm, rightMargin=19 * mm, topMargin=21 * mm, bottomMargin=19 * mm,
        title=spec.short_title, author="Команда FireWatch", subject="КосмоХакатон 2026",
    )
    st = styles(font, bold)
    story = parse_markdown(markdown, st, A4[0] - 38 * mm)
    document.build(story, onFirstPage=lambda c, d: header_footer(c, d, spec.short_title), onLaterPages=lambda c, d: header_footer(c, d, spec.short_title))
    digest = hashlib.sha256(spec.output.read_bytes()).hexdigest()
    return {"source": str(spec.source.relative_to(ROOT)), "output": str(spec.output.relative_to(ROOT)), "sha256": digest}


def main() -> int:
    missing = [str(spec.source.relative_to(ROOT)) for spec in SPECS if not spec.source.exists()]
    if missing:
        print("Waiting for Markdown sources: " + ", ".join(missing), file=sys.stderr)
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    font, bold = register_fonts()
    records = [build(spec, font, bold) for spec in SPECS]
    (PRIVATE_DIR / "project_documents_manifest.json").write_text(
        json.dumps({"generated": date.today().isoformat(), "documents": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for record in records:
        print(record["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
