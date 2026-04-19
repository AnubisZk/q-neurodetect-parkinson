"""PDF report generator using ReportLab (skeleton)."""
from __future__ import annotations
import uuid
import logging
from pathlib import Path
from datetime import datetime
from app.schemas.prediction_schema import PredictionResponse
from app.utils.config import settings

logger = logging.getLogger(__name__)

REPORTS_DIR = settings.BASE_DIR / "data" / "reports"


def generate_pdf_report(response: PredictionResponse) -> str:
    """
    Generate a PDF report and return the file path (relative URL).
    Falls back gracefully if ReportLab is not installed.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"report_{response.request_id}.pdf"
    output_path = REPORTS_DIR / filename

    try:
        from reportlab.lib.pagesizes import A4  # type: ignore
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle  # type: ignore
        from reportlab.lib.styles import getSampleStyleSheet  # type: ignore
        from reportlab.lib import colors  # type: ignore

        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Q-NeuroDetect Parkinson – Analiz Raporu", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Rapor ID: {response.request_id}", styles["Normal"]))
        story.append(Paragraph(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Risk summary
        r = response.fusion
        story.append(Paragraph(f"Risk Skoru: {r.score:.0%} – {r.level} Risk ({r.label})", styles["Heading2"]))
        story.append(Spacer(1, 8))

        # Modality table
        table_data = [["Modalite", "Tahmin", "Olasılık", "Model Tipi"]]
        for m in response.modalities:
            table_data.append([m.modality.capitalize(), m.label, f"{m.probability:.2%}", m.model_type])
        t = Table(table_data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2D6A4F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F0F0")]),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

        # Explanation
        story.append(Paragraph("Açıklama", styles["Heading2"]))
        for line in response.explanation.split("\n"):
            story.append(Paragraph(line or " ", styles["Normal"]))
        story.append(Spacer(1, 8))

        # Model comparison
        story.append(Paragraph("Model Karşılaştırması", styles["Heading2"]))
        cmp_data = [["Model", "Tip", "Olasılık", "Sonuç"]]
        for c in response.model_comparison:
            cmp_data.append([c.model_name, c.model_type, f"{c.probability:.2%}", c.label])
        tc = Table(cmp_data, hAlign="LEFT")
        tc.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B4332")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(tc)

        doc.build(story)
        logger.info("PDF report generated: %s", output_path)

    except Exception as exc:
        logger.warning("PDF generation failed (%s) – creating placeholder", exc)
        output_path.write_text(f"Report placeholder for {response.request_id}\n{exc}")

    return f"/reports/{filename}"
