"""
AeroSmart / EpiGraph AI — Epidemic Risk Intelligence Report
Professional light-mode PDF generator using ReportLab platypus + pdfgen.canvas
"""

from io import BytesIO
import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.utils import simpleSplit


# ─────────────────────────────────────────────
#  Design tokens
# ─────────────────────────────────────────────
NAVY        = colors.HexColor("#0f172a")
BLUE_700    = colors.HexColor("#1d4ed8")
BLUE_500    = colors.HexColor("#3b82f6")
BLUE_100    = colors.HexColor("#dbeafe")
BLUE_50     = colors.HexColor("#eff6ff")
SLATE_800   = colors.HexColor("#1e293b")
SLATE_700   = colors.HexColor("#334155")
SLATE_600   = colors.HexColor("#475569")
SLATE_400   = colors.HexColor("#94a3b8")
SLATE_200   = colors.HexColor("#e2e8f0")
SLATE_100   = colors.HexColor("#f1f5f9")
SLATE_50    = colors.HexColor("#f8fafc")
WHITE       = colors.white
RED_700     = colors.HexColor("#b91c1c")
RED_100     = colors.HexColor("#fee2e2")
AMBER_700   = colors.HexColor("#b45309")
AMBER_100   = colors.HexColor("#fef3c7")
GREEN_700   = colors.HexColor("#15803d")
GREEN_100   = colors.HexColor("#dcfce7")
RED_500     = colors.HexColor("#ef4444")
AMBER_500   = colors.HexColor("#f59e0b")
GREEN_500   = colors.HexColor("#22c55e")

PAGE_W, PAGE_H = A4
TOP_CHROME  = 28
BOT_CHROME  = 22
LEFT_M      = 2 * cm
RIGHT_M     = 2 * cm
TOP_M       = TOP_CHROME + 1.2 * cm
BOT_M       = BOT_CHROME + 1.0 * cm
CONTENT_W   = PAGE_W - LEFT_M - RIGHT_M


# ─────────────────────────────────────────────
#  Page chrome canvas subclass
# ─────────────────────────────────────────────
class PageCanvas(canvas.Canvas):
    def __init__(self, *args, hospital="City General Hospital", district="Metro District", **kwargs):
        super().__init__(*args, **kwargs)
        self._hospital = hospital
        self._district = district
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_chrome(self._pageNumber, num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def _draw_chrome(self, page_num, total_pages):
        self.saveState()

        # Top bar
        self.setFillColor(NAVY)
        self.rect(0, PAGE_H - TOP_CHROME, PAGE_W, TOP_CHROME, fill=1, stroke=0)
        self.setFillColor(WHITE)
        self.setFont("Helvetica-Bold", 9)
        self.drawString(LEFT_M, PAGE_H - TOP_CHROME + 9,
                        "EpiGraph AI  \u00b7  AeroSmart Surveillance Platform")
        self.setFillColor(BLUE_500)
        self.setFont("Helvetica", 8)
        self.drawRightString(PAGE_W - RIGHT_M, PAGE_H - TOP_CHROME + 9,
                             f"{self._hospital}  |  {self._district}")
        self.setStrokeColor(BLUE_700)
        self.setLineWidth(1.5)
        self.line(0, PAGE_H - TOP_CHROME, PAGE_W, PAGE_H - TOP_CHROME)

        # Bottom bar
        self.setFillColor(SLATE_100)
        self.rect(0, 0, PAGE_W, BOT_CHROME, fill=1, stroke=0)
        self.setStrokeColor(SLATE_200)
        self.setLineWidth(0.5)
        self.line(0, BOT_CHROME, PAGE_W, BOT_CHROME)
        self.setFillColor(RED_700)
        self.setFont("Helvetica-Bold", 7)
        self.drawString(LEFT_M, BOT_CHROME - 14, "CONFIDENTIAL")
        self.setFillColor(SLATE_600)
        self.setFont("Helvetica", 7)
        self.drawRightString(PAGE_W - RIGHT_M, BOT_CHROME - 14,
                             f"Page {page_num} of {total_pages}")
        self.restoreState()


# ─────────────────────────────────────────────
#  Custom Flowable: ProgressBar
# ─────────────────────────────────────────────
class ProgressBar(Flowable):
    HEIGHT = 14

    def __init__(self, value: float, max_value: float = 100,
                 label: str = "", bar_color=None, width=None):
        super().__init__()
        self.value     = value
        self.max_value = max_value or 1
        self.label     = label
        self._w        = width
        self.bar_color = bar_color

    def wrap(self, avail_w, avail_h):
        self._w = avail_w
        return avail_w, self.HEIGHT + 4

    def draw(self):
        ratio = min(self.value / self.max_value, 1.0)
        c = self.canv
        W, H = self._w, self.HEIGHT
        r = H / 2

        if self.bar_color:
            fill = self.bar_color
        elif ratio > 0.5:
            fill = RED_500
        elif ratio > 0.25:
            fill = AMBER_500
        else:
            fill = GREEN_500

        c.setFillColor(SLATE_200)
        c.roundRect(0, 2, W, H, r, fill=1, stroke=0)

        fill_w = max(ratio * W, r * 2 if ratio > 0 else 0)
        if fill_w > 0:
            c.setFillColor(fill)
            c.roundRect(0, 2, fill_w, H, r, fill=1, stroke=0)

        if self.label:
            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 6.5)
            c.drawCentredString(fill_w / 2, 2 + H / 2 - 3, self.label)

        c.setFillColor(SLATE_600)
        c.setFont("Helvetica", 6.5)
        c.drawRightString(W, 2 + H / 2 - 3, f"{ratio*100:.1f}%")


# ─────────────────────────────────────────────
#  Style helpers
# ─────────────────────────────────────────────
def _S(name, **kw):
    base = dict(fontName="Helvetica", fontSize=8.5, leading=12,
                textColor=SLATE_800, spaceAfter=2)
    base.update(kw)
    return ParagraphStyle(name, **base)


STYLES = {
    "section_hdr": _S("sh", fontName="Helvetica-Bold", fontSize=12,
                       textColor=BLUE_700, spaceBefore=14, spaceAfter=4),
    "body":        _S("body"),
    "small":       _S("small", fontSize=7.5, textColor=SLATE_600),
    "center":      _S("center", alignment=TA_CENTER),
    "meta":        _S("meta", fontSize=7.5, textColor=SLATE_600, alignment=TA_CENTER),
    "white_bold":  _S("wb", fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER),
    "kpi_num":     _S("kpi_num", fontName="Helvetica-Bold", fontSize=24,
                       alignment=TA_CENTER, leading=28),
    "kpi_label":   _S("kpi_lbl", fontSize=7, textColor=SLATE_600, alignment=TA_CENTER),
    "footer":      _S("foot", fontSize=6.5, textColor=SLATE_400, alignment=TA_CENTER),
}

HR = lambda: HRFlowable(width="100%", thickness=0.5, color=SLATE_200, spaceAfter=6)
SP = lambda h=0.4: Spacer(1, h * cm)

def P(text, style="body"):
    if isinstance(style, str):
        style = STYLES[style]
    return Paragraph(str(text), style)

def tbl_style(extra=None):
    base = [
        ("FONTNAME",       (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1,  0), 8),
        ("BACKGROUND",     (0, 0), (-1,  0), NAVY),
        ("TEXTCOLOR",      (0, 0), (-1,  0), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SLATE_50]),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 1), (-1, -1), 8),
        ("TEXTCOLOR",      (0, 1), (-1, -1), SLATE_800),
        ("ALIGN",          (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("GRID",           (0, 0), (-1, -1), 0.4, SLATE_200),
        ("LINEBELOW",      (0, 0), (-1,  0), 1.5, BLUE_700),
    ]
    if extra:
        base.extend(extra)
    return TableStyle(base)


# ─────────────────────────────────────────────
#  Section builders
# ─────────────────────────────────────────────

def _build_title_block(data: dict) -> list:
    hospital = data.get("hospital", "City General Hospital")
    district = data.get("district", "Metro District")
    census   = data.get("censuscode", "—")
    gen_at   = data.get("generated_at", datetime.datetime.now().isoformat())

    title_tbl = Table(
        [[P("Epidemic Risk Intelligence Report",
            _S("t", fontName="Helvetica-Bold", fontSize=22, textColor=WHITE,
               alignment=TA_CENTER, leading=26))],
         [P(hospital,
            _S("h", fontName="Helvetica-Bold", fontSize=11,
               textColor=BLUE_500, alignment=TA_CENTER))],
         [P(f"{district}  \u00b7  Census Code: {census}  \u00b7  Generated: {gen_at}",
            _S("m", fontSize=7.5, textColor=SLATE_400, alignment=TA_CENTER))]],
        colWidths=[CONTENT_W],
    )
    title_tbl.setStyle(TableStyle([
        ("BACKGROUND",      (0, 0), (-1, -1), NAVY),
        ("ROWBACKGROUNDS",  (0, 0), (-1, -1), [NAVY]),
        ("BOX",             (0, 0), (-1, -1), 0, NAVY),
        ("TOPPADDING",      (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",   (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS",  [6, 6, 6, 6]),
    ]))
    return [title_tbl, SP(0.5)]


def _build_kpi_row(data: dict) -> list:
    status = data.get("status", {})
    prob   = status.get("outbreak_prob", 0)
    pred   = status.get("pred_cases", 0)
    ehr_ct = len(data.get("uploaded_cases", []))

    def kpi_cell(value, label, colour):
        inner = Table(
            [[P(str(value), _S("kv", fontName="Helvetica-Bold", fontSize=24,
                               textColor=colour, alignment=TA_CENTER, leading=28))],
             [P(label, _S("kl", fontSize=7.5, textColor=SLATE_600, alignment=TA_CENTER))]],
            colWidths=[CONTENT_W / 3 - 10],
        )
        inner.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), WHITE),
            ("ROWBACKGROUNDS",(0, 0), (-1, -1), [WHITE]),
            ("BOX",           (0, 0), (-1, -1), 0.8, SLATE_200),
            ("TOPPADDING",    (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ("ROUNDEDCORNERS",[6, 6, 6, 6]),
        ]))
        return inner

    row = Table(
        [[kpi_cell(f"{prob:.1f}%",  "Outbreak Probability",
                   BLUE_700),
          kpi_cell(str(pred),       "Predicted Cases (7-day)",
                   RED_700 if prob > 50 else AMBER_700),
          kpi_cell(str(ehr_ct),     "EHR Records Uploaded",
                   GREEN_700)]],
        colWidths=[CONTENT_W / 3] * 3, hAlign="LEFT",
    )
    row.setStyle(TableStyle([
        ("LEFTPADDING",  (0,0),(-1,-1), 4),
        ("RIGHTPADDING", (0,0),(-1,-1), 4),
        ("TOPPADDING",   (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 0),
    ]))
    return [P("Key Performance Indicators", "section_hdr"), row, SP(0.4)]


def _build_alert_pill(data: dict) -> list:
    level   = data.get("status", {}).get("alert_level", "LOW").upper()
    prob    = data.get("status", {}).get("outbreak_prob", 0)
    pred    = data.get("status", {}).get("pred_cases", 0)

    colours = {
        "HIGH":   (RED_100,   RED_700),
        "MEDIUM": (AMBER_100, AMBER_700),
        "LOW":    (GREEN_100, GREEN_700),
    }
    icons = {"HIGH": "\u25b2", "MEDIUM": "\u25c6", "LOW": "\u2713"}
    bg, border = colours.get(level, colours["LOW"])
    icon = icons.get(level, "\u25cf")

    msg_style = _S("al", fontName="Helvetica-Bold", fontSize=10,
                   textColor=border, alignment=TA_CENTER)
    sub_style = _S("als", fontSize=8, textColor=border, alignment=TA_CENTER)

    pill = Table(
        [[P(f"{icon}  ALERT LEVEL: {level}  \u2014  Outbreak Probability {prob:.1f}%  |  "
            f"Predicted Cases: {pred}", msg_style)],
         [P("Immediate epidemiological review recommended based on current sensor and EHR data fusion.",
            sub_style)]],
        colWidths=[CONTENT_W],
    )
    pill.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [bg]),
        ("BOX",           (0, 0), (-1, -1), 1.5, border),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS",[8, 8, 8, 8]),
    ]))
    return [P("Active Alert Status", "section_hdr"), pill, SP(0.4)]


def _build_ehr_summary(uploaded_cases: list) -> list:
    total   = len(uploaded_cases)
    pos     = sum(1 for c in uploaded_cases if str(c.get("dengue_status","")).lower() in ("positive","1",1))
    neg     = total - pos
    rate    = (pos / total * 100) if total else 0
    avg_tmp = (sum(c.get("temperature_c", 0) for c in uploaded_cases) / total) if total else 0

    def stat_cell(value, label, colour):
        inner = Table(
            [[P(str(value), _S("sv", fontName="Helvetica-Bold", fontSize=18,
                               textColor=colour, alignment=TA_CENTER, leading=22))],
             [P(label, _S("sl", fontSize=7, textColor=SLATE_600, alignment=TA_CENTER))]],
            colWidths=[CONTENT_W / 5 - 4],
        )
        inner.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), SLATE_50),
            ("ROWBACKGROUNDS",(0, 0), (-1, -1), [SLATE_50]),
            ("BOX",           (0, 0), (-1, -1), 0.6, SLATE_200),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("ROUNDEDCORNERS",[4, 4, 4, 4]),
        ]))
        return inner

    stat_row = Table(
        [[stat_cell(total,           "Total Records",   BLUE_700),
          stat_cell(pos,             "Dengue Positive", RED_700),
          stat_cell(neg,             "Dengue Negative", GREEN_700),
          stat_cell(f"{rate:.1f}%",  "Positivity Rate", RED_700 if rate > 50 else AMBER_700),
          stat_cell(f"{avg_tmp:.1f}\u00b0C", "Avg Temp", SLATE_700)]],
        colWidths=[CONTENT_W / 5] * 5,
    )
    stat_row.setStyle(TableStyle([
        ("LEFTPADDING",  (0,0),(-1,-1), 3),
        ("RIGHTPADDING", (0,0),(-1,-1), 3),
        ("TOPPADDING",   (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 0),
    ]))

    bar = ProgressBar(rate, 100, label=f"Positivity Rate: {rate:.1f}%")

    # Patient table
    header = ["File / Record ID", "Temperature (\u00b0C)", "Dengue Status"]
    rows   = [header]
    status_colours = []
    for case in uploaded_cases[:25]:
        raw_status = case.get("dengue_status", "")
        if str(raw_status).lower() in ("positive", "1") or raw_status == 1:
            status_text = "Positive"
        elif str(raw_status).lower() in ("negative", "0") or raw_status == 0:
            status_text = "Negative"
        else:
            status_text = "Unknown"
        rows.append([
            case.get("filename", "\u2014"),
            f"{case.get('temperature_c', 0):.1f}",
            status_text,
        ])
        status_colours.append(RED_700 if status_text == "Positive" else GREEN_700)

    extra = []
    for i, col in enumerate(status_colours, start=1):
        extra.extend([
            ("TEXTCOLOR", (2, i), (2, i), col),
            ("FONTNAME",  (2, i), (2, i), "Helvetica-Bold"),
        ])

    patient_tbl = Table(rows, colWidths=[CONTENT_W * 0.6, CONTENT_W * 0.2, CONTENT_W * 0.2])
    patient_tbl.setStyle(tbl_style(extra))

    return [
        P("EHR Patient Summary", "section_hdr"),
        stat_row, SP(0.3),
        bar, SP(0.3),
        patient_tbl, SP(0.4),
    ]


def _build_shap_section(shap_data: dict) -> list:
    features = shap_data.get("feature_importance", [])
    if not features:
        return []

    max_imp = max(abs(f.get("importance", 0)) for f in features) or 1

    header = ["Rank", "Feature Name", "SHAP Score", "Attribution"]
    rows   = [header]
    bar_rows = []

    for i, feat in enumerate(features[:12], 1):
        imp       = feat.get("importance", 0)
        sign      = "+" if imp >= 0 else "\u2212"
        colour    = RED_700 if imp >= 0 else GREEN_700
        bar_color = RED_500 if imp >= 0 else GREEN_500
        bar       = ProgressBar(abs(imp), max_imp, label="", bar_color=bar_color)
        rows.append([str(i), feat.get("feature", "Unknown"), f"{sign}{abs(imp):.4f}", bar])
        bar_rows.append((i, colour))

    col_w = [CONTENT_W * 0.07, CONTENT_W * 0.40, CONTENT_W * 0.15, CONTENT_W * 0.38]
    shap_tbl = Table(rows, colWidths=col_w, rowHeights=[20] + [22] * (len(rows) - 1))

    extra = []
    for i, col in bar_rows:
        extra.extend([
            ("TEXTCOLOR", (2, i), (2, i), col),
            ("FONTNAME",  (2, i), (2, i), "Helvetica-Bold"),
        ])
    shap_tbl.setStyle(tbl_style(extra))

    return [
        P("SHAP Feature Importance", "section_hdr"),
        P("Features ranked by absolute SHAP value contribution to the outbreak risk prediction. "
          "Red bars = risk-increasing features; green bars = risk-reducing features.", "small"),
        SP(0.2), shap_tbl, SP(0.4),
    ]


def _build_timeline(timeline: list) -> list:
    def level_map(p):
        return "HIGH" if p > 0.7 else ("MEDIUM" if p > 0.4 else "LOW")
    colour_map = {"HIGH": RED_700, "MEDIUM": AMBER_700, "LOW": GREEN_700}

    header = ["Week", "Year", "Actual Cases", "Predicted Cases", "Probability %", "Risk Level"]
    rows   = [header]
    lvl_colours = []

    for row in timeline:
        prob  = row.get("prob", 0)
        level = level_map(prob)
        rows.append([
            str(row.get("week",  "\u2014")),
            str(row.get("year",  "\u2014")),
            str(row.get("actual_cases", "\u2014")),
            str(row.get("pred_cases",   "\u2014")),
            f"{prob*100:.1f}%",
            level,
        ])
        lvl_colours.append((len(rows) - 1, colour_map[level]))

    extra = []
    for i, col in lvl_colours:
        extra.extend([
            ("TEXTCOLOR", (5, i), (5, i), col),
            ("FONTNAME",  (5, i), (5, i), "Helvetica-Bold"),
        ])

    tl_tbl = Table(rows, colWidths=[CONTENT_W / 6] * 6)
    tl_tbl.setStyle(tbl_style(extra))

    return [P("Epidemiological Timeline", "section_hdr"), tl_tbl, SP(0.4)]


def _build_methodology() -> list:
    text = (
        "<b>Model Architecture &amp; Methodology:</b> The FedXGNN Explainable Federated Learning "
        "Framework employs a split Graph Attention Network (GAT) trained on multi-modal "
        "environmental sensor streams fused with anonymised EHR records. Each hospital edge node "
        "processes patient data locally via spaCy NLP, then transmits a 64-dimensional temporal "
        "embedding to the central GAT which aggregates spatial outbreak risk across 284 Indian "
        "districts. SHAP (KernelSHAP) values are computed per inference against a 15-sample "
        "historical baseline to provide XAI-grade attribution. Outbreak probability thresholds "
        "\u2014 HIGH (&gt;70%), MEDIUM (&gt;40%), LOW (\u226440%) \u2014 are calibrated against "
        "WHO Dengue surveillance benchmarks. Raw patient data never leaves the edge node, ensuring "
        "full privacy compliance. All outputs should be reviewed by a qualified epidemiologist "
        "before clinical or policy action."
    )
    meth_tbl = Table(
        [[Paragraph(text, _S("mp", fontSize=8, textColor=SLATE_700, leading=12))]],
        colWidths=[CONTENT_W],
    )
    meth_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), BLUE_50),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [BLUE_50]),
        ("BOX",           (0, 0), (-1, -1), 1.0, BLUE_100),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
        ("ROUNDEDCORNERS",[6, 6, 6, 6]),
    ]))
    return [P("Methodology &amp; Model Notes", "section_hdr"), meth_tbl, SP(0.5)]


def _build_footer() -> list:
    year = datetime.datetime.now().year
    return [
        HR(),
        P(f"\u00a9 {year} FedXGNN \u00b7 EpiGraph AI Platform  \u00b7  "
          f"This document is auto-generated and intended solely for authorised public health personnel.",
          "footer"),
    ]


# ─────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────
def generate_report(data: dict, uploaded_cases: list) -> BytesIO:
    """
    Build and return a BytesIO PDF buffer.

    data keys:
        district, censuscode, generated_at, hospital
        status  -> {alert_level, outbreak_prob, pred_cases}
        shap    -> {feature_importance: [{feature, importance}]}
        timeline -> [{year, week, actual_cases, pred_cases, prob}]
    uploaded_cases: [{filename, temperature_c, dengue_status}]
      dengue_status can be int (0/1) or str ("Positive"/"Negative")
    """
    hospital = data.get("hospital", data.get("district", "Hospital"))
    district = data.get("district", "District")
    data["uploaded_cases"] = uploaded_cases

    buf = BytesIO()

    def canvas_maker(*args, **kwargs):
        return PageCanvas(*args, hospital=hospital, district=district, **kwargs)

    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=LEFT_M, rightMargin=RIGHT_M,
        topMargin=TOP_M,   bottomMargin=BOT_M,
    )
    frame = Frame(
        LEFT_M, BOT_M,
        PAGE_W - LEFT_M - RIGHT_M,
        PAGE_H - TOP_M - BOT_M,
        id="main",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])

    story = []
    story.extend(_build_title_block(data))
    story.append(HR())
    story.extend(_build_kpi_row(data))
    story.append(HR())
    story.extend(_build_alert_pill(data))
    story.append(HR())
    story.extend(_build_ehr_summary(uploaded_cases))
    story.append(HR())
    story.extend(_build_shap_section(data.get("shap", {})))
    story.append(HR())
    story.extend(_build_timeline(data.get("timeline", [])))
    story.append(HR())
    story.extend(_build_methodology())
    story.extend(_build_footer())

    doc.build(story, canvasmaker=canvas_maker)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────
#  Standalone demo (python client/report_generator.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os, pathlib
    sample_data = {
        "hospital":     "Chennai Metro General Hospital",
        "district":     "Chennai Urban District",
        "censuscode":   "IN-TN-CHN-042",
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC+5:30"),
        "status": {"alert_level": "HIGH", "outbreak_prob": 78.4, "pred_cases": 142},
        "shap": {"feature_importance": [
            {"feature": "PM2.5 Concentration",     "importance":  0.3812},
            {"feature": "Ambient Humidity (%)",    "importance":  0.2945},
            {"feature": "Stagnant Water Index",    "importance":  0.2301},
            {"feature": "Dengue Case History 30d", "importance":  0.1987},
            {"feature": "Average Temperature",     "importance":  0.1654},
            {"feature": "CO2 Level (ppm)",         "importance":  0.0923},
            {"feature": "Rainfall (mm)",            "importance":  0.0811},
            {"feature": "Population Density",      "importance":  0.0702},
            {"feature": "Vaccination Coverage",    "importance": -0.1234},
            {"feature": "Wind Speed",              "importance": -0.0654},
        ]},
        "timeline": [
            {"year": 2025, "week": 20, "actual_cases": 18,  "pred_cases": 21,  "prob": 0.31},
            {"year": 2025, "week": 21, "actual_cases": 27,  "pred_cases": 24,  "prob": 0.42},
            {"year": 2025, "week": 22, "actual_cases": 35,  "pred_cases": 38,  "prob": 0.55},
            {"year": 2025, "week": 23, "actual_cases": 52,  "pred_cases": 49,  "prob": 0.64},
            {"year": 2025, "week": 24, "actual_cases": 71,  "pred_cases": 74,  "prob": 0.72},
            {"year": 2025, "week": 25, "actual_cases": 98,  "pred_cases": 103, "prob": 0.81},
            {"year": 2025, "week": 26, "actual_cases":  0,  "pred_cases": 142, "prob": 0.78},
        ],
    }
    sample_cases = [
        {"filename": f"EHR_TN_{1000+i:04d}.json",
         "temperature_c": 37.2 + (i % 5) * 0.3,
         "dengue_status": 1 if i % 3 == 0 else 0}
        for i in range(18)
    ]

    out = pathlib.Path(__file__).parent.parent / "outputs" / "demo_risk_report.pdf"
    out.parent.mkdir(exist_ok=True)
    pdf_buf = generate_report(sample_data, sample_cases)
    out.write_bytes(pdf_buf.read())
    print(f"[OK] Demo PDF written -> {out}")
