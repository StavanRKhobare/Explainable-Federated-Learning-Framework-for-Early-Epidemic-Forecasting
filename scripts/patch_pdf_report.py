"""
Replaces the /api/download-report endpoint in client_app.py
with a premium canvas-based PDF generator.
Run once from repo root:
    python scripts/patch_pdf_report.py
"""
import re, pathlib, sys

TARGET = pathlib.Path(__file__).parent.parent / "client" / "client_app.py"
text   = TARGET.read_bytes().decode("utf-8")

# ─── Locate the function boundaries ──────────────────────────────────────────
START_MARKER = '@app.get("/api/download-report")'
END_MARKER   = '@app.post("/api/run-fl")'

si = text.find(START_MARKER)
ei = text.find(END_MARKER)
if si == -1 or ei == -1:
    sys.exit("Markers not found — already patched or file changed.")

NEW_FUNCTION = r'''@app.get("/api/download-report")
def download_report():
    """Generate and stream a premium, canvas-based multi-section PDF report."""
    import io, math
    from datetime import datetime
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether
    )
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.platypus.flowables import Flowable

    # --- colour palette --------------------------------------------------
    NAVY      = colors.HexColor("#0f172a")
    BLUE_700  = colors.HexColor("#1d4ed8")
    BLUE_500  = colors.HexColor("#3b82f6")
    BLUE_100  = colors.HexColor("#dbeafe")
    BLUE_50   = colors.HexColor("#eff6ff")
    RED_700   = colors.HexColor("#b91c1c")
    RED_100   = colors.HexColor("#fee2e2")
    AMBER_700 = colors.HexColor("#b45309")
    AMBER_100 = colors.HexColor("#fef3c7")
    GREEN_700 = colors.HexColor("#15803d")
    GREEN_100 = colors.HexColor("#dcfce7")
    SLATE_800 = colors.HexColor("#1e293b")
    SLATE_600 = colors.HexColor("#475569")
    SLATE_400 = colors.HexColor("#94a3b8")
    SLATE_200 = colors.HexColor("#e2e8f0")
    SLATE_100 = colors.HexColor("#f1f5f9")
    SLATE_50  = colors.HexColor("#f8fafc")
    WHITE     = colors.white

    def alert_colors(level):
        return {"HIGH": (RED_700, RED_100), "MEDIUM": (AMBER_700, AMBER_100),
                "LOW":  (GREEN_700, GREEN_100)}.get(level, (GREEN_700, GREEN_100))

    # --- custom flowable: solid progress bar --------------------------------
    class ProgressBar(Flowable):
        def __init__(self, value, max_val, width, height=11,
                     color=BLUE_500, bg=SLATE_200, label=""):
            super().__init__()
            self.value = value; self.max_val = max_val or 1
            self.width = width; self.height = height
            self.color = color; self.bg = bg; self.label = label
        def draw(self):
            c = self.canv; r = self.height / 2
            c.setFillColor(self.bg)
            c.roundRect(0, 0, self.width, self.height, r, fill=1, stroke=0)
            fw = max(r*2, (self.value / self.max_val) * self.width)
            c.setFillColor(self.color)
            c.roundRect(0, 0, fw, self.height, r, fill=1, stroke=0)
            if self.label:
                c.setFillColor(WHITE); c.setFont("Helvetica-Bold", 6.5)
                c.drawString(7, 2.5, self.label)
        def wrap(self, *args): return self.width, self.height + 2

    # --- canvas subclass: running header + footer on every page -------------
    class PageCanvas(rl_canvas.Canvas):
        def __init__(self, *args, hospital="", district="", **kwargs):
            super().__init__(*args, **kwargs)
            self._hospital = hospital; self._district = district; self._pg = 0
        def showPage(self):
            self._pg += 1; self._chrome(); super().showPage()
        def save(self):
            self._pg += 1; self._chrome(); super().save()
        def _chrome(self):
            W, H = A4
            self.setFillColor(NAVY)
            self.rect(0, H - 28, W, 28, fill=1, stroke=0)
            self.setFont("Helvetica-Bold", 8); self.setFillColor(WHITE)
            self.drawString(20, H - 18, "FedXGNN  Epidemic Intelligence Platform")
            self.setFont("Helvetica", 7)
            self.setFillColor(colors.HexColor("#93c5fd"))
            self.drawRightString(W - 20, H - 18,
                                 f"{self._hospital}  |  {self._district}")
            self.setFillColor(SLATE_100)
            self.rect(0, 0, W, 22, fill=1, stroke=0)
            self.setStrokeColor(SLATE_200); self.setLineWidth(0.5)
            self.line(0, 22, W, 22)
            self.setFont("Helvetica", 7); self.setFillColor(SLATE_600)
            self.drawString(20, 7, "CONFIDENTIAL  |  For Clinical & Research Use Only")
            self.drawRightString(W - 20, 7, f"Page {self._pg}")

    try:
        # fetch live data
        ep = f"{CLIENT_CONFIG['server_url']}/api/report-data/{CLIENT_CONFIG['censuscode']}"
        res = requests.get(ep, timeout=8)
        if res.status_code != 200:
            raise HTTPException(500, "Could not fetch report data from central server")
        data = res.json()

        district     = data.get("district", "Unknown")
        censuscode   = data.get("censuscode", CLIENT_CONFIG["censuscode"])
        generated_at = data.get("generated_at", datetime.utcnow().isoformat())
        status_d     = data.get("status", {})
        shap_data    = data.get("shap", {})
        timeline     = data.get("timeline", [])
        alert_level  = status_d.get("alert_level", "LOW")
        outbreak_prob= status_d.get("outbreak_prob", 0.0)
        pred_cases   = status_d.get("pred_cases", 0.0)
        alert_fg, alert_bg = alert_colors(alert_level)

        # --- styles ----------------------------------------------------------
        base = getSampleStyleSheet()
        def S(n, **kw): return ParagraphStyle(n, parent=base["Normal"], **kw)
        H1    = S("h1", fontSize=20, fontName="Helvetica-Bold", textColor=NAVY,
                   leading=26, spaceAfter=4)
        H2    = S("h2", fontSize=12, fontName="Helvetica-Bold", textColor=BLUE_700,
                   leading=16, spaceBefore=14, spaceAfter=6)
        BODY  = S("bd", fontSize=8.5, textColor=SLATE_800, leading=13)
        SMALL = S("sm", fontSize=7.5, textColor=SLATE_600, leading=11)
        CH    = S("ch", fontSize=8,  fontName="Helvetica-Bold", textColor=WHITE,
                   leading=12, alignment=TA_CENTER)
        CR    = S("cr", fontSize=8,  textColor=SLATE_800, leading=12, alignment=TA_CENTER)
        CL    = S("cl", fontSize=8,  textColor=SLATE_800, leading=12)
        AH    = S("ah", fontSize=15, fontName="Helvetica-Bold", textColor=alert_fg,
                   leading=20, alignment=TA_CENTER)
        AB    = S("ab", fontSize=8.5, textColor=alert_fg, leading=13, alignment=TA_CENTER)
        FT    = S("ft", fontSize=7,  textColor=SLATE_400, leading=10, alignment=TA_CENTER)

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=28 + 1.2*cm, bottomMargin=22 + 1.0*cm,
            title=f"FedXGNN Risk Report  {district}",
            author="FedXGNN Epidemic Platform",
        )
        W = doc.width
        story = []

        # =====================================================================
        # 1. TITLE BLOCK
        # =====================================================================
        t_rows = [[Paragraph("Epidemic Risk Intelligence Report", H1)],
                  [Paragraph(CLIENT_CONFIG["name"],
                             S("sn", fontSize=11, fontName="Helvetica-Bold",
                               textColor=BLUE_500, leading=15))],
                  [Paragraph(f"District: <b>{district}</b>  |  Census: <b>{censuscode}</b>"
                             f"  |  {generated_at[:19].replace('T','  ')}  UTC", SMALL)]]
        t_tbl = Table(t_rows, colWidths=[W])
        t_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), NAVY),
            ("ROUNDEDCORNERS",(0,0),(-1,-1), [8]),
            ("TOPPADDING",    (0,0),(-1, 0), 22),
            ("BOTTOMPADDING", (0,-1),(-1,-1), 18),
            ("LEFTPADDING",   (0,0),(-1,-1), 18),
            ("RIGHTPADDING",  (0,0),(-1,-1), 18),
            ("ROWPADDING",    (0,0),(-1,-1), 5),
        ]))
        story.append(t_tbl); story.append(Spacer(1, .5*cm))

        # =====================================================================
        # 2. KPI CARDS + ALERT PILL
        # =====================================================================
        story.append(Paragraph("Risk Overview", H2))
        card_sty = TableStyle([
            ("BACKGROUND",     (0,0),(-1,-1), SLATE_50),
            ("BOX",            (0,0),(-1,-1), 0.8, SLATE_200),
            ("ROUNDEDCORNERS", (0,0),(-1,-1), [6]),
            ("TOPPADDING",     (0,0),(-1,-1), 12),
            ("BOTTOMPADDING",  (0,0),(-1,-1), 12),
            ("LEFTPADDING",    (0,0),(-1,-1), 12),
            ("RIGHTPADDING",   (0,0),(-1,-1), 12),
            ("ALIGN",          (0,0),(-1,-1), "CENTER"),
            ("VALIGN",         (0,0),(-1,-1), "MIDDLE"),
        ])
        cw = W / 3
        def kpi_card(val, lbl, col):
            c = Table([[
                Paragraph(val, S("kv", fontSize=24, fontName="Helvetica-Bold",
                                  textColor=col, leading=28, alignment=TA_CENTER)),
                Paragraph(lbl, S("kl", fontSize=7.5, textColor=SLATE_600,
                                  leading=11, alignment=TA_CENTER)),
            ]], colWidths=[cw - 0.5*cm])
            c.setStyle(card_sty); return c

        c1 = kpi_card(f"{outbreak_prob*100:.1f}%", "Outbreak Probability", alert_fg)
        c2 = kpi_card(f"{pred_cases:.0f}",          "Predicted Cases",      BLUE_700)
        c3 = kpi_card(str(len(uploaded_cases)),      "EHR Records",         SLATE_800)

        kpi_row = Table([[c1, c2, c3]], colWidths=[cw, cw, cw], hAlign="LEFT")
        kpi_row.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),4),
                                      ("RIGHTPADDING",(0,0),(-1,-1),4)]))
        story.append(kpi_row); story.append(Spacer(1, .3*cm))

        # alert pill
        pill = Table([[Paragraph(
            f"ALERT LEVEL: {alert_level}  |  "
            f"Outbreak Probability {outbreak_prob*100:.1f}%  |  "
            f"Predicted {pred_cases:.0f} cases next week",
            AH if alert_level == "HIGH" else AB
        )]], colWidths=[W])
        pill.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(-1,-1), alert_bg),
            ("BOX",            (0,0),(-1,-1), 1.5, alert_fg),
            ("ROUNDEDCORNERS", (0,0),(-1,-1), [8]),
            ("TOPPADDING",     (0,0),(-1,-1), 10),
            ("BOTTOMPADDING",  (0,0),(-1,-1), 10),
        ]))
        story.append(pill); story.append(Spacer(1, .5*cm))

        # =====================================================================
        # 3. EHR PATIENT SUMMARY
        # =====================================================================
        story.append(HRFlowable(width="100%", thickness=0.5, color=SLATE_200, spaceAfter=2))
        story.append(Paragraph("Clinical EHR Patient Summary", H2))

        if uploaded_cases:
            n_pos    = sum(1 for e in uploaded_cases if e.get("dengue_status") == 1)
            n_neg    = sum(1 for e in uploaded_cases if e.get("dengue_status") == 0)
            avg_temp = sum(e.get("temperature_c", 0) for e in uploaded_cases) / len(uploaded_cases)
            pos_rate = n_pos / len(uploaded_cases) * 100

            stat_cs  = [W/5]*5
            sc_sty   = TableStyle([
                ("BACKGROUND",(0,0),(-1,-1),SLATE_50),
                ("BOX",(0,0),(-1,-1),0.6,SLATE_200),
                ("ROUNDEDCORNERS",(0,0),(-1,-1),[5]),
                ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
                ("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ])
            def sc(val, lbl, col):
                t = Table([[
                    Paragraph(str(val), S("sv", fontSize=18, fontName="Helvetica-Bold",
                                          textColor=col, leading=22, alignment=TA_CENTER)),
                    Paragraph(lbl, S("sl", fontSize=7, textColor=SLATE_600,
                                     leading=10, alignment=TA_CENTER)),
                ]], colWidths=[W/5 - 4])
                t.setStyle(sc_sty); return t

            sr = Table([[
                sc(len(uploaded_cases), "Total Records",   BLUE_700),
                sc(n_pos,               "Dengue Positive", RED_700),
                sc(n_neg,               "Dengue Negative", GREEN_700),
                sc(f"{pos_rate:.1f}%",  "Positivity Rate", AMBER_700),
                sc(f"{avg_temp:.1f}C",  "Avg Temp",        SLATE_800),
            ]], colWidths=stat_cs, hAlign="LEFT")
            sr.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),3),
                                    ("RIGHTPADDING",(0,0),(-1,-1),3)]))
            story.append(sr); story.append(Spacer(1, .3*cm))

            bar_col = RED_700 if pos_rate>50 else AMBER_700 if pos_rate>25 else GREEN_700
            story.append(Paragraph("Dengue Positivity Rate", SMALL))
            story.append(ProgressBar(pos_rate, 100, W, height=14,
                                     color=bar_col, label=f"{pos_rate:.1f}%"))
            story.append(Spacer(1, .3*cm))

            hdr = [Paragraph(h, CH) for h in ["#", "Filename", "Temp (C)", "Status"]]
            rows = [hdr]
            for i, e in enumerate(uploaded_cases[:15], 1):
                ds = e.get("dengue_status")
                st_text  = "Positive" if ds==1 else "Negative" if ds==0 else "Unknown"
                st_color = RED_700 if ds==1 else (GREEN_700 if ds==0 else SLATE_400)
                rows.append([
                    Paragraph(str(i), CR),
                    Paragraph(e.get("filename","")[: 42], CL),
                    Paragraph(f"{e.get('temperature_c',0):.1f}", CR),
                    Paragraph(f"<font color='{st_color.hexval()}'><b>{st_text}</b></font>", CR),
                ])
            if len(uploaded_cases) > 15:
                rows.append([Paragraph("...", CR),
                              Paragraph(f"+{len(uploaded_cases)-15} more", SMALL),
                              Paragraph("",CR), Paragraph("",CR)])
            ehr_t = Table(rows, colWidths=[W*.07, W*.50, W*.17, W*.26])
            ehr_t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),WHITE),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,SLATE_50]),
                ("LINEBELOW",(0,0),(-1,-1),0.3,SLATE_200),
                ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
                ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ]))
            story.append(ehr_t)
        else:
            story.append(Paragraph("No EHR records uploaded to this node yet.", SMALL))

        story.append(Spacer(1, .5*cm))

        # =====================================================================
        # 4. SHAP FEATURE ATTRIBUTION
        # =====================================================================
        story.append(HRFlowable(width="100%", thickness=0.5, color=SLATE_200, spaceAfter=2))
        story.append(Paragraph("Explainability  -  SHAP Feature Attribution", H2))
        story.append(Paragraph(
            "KernelSHAP values show each feature's contribution. "
            "Red bars increase outbreak risk; green bars reduce it.", SMALL))
        story.append(Spacer(1, .2*cm))

        shap_features = shap_data.get("feature_importance", [])
        if shap_features:
            max_imp = max((abs(f.get("importance",0)) for f in shap_features), default=1) or 1
            shdr = [Paragraph(h, CH) for h in ["#", "Feature", "Score", "Attribution"]]
            srows = [shdr]
            for rank, f in enumerate(shap_features[:10], 1):
                imp = f.get("importance", 0)
                bc  = RED_700 if imp > 0 else GREEN_700
                srows.append([
                    Paragraph(f"#{rank}", CR),
                    Paragraph(f.get("feature",""), CL),
                    Paragraph(f"<b>{imp:+.4f}</b>", CR),
                    ProgressBar(abs(imp), max_imp, W*0.37, height=11, color=bc,
                                label=f"{'+ increases' if imp>0 else '- reduces'} risk"),
                ])
            s_tbl = Table(srows, colWidths=[W*.08, W*.28, W*.13, W*.39],
                          rowHeights=[20]+[18]*len(srows[1:]))
            s_tbl.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),WHITE),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,SLATE_50]),
                ("LINEBELOW",(0,0),(-1,-1),0.3,SLATE_200),
                ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
                ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ]))
            story.append(s_tbl)
        else:
            story.append(Paragraph("SHAP data not yet available. Transmit an embedding first.", SMALL))

        story.append(Spacer(1, .5*cm))

        # =====================================================================
        # 5. TIMELINE
        # =====================================================================
        story.append(HRFlowable(width="100%", thickness=0.5, color=SLATE_200, spaceAfter=2))
        story.append(Paragraph("Epidemiological Timeline  -  Last 10 Weeks", H2))
        if timeline:
            thdr = [Paragraph(h, CH) for h in
                    ["Week","Year","Actual","Predicted","Probability","Risk"]]
            trows = [thdr]
            for t in reversed(timeline[-10:]):
                prob = t.get("prob", 0)
                rl   = "HIGH" if prob>.5 else "MEDIUM" if prob>.25 else "LOW"
                rc   = RED_700 if rl=="HIGH" else AMBER_700 if rl=="MEDIUM" else GREEN_700
                trows.append([
                    Paragraph(f"W{t.get('week','?')}", CR),
                    Paragraph(str(t.get("year","?")), CR),
                    Paragraph(str(t.get("actual_cases","?")), CR),
                    Paragraph(f"{t.get('pred_cases',0):.1f}", CR),
                    Paragraph(f"{prob*100:.1f}%", CR),
                    Paragraph(f"<font color='{rc.hexval()}'><b>{rl}</b></font>", CR),
                ])
            tl_t = Table(trows, colWidths=[W*.09, W*.09, W*.16, W*.16, W*.18, W*.22])
            tl_t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),WHITE),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,SLATE_50]),
                ("LINEBELOW",(0,0),(-1,-1),0.3,SLATE_200),
                ("ALIGN",(0,0),(-1,-1),"CENTER"),
                ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
                ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ]))
            story.append(tl_t)
        else:
            story.append(Paragraph("No timeline data yet.", SMALL))

        story.append(Spacer(1, .6*cm))

        # =====================================================================
        # 6. METHODOLOGY + FOOTER
        # =====================================================================
        m_data = [[Paragraph(
            "<b>Methodology.</b>  FedXGNN is a split-federated spatio-temporal GNN. Each hospital "
            "processes EHR records locally via spaCy NLP, then transmits a 64-dim embedding to the "
            "central Graph Attention Network, which aggregates spatial risk across 284 Indian "
            "districts. Outbreak probability is produced by a dual-task sigmoid head trained on "
            "728 epidemic windows. SHAP attributions use KernelSHAP with a 15-sample historical "
            "baseline. Raw patient data never leaves the device.", SMALL)]]
        m_tbl = Table(m_data, colWidths=[W])
        m_tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),BLUE_50),
            ("BOX",(0,0),(-1,-1),0.8,BLUE_100),
            ("ROUNDEDCORNERS",(0,0),(-1,-1),[6]),
            ("LEFTPADDING",(0,0),(-1,-1),14),("RIGHTPADDING",(0,0),(-1,-1),14),
            ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
        ]))
        story.append(m_tbl); story.append(Spacer(1, .3*cm))
        story.append(Paragraph(
            f"(c) {datetime.utcnow().year} FedXGNN Epidemic Intelligence Platform  |  "
            f"{CLIENT_CONFIG['name']}  |  Confidential - For Clinical Use Only", FT))

        # build
        doc.build(story, canvasmaker=lambda *a, **kw: PageCanvas(
            *a, hospital=CLIENT_CONFIG["name"], district=district, **kw))
        buf.seek(0)
        safe_d = district.replace(" ","_").replace("/","-")
        fn = f"FedXGNN_RiskReport_{safe_d}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.pdf"
        return StreamingResponse(buf, media_type="application/pdf",
                                 headers={"Content-Disposition": f"attachment; filename={fn}"})

    except HTTPException: raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

'''

patched = text[:si] + NEW_FUNCTION + "\n" + text[ei:]
TARGET.write_bytes(patched.encode("utf-8"))
print("[OK] client_app.py patched successfully.")
