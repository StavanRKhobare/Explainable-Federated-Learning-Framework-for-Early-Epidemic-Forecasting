"""Patches client_app.py: replaces /api/download-report with report_generator.generate_report"""
import pathlib, sys

TARGET = pathlib.Path(__file__).parent.parent / "client" / "client_app.py"
text   = TARGET.read_bytes().decode("utf-8")

START = '@app.get("/api/download-report")'
END   = '@app.post("/api/run-fl")'

si = text.find(START)
ei = text.find(END)
if si == -1 or ei == -1:
    sys.exit("ERROR: markers not found.")

NEW = '''@app.get("/api/download-report")
def download_report():
    """Generate and stream a premium PDF report using the AeroSmart/EpiGraph AI design."""
    import datetime
    from fastapi.responses import StreamingResponse
    from client.report_generator import generate_report

    try:
        ep  = f"{CLIENT_CONFIG['server_url']}/api/report-data/{CLIENT_CONFIG['censuscode']}"
        res = requests.get(ep, timeout=8)
        if res.status_code != 200:
            raise HTTPException(status_code=500, detail="Could not fetch report data")
        data = res.json()

        # Enrich data with hospital name for the PDF header
        data["hospital"] = CLIENT_CONFIG["name"]

        # Normalise uploaded_cases: convert int dengue_status to string for the renderer
        cases_for_pdf = [
            {
                "filename":      e.get("filename", ""),
                "temperature_c": e.get("temperature_c", 0),
                "dengue_status": (
                    "Positive" if e.get("dengue_status") == 1
                    else "Negative" if e.get("dengue_status") == 0
                    else "Unknown"
                ),
            }
            for e in uploaded_cases
        ]

        buf = generate_report(data, cases_for_pdf)

        district  = data.get("district", "District").replace(" ", "_").replace("/", "-")
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M")
        filename  = f"FedXGNN_EpiReport_{district}_{timestamp}.pdf"

        return StreamingResponse(
            buf,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

'''

patched = text[:si] + NEW + "\n" + text[ei:]
TARGET.write_bytes(patched.encode("utf-8"))
print("[OK] client_app.py patched to use report_generator.generate_report")
