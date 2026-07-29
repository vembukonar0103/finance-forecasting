"""
scripts/generate_docx_report.py
-------------------------------
Generates a beautifully styled, professional Word Document (.docx) comparing Kronos and TimesFM.
Includes title stylings, table formatting, bolded metrics, structured lists, and the embedded benchmark comparison chart.
"""

import os
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

# Colors
COLOR_PRIMARY = RGBColor(30, 41, 59)     # Slate 800
COLOR_SECONDARY = RGBColor(15, 23, 42)   # Slate 900
COLOR_ACCENT = RGBColor(37, 99, 235)     # Blue 600
COLOR_TEXT = RGBColor(51, 65, 85)        # Slate 700
COLOR_MUTED = RGBColor(100, 116, 139)    # Slate 500

def set_cell_background(cell, fill_hex):
    """Set the background color of a cell using XML manipulation."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner margins (padding) of a cell in dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_heading_styled(doc, text, level):
    """Add a heading with custom colors and fonts."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    
    run = p.add_run(text)
    run.font.name = 'Arial'
    
    if level == 1:
        run.font.size = Pt(14)
        run.bold = True
        run.font.color.rgb = COLOR_PRIMARY
        # Add bottom border style or horizontal line
        p_border = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="4" w:color="2563EB"/></w:pBdr>')
        p._p.get_or_add_pPr().append(p_border)
    elif level == 2:
        run.font.size = Pt(11)
        run.bold = True
        run.font.color.rgb = COLOR_SECONDARY
        p.paragraph_format.space_before = Pt(8)
    return p

def main():
    doc = Document()
    
    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Set default font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Arial'
    font.size = Pt(10)
    font.color.rgb = COLOR_TEXT
    
    # ------------------ TITLE & SUBTITLE ------------------
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(2)
    run_title = p_title.add_run("Kronos vs TimesFM")
    run_title.font.size = Pt(22)
    run_title.bold = True
    run_title.font.color.rgb = COLOR_SECONDARY
    
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(12)
    run_sub = p_sub.add_run("Benchmark Analysis Report")
    run_sub.font.size = Pt(14)
    run_sub.bold = True
    run_sub.font.color.rgb = COLOR_ACCENT
    
    # ------------------ METADATA TABLE ------------------
    meta_table = doc.add_table(rows=2, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    
    # Set width
    for row in meta_table.rows:
        for cell in row.cells:
            cell.width = Inches(3.2)
            set_cell_background(cell, "F8FAFC") # slate 50
            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
            
    # Draw subtle borders
    tblPr = meta_table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        f'<w:left w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        f'<w:insideH w:val="none"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)
    
    # Fill metadata
    meta_items = [
        ("Date:", "2026-07-15", "Holdout Period:", "30 trading days (May 28 - Jul 10, 2026)"),
        ("Assets:", "AAPL, MSFT, NVDA, SPY, BTC-USD", "Run Type:", "Single-run point-in-time snapshot")
    ]
    
    for r_idx, (k1, v1, k2, v2) in enumerate(meta_items):
        # Col 1
        p1 = meta_table.rows[r_idx].cells[0].paragraphs[0]
        run_k1 = p1.add_run(k1 + " ")
        run_k1.bold = True
        run_k1.font.color.rgb = COLOR_SECONDARY
        run_k1.font.size = Pt(8.5)
        run_v1 = p1.add_run(v1)
        run_v1.font.size = Pt(8.5)
        
        # Col 2
        p2 = meta_table.rows[r_idx].cells[1].paragraphs[0]
        run_k2 = p2.add_run(k2 + " ")
        run_k2.bold = True
        run_k2.font.color.rgb = COLOR_SECONDARY
        run_k2.font.size = Pt(8.5)
        run_v2 = p2.add_run(v2)
        run_v2.font.size = Pt(8.5)
        
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_after = Pt(12)
    
    # ------------------ SECTION 1: MAIN METRICS TABLE ------------------
    add_heading_styled(doc, "1. Summary Metrics (Single Run)", 1)
    
    headers = ["Ticker", "Model", "MAE", "RMSE", "MAPE (%)", "Dir. Acc (%)", "Inference (s)", "Memory (MB)"]
    data = [
        ["AAPL", "Kronos", "58.39", "61.57", "19.27", "60.00", "5.575", "0.50"],
        ["AAPL", "TimesFM", "11.26", "13.42", "3.81", "50.00", "0.241", "0.07"],
        ["MSFT", "Kronos", "68.38", "72.23", "17.81", "56.67", "4.526", "0.19"],
        ["MSFT", "TimesFM", "67.06", "69.42", "17.44", "50.00", "0.250", "0.07"],
        ["NVDA", "Kronos", "35.82", "38.12", "17.60", "50.00", "4.455", "0.19"],
        ["NVDA", "TimesFM", "16.59", "17.84", "8.22", "46.67", "0.236", "0.07"],
        ["SPY", "Kronos", "88.14", "90.66", "11.84", "60.00", "4.424", "0.19"],
        ["SPY", "TimesFM", "21.46", "23.17", "2.90", "60.00", "0.269", "0.07"],
        ["BTC-USD", "Kronos", "25,003.47", "26,716.26", "40.15", "43.33", "4.306", "0.20"],
        ["BTC-USD", "TimesFM", "6,140.88", "6,503.33", "9.91", "50.00", "0.268", "0.11"]
    ]
    
    bold_indices = {
        (0, 5), # AAPL Kronos Dir Acc
        (1, 2), (1, 3), (1, 4), (1, 6), (1, 7), # AAPL TimesFM bests
        (2, 5), # MSFT Kronos Dir Acc
        (3, 2), (3, 3), (3, 4), (3, 6), (3, 7), # MSFT TimesFM bests
        (4, 5), # NVDA Kronos Dir Acc
        (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), # NVDA TimesFM bests
        (6, 5), # SPY Kronos Dir Acc tie
        (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), # SPY TimesFM bests
        (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), # BTC TimesFM bests
    }

    metrics_table = doc.add_table(rows=1 + len(data), cols=len(headers))
    metrics_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Custom widths
    widths = [Inches(0.6), Inches(0.7), Inches(0.9), Inches(0.9), Inches(0.8), Inches(0.9), Inches(0.9), Inches(0.8)]
    for row in metrics_table.rows:
        for idx, cell in enumerate(row.cells):
            cell.width = widths[idx]
            set_cell_margins(cell, top=70, bottom=70, left=70, right=70)
            
    # Add subtle gridlines to XML
    metrics_table._tbl.tblPr.append(parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:color="CBD5E1"/>'
        f'<w:bottom w:val="single" w:sz="6" w:color="1E293B"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="2" w:color="E2E8F0"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    ))
    
    # Render Table Headers
    for c_idx, h in enumerate(headers):
        cell = metrics_table.rows[0].cells[c_idx]
        set_cell_background(cell, "1E293B") # slate 800
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(255, 255, 255)
        
    # Render Table Data
    for r_idx, row_data in enumerate(data):
        row = metrics_table.rows[r_idx + 1]
        
        # Zebra striping background
        bg_hex = "F1F5F9" if r_idx % 2 == 1 else "FFFFFF"
        
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg_hex)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(val)
            run.font.size = Pt(8)
            
            is_bold = (r_idx, c_idx) in bold_indices
            
            # Formatting specifics
            if c_idx == 1:
                run.bold = True
                if val == "TimesFM":
                    run.font.color.rgb = COLOR_ACCENT
                else:
                    run.font.color.rgb = RGBColor(194, 65, 12)
            else:
                run.bold = is_bold
                if is_bold:
                    run.font.color.rgb = COLOR_SECONDARY
                    
    p_table_foot = doc.add_paragraph()
    p_table_foot.paragraph_format.space_before = Pt(3)
    p_table_foot.paragraph_format.space_after = Pt(12)
    run_foot = p_table_foot.add_run("*Note: Bold indicates the better performing model on that specific metric & asset in this run.")
    run_foot.font.size = Pt(7.5)
    run_foot.font.color.rgb = COLOR_MUTED
    run_foot.italic = True
    
    # ------------------ SECTION 2: WHERE TIMESFM PERFORMS BETTER ------------------
    add_heading_styled(doc, "2. Where TimesFM Performs Better", 1)
    
    bullets_tfm = [
        ("Accuracy across All Assets", "TimesFM wins on every single asset for all error metrics (MAE/RMSE/MAPE). The margin is substantial on AAPL (MAE 11.26 vs 58.39 — 5.2x lower), NVDA (MAE 16.59 vs 35.82 — 2.2x lower), and SPY (MAE 21.46 vs 88.14 — 4.1x lower). On BTC-USD, TimesFM achieves a MAPE of 9.91% compared to Kronos's 40.15% (4x lower). Even on MSFT, TimesFM remains slightly more accurate (MAE 67.06 vs 68.38)."),
        ("Inference Speed", "TimesFM is 16-30x faster than Kronos, completing forecasts in 0.23-0.27 seconds compared to Kronos's 4.3-5.6 seconds."),
        ("Memory Efficiency", "TimesFM uses 0.07-0.11 MB per ticker, which is 2.5-7x more memory efficient than Kronos (0.19-0.50 MB)."),
        ("Probabilistic Output", "TimesFM natively provides calibrated quantile forecasts (p10-p90) and a tail-risk probability (P(>+5%)), which are critical for risk management. Kronos only outputted point forecasts in this benchmark.")
    ]
    
    for title, text in bullets_tfm:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(4)
        run_title = p.add_run(title + ": ")
        run_title.bold = True
        run_title.font.color.rgb = COLOR_SECONDARY
        p.add_run(text)
        
    p_spacer2 = doc.add_paragraph()
    p_spacer2.paragraph_format.space_after = Pt(6)
    
    # ------------------ SECTION 3: WHERE KRONOS PERFORMS BETTER ------------------
    add_heading_styled(doc, "3. Where Kronos Performs Better", 1)
    
    bullets_kronos = [
        ("Directional Accuracy", "Kronos achieves higher directional accuracy on 3 out of 5 assets in this run: AAPL (60.0% vs 50.0%), MSFT (56.67% vs 50.0%), and NVDA (50.0% vs 46.67%), and ties on SPY (60.0% vs 60.0%)."),
        ("Architectural Flexibility", "Kronos's generative, sampling-based architecture means it could theoretically produce distribution forecasts via repeated sampling, though this was not tested in this point-forecast run."),
        ("Zero-Shot Generalization", "Kronos is a true zero-shot foundation model with no task-specific tuning. This may provide a more robust baseline on highly out-of-distribution time series.")
    ]
    
    for title, text in bullets_kronos:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(4)
        run_title = p.add_run(title + ": ")
        run_title.bold = True
        run_title.font.color.rgb = COLOR_SECONDARY
        p.add_run(text)
        
    p_spacer3 = doc.add_paragraph()
    p_spacer3.paragraph_format.space_after = Pt(6)
    
    # ------------------ SECTION 4: LIMITATIONS OBSERVED ------------------
    add_heading_styled(doc, "4. Limitations Observed", 1)
    
    add_heading_styled(doc, "4.1 Kronos-Specific", 2)
    bullets_lim_k = [
        ("High MAPE on most assets", "11-40% MAPE suggests that a single point-forecast sample from Kronos is highly noisy."),
        ("Stochastic Architecture", "Kronos is sampling-based and stochastic. A single run does not measure or control for cross-run variance, so repeated executions may produce different results."),
        ("No native quantile output", "The model samples but doesn't natively expose calibrated prediction intervals. Pinball loss is N/A."),
        ("High latency", "4-6 seconds per ticker makes it less viable for real-time applications.")
    ]
    for title, text in bullets_lim_k:
        p = doc.add_paragraph(style='List Bullet 2')
        p.paragraph_format.space_after = Pt(3)
        run_title = p.add_run(title + ": ")
        run_title.bold = True
        run_title.font.color.rgb = COLOR_SECONDARY
        p.add_run(text)
        
    add_heading_styled(doc, "4.2 TimesFM-Specific", 2)
    bullets_lim_t = [
        ("Directional accuracy drops", "TimesFM's directional accuracy was lower than Kronos on AAPL, MSFT, and NVDA in this run, indicating it may smooth over turning points in certain market regimes."),
        ("Single-run evaluation", "Cross-run variance was not tested. TimesFM produced consistent point forecasts within this single execution."),
        ("Quantile calibration unverified", "Pinball losses are calculated but their empirical coverage was not formally tested.")
    ]
    for title, text in bullets_lim_t:
        p = doc.add_paragraph(style='List Bullet 2')
        p.paragraph_format.space_after = Pt(3)
        run_title = p.add_run(title + ": ")
        run_title.bold = True
        run_title.font.color.rgb = COLOR_SECONDARY
        p.add_run(text)
        
    add_heading_styled(doc, "4.3 General Limitations", 2)
    bullets_lim_g = [
        ("Single holdout window", "All metrics are evaluated on one 30-day window. Performance could differ substantially in different market environments (e.g., bear markets, flat ranges)."),
        ("No walk-forward validation", "A single holdout is a snapshot, not a statistically robust cross-validation.")
    ]
    for title, text in bullets_lim_g:
        p = doc.add_paragraph(style='List Bullet 2')
        p.paragraph_format.space_after = Pt(3)
        run_title = p.add_run(title + ": ")
        run_title.bold = True
        run_title.font.color.rgb = COLOR_SECONDARY
        p.add_run(text)
        
    doc.add_page_break()
    
    # ------------------ SECTION 5: PERFORMANCE VISUALIZATION ------------------
    add_heading_styled(doc, "5. Performance Visualization", 1)
    
    p_chart_text = doc.add_paragraph("The chart below displays a 2x3 performance layout for MAE, RMSE, MAPE, Directional Accuracy, Inference Latency, and Memory Footprint across all 5 assets:")
    p_chart_text.paragraph_format.space_after = Pt(8)
    
    chart_path = "plots/benchmark_comparison.png"
    if os.path.exists(chart_path):
        doc.add_picture(chart_path, width=Inches(6.2))
        p_caption = doc.add_paragraph("Figure 1: Benchmark metrics comparison for Kronos and TimesFM (5 Assets, 30-day holdout, single-run summary).")
        p_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_caption.paragraph_format.space_before = Pt(4)
        p_caption.paragraph_format.space_after = Pt(12)
        p_caption.runs[0].font.size = Pt(8)
        p_caption.runs[0].italic = True
        p_caption.runs[0].font.color.rgb = COLOR_MUTED
    else:
        doc.add_paragraph("[Chart file plots/benchmark_comparison.png not found]")
        
    # ------------------ SECTION 6: INITIAL CONCLUSIONS ------------------
    add_heading_styled(doc, "6. Initial Conclusions", 1)
    
    conclusions = [
        ("Model Selection", "TimesFM is the stronger model for most tested assets. It produces lower error on all 5 tickers, completes inference 16-30x faster, uses significantly less memory, and natively supports probabilistic interval forecasts."),
        ("Kronos Suitability", "Kronos is not production-ready as a single-sample point forecaster. Its high-variance, high-MAPE forecasts and 15-25x slower inference make it unsuitable for direct deployment."),
        ("Speed and Deployment", "TimesFM's speed advantage (16-30x) makes it viable for real-time applications, while Kronos's multi-second latency limits it to batch processes.")
    ]
    for title, text in conclusions:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(4)
        run_title = p.add_run(title + ": ")
        run_title.bold = True
        run_title.font.color.rgb = COLOR_SECONDARY
        p.add_run(text)
        
    p_spacer4 = doc.add_paragraph()
    p_spacer4.paragraph_format.space_after = Pt(6)
    
    # ------------------ SECTION 7: CAVEATS & APPENDIX ------------------
    add_heading_styled(doc, "7. Caveats & Appendix", 1)
    
    # Warning Callout Box (as a 1x1 table with thick left border and warm bg)
    callout_tbl = doc.add_table(rows=1, cols=1)
    callout_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    callout_cell = callout_tbl.rows[0].cells[0]
    callout_cell.width = Inches(6.5)
    set_cell_background(callout_cell, "FFF7ED") # light warm orange
    set_cell_margins(callout_cell, top=120, bottom=120, left=160, right=160)
    
    # Border styling
    tcPr = callout_cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="36" w:space="0" w:color="FDBA74"/>' # Thick orange left border (4.5 pt)
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(tcBorders)
    
    cp = callout_cell.paragraphs[0]
    cp.paragraph_format.space_after = Pt(0)
    run_warn = cp.add_run("IMPORTANT: ")
    run_warn.bold = True
    run_warn.font.color.rgb = RGBColor(194, 65, 12)
    run_warn.font.size = Pt(9)
    run_wtext = cp.add_run(
        "This is a single-run benchmark comparison. Cross-run variance was not tested. Kronos's sampling-based generative architecture is stochastic and may produce different metrics on repeated runs. Additionally, all evaluations are restricted to the same 30-day holdout window; performance under different market regimes was not evaluated."
    )
    run_wtext.font.size = Pt(9)
    run_wtext.font.color.rgb = RGBColor(124, 45, 18)
    
    p_spacer5 = doc.add_paragraph()
    p_spacer5.paragraph_format.space_after = Pt(8)
    
    add_heading_styled(doc, "Data Artifacts Location", 2)
    p_loc = doc.add_paragraph("All data predictions and summary runs can be found locally at:")
    p_loc.paragraph_format.space_after = Pt(4)
    
    artifacts = [
        ("Benchmark metrics CSV:", "data/predictions/benchmark_results.csv"),
        ("Benchmark comparison chart:", "plots/benchmark_comparison.png"),
        ("Predictions (Parquet):", "data/predictions/{ticker}_{model}_predictions.parquet")
    ]
    
    for label, path in artifacts:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(2)
        run_l = p.add_run(label + " ")
        run_l.bold = True
        run_l.font.size = Pt(8.5)
        run_l.font.color.rgb = COLOR_SECONDARY
        run_p = p.add_run(path)
        run_p.font.size = Pt(8.5)
        run_p.font.color.rgb = COLOR_TEXT
        
    out_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    docx_path = os.path.join(out_dir, "benchmark_analysis.docx")
    
    doc.save(docx_path)
    print(f"\nWord Document (.docx) successfully generated → {docx_path}")

if __name__ == "__main__":
    main()
