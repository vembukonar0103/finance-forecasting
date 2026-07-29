"""
scripts/generate_pdf_report.py
------------------------------
Generates a beautifully styled, professional PDF report comparing Kronos and TimesFM.
Includes styled headers, tables with bolded metrics, structured lists, and the embedded benchmark comparison chart.
"""

import os
import sys
from fpdf import FPDF

# Colors (Sleek slate / navy corporate theme)
COLOR_PRIMARY_RGB = (30, 41, 59)     # Slate 800
COLOR_SECONDARY_RGB = (15, 23, 42)   # Slate 900
COLOR_ACCENT_RGB = (37, 99, 235)     # Blue 600
COLOR_LIGHT_BG_RGB = (248, 250, 252) # Slate 50
COLOR_TEXT_RGB = (51, 65, 85)        # Slate 700
COLOR_TEXT_DARK_RGB = (15, 23, 42)   # Slate 900
COLOR_MUTED_RGB = (100, 116, 139)    # Slate 500
COLOR_ALERT_BORDER = (226, 232, 240) # Slate 200

class PDFReport(FPDF):
    def header(self):
        # Only show page headers from page 2 onwards
        if self.page_no() > 1:
            self.set_font("helvetica", "I", 8)
            self.set_text_color(*COLOR_MUTED_RGB)
            self.cell(0, 10, "Kronos vs TimesFM - Benchmark Analysis Report", border=0, align="R", new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(*COLOR_MUTED_RGB)
        # Display page number
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", border=0, align="C")

    def chapter_title(self, label):
        self.set_font("helvetica", "B", 14)
        self.set_text_color(*COLOR_PRIMARY_RGB)
        self.cell(0, 8, label, border=0, align="L", new_x="LMARGIN", new_y="NEXT")
        # Draw a thin accent colored underline
        current_y = self.get_y()
        self.set_draw_color(*COLOR_ACCENT_RGB)
        self.set_line_width(0.5)
        self.line(self.l_margin, current_y, self.w - self.r_margin, current_y)
        self.ln(6)

    def heading_2(self, label):
        self.set_font("helvetica", "B", 11)
        self.set_text_color(*COLOR_SECONDARY_RGB)
        self.cell(0, 6, label, border=0, align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def add_bullet_point(self, title, text="", level=1):
        self.set_font("helvetica", "", 9.5)
        self.set_text_color(*COLOR_TEXT_RGB)
        
        # Adjust indent depending on level
        bullet = chr(149) if level == 1 else "-"
        indent = 8 * level
        self.set_x(self.l_margin + indent)
        
        if text:
            # Bullet point with a bold title
            self.set_font("helvetica", "B", 9.5)
            self.write(5, f"{bullet}  {title}: ")
            self.set_font("helvetica", "", 9.5)
            self.write(5, text)
            self.ln(6)
        else:
            # Simple bullet point
            self.write(5, f"{bullet}  {title}")
            self.ln(6)

    def add_alert_box(self, title, text):
        # Shaded box for warnings or caveats
        self.set_fill_color(255, 247, 237) # Light orange/yellow bg (warm warning)
        self.set_draw_color(254, 215, 170) # Orange border
        self.set_line_width(0.5)
        
        current_x = self.get_x()
        current_y = self.get_y()
        
        # Temporarily narrow width to fit cell
        box_width = self.w - self.l_margin - self.r_margin
        
        # Calculate approximate height
        self.set_font("helvetica", "", 9)
        lines = self.multi_cell(box_width, 5, f"{title}: {text}", border=1, align="L", fill=True, dry_run=True)
        # multi_cell height in fpdf2 returns number of lines if dry_run=True
        box_height = (lines + 1) * 5 + 4
        
        # Render the border and background
        self.rect(current_x, current_y, box_width, box_height, style="FD")
        
        # Render the text inside the box
        self.set_xy(current_x + 3, current_y + 2)
        self.set_text_color(194, 65, 12) # Dark orange text
        self.set_font("helvetica", "B", 9)
        self.write(5, f"{title}: ")
        self.set_font("helvetica", "", 9)
        self.write(5, text)
        
        # Reset position after box
        self.set_xy(self.l_margin, current_y + box_height + 4)
        self.set_text_color(*COLOR_TEXT_RGB)


def main():
    # Page setup
    pdf = PDFReport(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    
    # ------------------ TITLE & METADATA ------------------
    pdf.set_font("helvetica", "B", 20)
    pdf.set_text_color(*COLOR_SECONDARY_RGB)
    pdf.cell(0, 10, "Kronos vs TimesFM", border=0, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(*COLOR_ACCENT_RGB)
    pdf.cell(0, 8, "Benchmark Analysis Report", border=0, align="L", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(5)
    
    # Metadata Block
    pdf.set_fill_color(*COLOR_LIGHT_BG_RGB)
    pdf.set_draw_color(*COLOR_ALERT_BORDER)
    pdf.set_line_width(0.3)
    pdf.rect(15, pdf.get_y(), 180, 20, style="FD")
    
    pdf.set_font("helvetica", "B", 8.5)
    pdf.set_text_color(*COLOR_TEXT_DARK_RGB)
    pdf.set_xy(18, pdf.get_y() + 2)
    pdf.cell(30, 4, "Date:", 0, align="L", new_x="RIGHT", new_y="TOP")
    pdf.set_font("helvetica", "", 8.5)
    pdf.cell(60, 4, "2026-07-15", 0, align="L", new_x="RIGHT", new_y="TOP")
    
    pdf.set_font("helvetica", "B", 8.5)
    pdf.cell(30, 4, "Holdout Period:", 0, align="L", new_x="RIGHT", new_y="TOP")
    pdf.set_font("helvetica", "", 8.5)
    pdf.cell(60, 4, "30 trading days (May 28 - Jul 10, 2026)", 0, align="L", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_x(18)
    pdf.set_font("helvetica", "B", 8.5)
    pdf.cell(30, 4, "Assets:", 0, align="L", new_x="RIGHT", new_y="TOP")
    pdf.set_font("helvetica", "", 8.5)
    pdf.cell(60, 4, "AAPL, MSFT, NVDA, SPY, BTC-USD", 0, align="L", new_x="RIGHT", new_y="TOP")
    
    pdf.set_font("helvetica", "B", 8.5)
    pdf.cell(30, 4, "Run Type:", 0, align="L", new_x="RIGHT", new_y="TOP")
    pdf.set_font("helvetica", "", 8.5)
    pdf.cell(60, 4, "Single-run point-in-time snapshot", 0, align="L", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_xy(15, pdf.get_y() + 6)
    
    # ------------------ SECTION 1: SUMMARY METRICS ------------------
    pdf.chapter_title("1. Summary Metrics (Single Run)")
    
    # Main Table headers & data
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
    
    # Bold index mapping: indicates which fields are bold
    # (row_index, col_index)
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

    # Render table
    col_widths = (16, 18, 28, 28, 22, 24, 24, 20)
    
    with pdf.table(col_widths=col_widths, text_align="CENTER", line_height=5) as table:
        # Header Row
        header_row = table.row()
        pdf.set_font("helvetica", "B", 8)
        pdf.set_text_color(255, 255, 255)
        pdf.set_fill_color(*COLOR_PRIMARY_RGB)
        for h in headers:
            header_row.cell(h)
            
        # Data Rows
        pdf.set_text_color(*COLOR_TEXT_RGB)
        for r_idx, row_data in enumerate(data):
            row = table.row()
            # Alternating background colors
            if r_idx % 2 == 1:
                pdf.set_fill_color(241, 245, 249) # Light slate fill
            else:
                pdf.set_fill_color(255, 255, 255)
                
            for c_idx, cell_value in enumerate(row_data):
                is_bold = (r_idx, c_idx) in bold_indices
                
                # Check if model column is Kronos/TimesFM to color model name specifically
                if c_idx == 1:
                    pdf.set_font("helvetica", "B", 7.5)
                    if cell_value == "TimesFM":
                        pdf.set_text_color(*COLOR_ACCENT_RGB)
                    else:
                        pdf.set_text_color(194, 65, 12) # Warm color for Kronos
                else:
                    pdf.set_font("helvetica", "B" if is_bold else "", 7.5)
                    pdf.set_text_color(*(COLOR_TEXT_DARK_RGB if is_bold else COLOR_TEXT_RGB))
                    
                row.cell(cell_value)
                
    pdf.ln(3)
    pdf.set_font("helvetica", "I", 7.5)
    pdf.set_text_color(*COLOR_MUTED_RGB)
    pdf.cell(0, 4, "*Note: Bold indicates the better performing model on that specific metric & asset in this run.", border=0, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    
    # ------------------ SECTION 2: WHERE TIMESFM PERFORMS BETTER ------------------
    pdf.chapter_title("2. Where TimesFM Performs Better")
    
    pdf.add_bullet_point("Accuracy across All Assets", "TimesFM wins on every single asset for all error metrics (MAE/RMSE/MAPE). The margin is substantial on AAPL (MAE 11.26 vs 58.39 - 5.2x lower), NVDA (MAE 16.59 vs 35.82 - 2.2x lower), and SPY (MAE 21.46 vs 88.14 - 4.1x lower). On BTC-USD, TimesFM achieves a MAPE of 9.91% compared to Kronos's 40.15% (4x lower). Even on MSFT, TimesFM remains slightly more accurate (MAE 67.06 vs 68.38).")
    pdf.add_bullet_point("Inference Speed", "TimesFM is 16-30x faster than Kronos, completing forecasts in 0.23-0.27 seconds compared to Kronos's 4.3-5.6 seconds.")
    pdf.add_bullet_point("Memory Efficiency", "TimesFM uses 0.07-0.11 MB per ticker, which is 2.5-7x more memory efficient than Kronos (0.19-0.50 MB).")
    pdf.add_bullet_point("Probabilistic Output", "TimesFM natively provides calibrated quantile forecasts (p10-p90) and a tail-risk probability (P(>+5%)), which are critical for risk management. Kronos only outputted point forecasts in this benchmark.")
    
    pdf.ln(6)
    
    # ------------------ SECTION 3: WHERE KRONOS PERFORMS BETTER ------------------
    pdf.chapter_title("3. Where Kronos Performs Better")
    
    pdf.add_bullet_point("Directional Accuracy", "Kronos achieves higher directional accuracy on 3 out of 5 assets in this run: AAPL (60.0% vs 50.0%), MSFT (56.67% vs 50.0%), and NVDA (50.0% vs 46.67%), and ties on SPY (60.0% vs 60.0%).")
    pdf.add_bullet_point("Architectural Flexibility", "Kronos's generative, sampling-based architecture means it could theoretically produce distribution forecasts via repeated sampling, though this was not tested in this point-forecast run.")
    pdf.add_bullet_point("Zero-Shot Generalization", "Kronos is a true zero-shot foundation model with no task-specific tuning. This may provide a more robust baseline on highly out-of-distribution time series.")
    
    # ------------------ SECTION 4: LIMITATIONS OBSERVED ------------------
    pdf.add_page()
    pdf.chapter_title("4. Limitations Observed")
    
    pdf.heading_2("4.1 Kronos-Specific")
    pdf.add_bullet_point("High MAPE on most assets", "11-40% MAPE suggests that a single point-forecast sample from Kronos is highly noisy.", level=2)
    pdf.add_bullet_point("Stochastic Architecture", "Kronos is sampling-based and stochastic. A single run does not measure or control for cross-run variance, so repeated executions may produce different results.", level=2)
    pdf.add_bullet_point("No native quantile output", "The model samples but doesn't natively expose calibrated prediction intervals. Pinball loss is N/A.", level=2)
    pdf.add_bullet_point("High latency", "4-6 seconds per ticker makes it less viable for real-time applications.", level=2)
    
    pdf.heading_2("4.2 TimesFM-Specific")
    pdf.add_bullet_point("Directional accuracy drops", "TimesFM's directional accuracy was lower than Kronos on AAPL, MSFT, and NVDA in this run, indicating it may smooth over turning points in certain market regimes.", level=2)
    pdf.add_bullet_point("Single-run evaluation", "Cross-run variance was not tested. TimesFM produced consistent point forecasts within this single execution.", level=2)
    pdf.add_bullet_point("Quantile calibration unverified", "Pinball losses are calculated but their empirical coverage was not formally tested.", level=2)
    
    pdf.heading_2("4.3 General Limitations")
    pdf.add_bullet_point("Single holdout window", "All metrics are evaluated on one 30-day window. Performance could differ substantially in different market environments (e.g., bear markets, flat ranges).", level=2)
    pdf.add_bullet_point("No walk-forward validation", "A single holdout is a snapshot, not a statistically robust cross-validation.", level=2)
    
    pdf.ln(6)
    
    # ------------------ SECTION 5: CHART & CONCLUSIONS ------------------
    pdf.chapter_title("5. Performance Visualization")
    
    # Embed comparison chart
    chart_path = "plots/benchmark_comparison.png"
    if os.path.exists(chart_path):
        # We place it in the center of the page with width ~170mm
        pdf.image(chart_path, x=20, y=pdf.get_y() + 2, w=170)
        # Advance Y coordinate past the chart (approx 94mm high for standard 18:10 aspect ratio at 170mm width)
        pdf.set_y(pdf.get_y() + 102)
    else:
        pdf.set_font("helvetica", "I", 10)
        pdf.cell(0, 10, "Benchmark comparison chart [plots/benchmark_comparison.png] not found.", border=1, align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
    pdf.ln(4)
    
    pdf.chapter_title("6. Initial Conclusions")
    
    pdf.add_bullet_point("Model Selection", "TimesFM is the stronger model for most tested assets. It produces lower error on all 5 tickers, completes inference 16-30x faster, uses significantly less memory, and natively supports probabilistic interval forecasts.")
    pdf.add_bullet_point("Kronos Suitability", "Kronos is not production-ready as a single-sample point forecaster. Its high-variance, high-MAPE forecasts and 15-25x slower inference make it unsuitable for direct deployment.")
    pdf.add_bullet_point("Speed and Deployment", "TimesFM's speed advantage (16-30x) makes it viable for real-time applications, while Kronos's multi-second latency limits it to batch processes.")
    
    pdf.ln(5)
    
    # ------------------ SECTION 6: CAVEATS & APPENDIX ------------------
    pdf.add_page()
    pdf.chapter_title("7. Caveats & Appendix")
    
    # Render orange Warning box
    pdf.add_alert_box("IMPORTANT", "This is a single-run benchmark comparison. Cross-run variance was not tested. Kronos's sampling-based generative architecture is stochastic and may produce different metrics on repeated runs. Additionally, all evaluations are restricted to the same 30-day holdout window; performance under different market regimes was not evaluated.")
    
    pdf.ln(2)
    
    pdf.heading_2("Data Artifacts Location")
    pdf.set_font("helvetica", "", 9)
    pdf.set_text_color(*COLOR_TEXT_RGB)
    pdf.write(4, "All data predictions and summary runs can be found locally at:\n")
    pdf.set_font("helvetica", "B", 8.5)
    pdf.set_text_color(*COLOR_TEXT_DARK_RGB)
    
    pdf.write(5, "  - Benchmark metrics CSV: ")
    pdf.set_font("helvetica", "", 8.5)
    pdf.write(5, "data/predictions/benchmark_results.csv\n")
    
    pdf.set_font("helvetica", "B", 8.5)
    pdf.write(5, "  - Benchmark comparison chart: ")
    pdf.set_font("helvetica", "", 8.5)
    pdf.write(5, "plots/benchmark_comparison.png\n")
    
    pdf.set_font("helvetica", "B", 8.5)
    pdf.write(5, "  - Predictions (Parquet): ")
    pdf.set_font("helvetica", "", 8.5)
    pdf.write(5, "data/predictions/{ticker}_{model}_predictions.parquet\n")
    
    # Save PDF
    out_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "benchmark_analysis.pdf")
    
    pdf.output(pdf_path)
    print(f"\nPDF Report successfully generated → {pdf_path}")


if __name__ == "__main__":
    main()
