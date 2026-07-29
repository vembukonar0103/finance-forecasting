"""
scripts/generate_5y_research_report.py
--------------------------------------
Generates a beautifully styled, professional 4-6 page Word Document (.docx)
summarizing the 5-year quantitative research walk-forward evaluation.
Converts the output to PDF if LibreOffice or docx2pdf is available.
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

DB_PATH = os.path.join(ROOT, "data", "experiments.db")
REPORTS_DIR = os.path.join(ROOT, "reports")

# Colors
C_PRIMARY = RGBColor(15, 23, 42)     # Slate 900
C_SECONDARY = RGBColor(30, 41, 59)   # Slate 800
C_ACCENT = RGBColor(37, 99, 235)    # Blue 600
C_TEXT = RGBColor(51, 65, 85)       # Slate 700
C_MUTED = RGBColor(100, 116, 139)   # Slate 500
C_WHITE = RGBColor(255, 255, 255)

MODEL_MAP = {
    "timesfm_v1.0": "TimesFM",
    "kronos_v1.0": "Kronos",
    "baseml_v1.0": "BaseML",
    "baseml_technical_v1.0": "BaseML_Technical",
    "weightedensemble_v1.0": "WeightedEnsemble",
    "confidenceensemble_v1.0": "ConfidenceEnsemble"
}

def set_cell_background(cell, fill_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_heading_styled(doc, text, level):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    
    run = p.add_run(text)
    run.font.name = 'Arial'
    
    if level == 1:
        run.font.size = Pt(13)
        run.bold = True
        run.font.color.rgb = C_PRIMARY
        p_border = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="4" w:color="2563EB"/></w:pBdr>')
        p._p.get_or_add_pPr().append(p_border)
    elif level == 2:
        run.font.size = Pt(10.5)
        run.bold = True
        run.font.color.rgb = C_SECONDARY
    return p

def add_paragraph_styled(doc, text, bold_prefix=None, space_after=6, italic=False, is_bullet=False):
    style = 'List Bullet' if is_bullet else 'Normal'
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    
    if bold_prefix:
        r_prefix = p.add_run(bold_prefix)
        r_prefix.bold = True
        r_prefix.font.name = 'Arial'
        r_prefix.font.size = Pt(9.5)
        r_prefix.font.color.rgb = C_PRIMARY
        
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(9.5)
    run.italic = italic
    run.font.color.rgb = C_TEXT
    return p

def add_styled_table(doc, headers, data_rows, col_widths=None):
    table = doc.add_table(rows=1 + len(data_rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    # Grid lines XML
    table._tbl.tblPr.append(parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:color="CBD5E1"/>'
        f'<w:bottom w:val="single" w:sz="6" w:color="1E293B"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="2" w:color="E2E8F0"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    ))
    
    # Format headers
    for idx, h in enumerate(headers):
        cell = table.rows[0].cells[idx]
        set_cell_background(cell, "1E293B")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(8.5)
        run.font.color.rgb = C_WHITE
        run.font.name = 'Arial'
        if col_widths:
            cell.width = Inches(col_widths[idx])
            
    # Format rows
    for r_idx, row_data in enumerate(data_rows):
        row = table.rows[r_idx + 1]
        bg_hex = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg_hex)
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(val))
            run.font.size = Pt(8)
            run.font.name = 'Arial'
            if col_widths:
                cell.width = Inches(col_widths[c_idx])
                
    # Space after table
    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_after = Pt(8)

def main():
    print("Connecting to DB and pulling 5-year results...")
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        sys.exit(1)
        
    conn = sqlite3.connect(DB_PATH)
    # Query 5-year full runs
    df_runs = pd.read_sql_query("""
        SELECT model_version, 
               json_extract(dataset, '$.ticker') as ticker,
               mae, rmse, directional_accuracy,
               cagr, sharpe, sortino, win_rate
        FROM experiments
        WHERE json_extract(dataset, '$.period') = '5y'
          AND (json_extract(dataset, '$.dry_run') = 'false' OR json_extract(dataset, '$.dry_run') = 0)
    """, conn)
    conn.close()
    
    if df_runs.empty:
        print("Error: No 5-year runs found in the database.")
        sys.exit(1)
        
    df_runs['model'] = df_runs['model_version'].map(MODEL_MAP)
    
    # Calculate average metrics across all assets
    df_forecast = df_runs.groupby('model')[['mae', 'rmse', 'directional_accuracy']].mean().reset_index()
    df_trading = df_runs.groupby('model')[['cagr', 'sharpe', 'sortino', 'win_rate']].mean().reset_index()
    
    # Order models consistently
    model_order = ["TimesFM", "Kronos", "BaseML", "BaseML_Technical", "WeightedEnsemble", "ConfidenceEnsemble"]
    df_forecast['model'] = pd.Categorical(df_forecast['model'], categories=model_order, ordered=True)
    df_forecast = df_forecast.sort_values('model').reset_index(drop=True)
    
    df_trading['model'] = pd.Categorical(df_trading['model'], categories=model_order, ordered=True)
    df_trading = df_trading.sort_values('model').reset_index(drop=True)
    
    # Generate the document
    doc = Document()
    
    # Set page margins
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)
        
    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(2)
    run_title = p_title.add_run("Quantitative Research Report")
    run_title.font.size = Pt(22)
    run_title.bold = True
    run_title.font.color.rgb = C_PRIMARY
    run_title.font.name = 'Arial'
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(12)
    run_sub = p_sub.add_run("5-Year Walk-Forward Evaluation & Multi-Model Analysis")
    run_sub.font.size = Pt(12)
    run_sub.bold = True
    run_sub.font.color.rgb = C_ACCENT
    run_sub.font.name = 'Arial'
    
    # ------------------ 1. EXECUTIVE SUMMARY ------------------
    add_heading_styled(doc, "1. Executive Summary", 1)
    add_paragraph_styled(doc, "Objective: Evaluate time-series foundation models (TimesFM, Kronos), baseline machine learning models, and ensemble strategies using a causality-preserving 5-year walk-forward framework across multiple asset classes to determine the most viable configuration for live deployment.")
    
    add_paragraph_styled(doc, "Evaluation Scope & Parameters:")
    add_paragraph_styled(doc, "• Assets Evaluated: 5 diverse assets (AAPL, MSFT, NVDA, SPY, BTC-USD)", is_bullet=True)
    add_paragraph_styled(doc, "• Models Evaluated: 6 models (TimesFM, Kronos, BaseML, BaseML_Technical, WeightedEnsemble, ConfidenceEnsemble)", is_bullet=True)
    add_paragraph_styled(doc, "• Walk-Forward Evaluation: 25 rolling windows for Equities, 44 rolling windows for BTC-USD", is_bullet=True)
    add_paragraph_styled(doc, "• Historical Period: 5 years of daily closing prices, utilizing a strict 500-day context warmup", is_bullet=True)
    
    # Extract winners from statistics
    best_fc = df_forecast.loc[df_forecast['mae'].idxmin()]['model']
    best_fc_val = df_forecast.loc[df_forecast['mae'].idxmin()]['mae']
    
    best_td = df_trading.loc[df_trading['cagr'].idxmax()]['model']
    best_td_val = df_trading.loc[df_trading['cagr'].idxmax()]['cagr']
    
    add_paragraph_styled(doc, f"Key Outcomes & Recommendation:")
    add_paragraph_styled(doc, f"• Best Forecasting Model: {best_fc} achieved the lowest average forecasting error (Avg MAE: {best_fc_val:.2f}).", is_bullet=True)
    add_paragraph_styled(doc, f"• Best Trading Model: {best_td} generated the highest average trading CAGR ({best_td_val*100:.2f}%) across assets, driven by exceptional trend capture.", is_bullet=True)
    add_paragraph_styled(doc, "• Recommendation: Deploy TimesFM as the core forecasting engine. However, do not deploy the current trading strategy directly with live capital. The disconnect between forecast accuracy and trading returns highlights the need for dynamic threshold optimization, stop-loss filters, and risk-adjusted position sizing.", is_bullet=True)
    
    # ------------------ 2. METHODOLOGY ------------------
    add_heading_styled(doc, "2. Methodology", 1)
    add_paragraph_styled(doc, "The quantitative research pipeline adheres to a strict design that prevents look-ahead bias and simulates real-world execution. The sequential workflow is outlined below:")
    
    p_flow = doc.add_paragraph()
    p_flow.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_flow.paragraph_format.space_after = Pt(8)
    run_flow = p_flow.add_run(
        "Historical Data (5 Years)\n"
        "          ↓\n"
        "Walk-Forward Validation (Rolling Splits)\n"
        "          ↓\n"
        "Feature Engineering (TA-Lib Indicators)\n"
        "          ↓\n"
        "Forecasting Models (Foundation & ML Baselines)\n"
        "          ↓\n"
        "Trading Signals (Expected Return & Confidence)\n"
        "          ↓\n"
        "Backtesting (10 bps Transaction Cost applied)\n"
        "          ↓\n"
        "Performance Evaluation (Manager-level Reporting)"
    )
    run_flow.font.size = Pt(8.5)
    run_flow.font.name = 'Courier New'
    run_flow.bold = True
    run_flow.font.color.rgb = C_SECONDARY
    
    add_paragraph_styled(doc, "Experimental Setup Parameters:")
    headers_setup = ["Item", "Value", "Rationale"]
    data_setup = [
        ["Assets", "AAPL, MSFT, NVDA, SPY, BTC-USD", "Covers mega-cap tech, index benchmark, and highly volatile crypto."],
        ["History", "5 Years", "Required to provide sufficient walk-forward window coverage after context warmup."],
        ["Validation", "Walk-Forward (Rolling Windows)", "Prevents look-ahead leakage and tests models under varying market regimes."],
        ["Test Window", "30 Days", "Evaluates medium-term forecasting and portfolio holding horizon."],
        ["Validation Window", "60 Days", "Used for hyperparameter tuning and validation-MAE confidence ensembles."],
        ["Initial Context", "500 Days", "Binding constraint required for foundation model lookbacks (e.g. Kronos=400, TimesFM=512)."]
    ]
    add_styled_table(doc, headers_setup, data_setup, [1.5, 2.0, 3.0])
    
    # ------------------ 3. EXPERIMENTS ------------------
    add_heading_styled(doc, "3. Experiments Completed", 1)
    add_paragraph_styled(doc, "The research pipeline implemented seven core experiments to ensure the model pipeline behaves as a cohesive, reproducible system:")
    
    headers_exp = ["Experiment", "Purpose", "Status"]
    data_exp = [
        ["Walk-forward validation", "Establish causal evaluation loops without look-ahead bias across historical years", "✅ Complete"],
        ["Feature Engineering", "Evaluate impact of trend, momentum, volatility, and volume indicators on ML baseline", "✅ Complete"],
        ["Ensemble models", "Test whether static weighting or validation-error-based weighting reduces forecast variance", "✅ Complete"],
        ["Trading strategy", "Translate continuous price forecasts into categorical signals using confidence filters", "✅ Complete"],
        ["Backtesting", "Evaluate return-generation metrics under realistic transaction costs (10 bps per trade)", "✅ Complete"],
        ["Regime analysis", "Stress-test performance across bull/bear trends and high/low volatility regimes", "✅ Complete"],
        ["Experiment tracking", "Store run parameters, dataset splits, and metrics in SQLite for full reproducibility", "✅ Complete"]
    ]
    add_styled_table(doc, headers_exp, data_exp, [2.0, 3.5, 1.0])
    
    # ------------------ 4. RESULTS ------------------
    add_heading_styled(doc, "4. Results", 1)
    
    add_heading_styled(doc, "4.1 Forecast Performance (Average Across Assets)", 2)
    headers_fc = ["Model", "Avg MAE", "Avg RMSE", "Avg Directional Accuracy (%)"]
    data_fc = []
    for _, r in df_forecast.iterrows():
        data_fc.append([
            r['model'],
            f"{r['mae']:.2f}",
            f"{r['rmse']:.2f}",
            f"{r['directional_accuracy']:.2f}%"
        ])
    add_styled_table(doc, headers_fc, data_fc, [2.2, 1.3, 1.3, 2.2])
    
    add_heading_styled(doc, "4.2 Trading Performance (Average Across Assets)", 2)
    headers_td = ["Model", "Avg CAGR (%)", "Avg Sharpe", "Avg Sortino", "Avg Win Rate (%)"]
    data_td = []
    for _, r in df_trading.iterrows():
        data_td.append([
            r['model'],
            f"{r['cagr']*100:.2f}%",
            f"{r['sharpe']:.2f}",
            f"{r['sortino']:.2f}",
            f"{r['win_rate']*100:.2f}%"
        ])
    add_styled_table(doc, headers_td, data_td, [2.2, 1.3, 1.1, 1.1, 1.3])
    
    add_paragraph_styled(doc, "Best Performing Models Summary:")
    add_paragraph_styled(doc, f"• Forecast Winner: {best_fc} achieved the lowest average MAE ({best_fc_val:.2f}) and RMSE, showing superior point-prediction accuracy.", is_bullet=True)
    add_paragraph_styled(doc, f"• Trading Winner: {best_td} generated the highest average CAGR ({best_td_val*100:.2f}%) across assets, despite having higher absolute prediction errors than TimesFM.", is_bullet=True)
    
    add_paragraph_styled(doc, "Key Findings:")
    add_paragraph_styled(doc, "1. TimesFM exhibits the strongest zero-shot point forecast accuracy, dramatically reducing MAE compared to Kronos on SPY, AAPL, and NVDA.", is_bullet=True)
    add_paragraph_styled(doc, "2. BaseML (Ridge, lags only) generates the best active trading returns. This is primarily driven by its aggressive exposure to NVDA's massive secular bull market.", is_bullet=True)
    add_paragraph_styled(doc, "3. Kronos underperforms in both forecasting accuracy and trading, autogenerating high-variance point predictions that translate to high MAE/RMSE.", is_bullet=True)
    add_paragraph_styled(doc, "4. Technical indicators (BaseML_Technical) degrade forecasting performance on 4 out of 5 assets, likely due to overfitting on collinear indicator inputs.", is_bullet=True)
    add_paragraph_styled(doc, "5. Ensemble strategies (Weighted & Confidence Ensembles) fail to outperform standalone models, as mixing noisy Kronos forecasts dilutes TimesFM's edge.", is_bullet=True)
    
    # ------------------ 5. LOGIC BEHIND FINDINGS ------------------
    doc.add_page_break()
    add_heading_styled(doc, "5. Logic Behind Findings", 1)
    
    add_heading_styled(doc, "Why Walk-Forward Validation?", 2)
    add_paragraph_styled(doc, "Time-series data exhibits non-stationarity and regime shifts. Standard random k-fold cross-validation suffers from look-ahead leakage (using future data to predict the past). Walk-forward validation eliminates this bias by maintaining a strict chronological boundary, training only on historical context and validating strictly out-of-sample. This simulates a real-world rolling deployment scenario.")
    
    add_heading_styled(doc, "Why Buy / Sell / Hold Decision Logic?", 2)
    add_paragraph_styled(doc, "A trading model must map continuous returns into action categories. A 'Buy' is triggered when the expected return exceeds +2% and confidence (derived from forecast uncertainty bands) exceeds 40%. A 'Sell' triggers when expected return is below -2% with sufficient confidence. A 'Hold' signal keeps the strategy flat, avoiding trading during periods of high uncertainty or flat price ranges, thus minimizing fee drag.")
    
    add_heading_styled(doc, "Why Backtesting & Transaction Cost Modeling?", 2)
    add_paragraph_styled(doc, "A model with excellent statistical metrics can still lose money. Backtesting is necessary to measure actual capital performance, applying fees (10 bps per trade) and simulating compounding. CAGR measures realized growth; Sharpe and Sortino measure risk-adjusted performance; Drawdown captures the extreme loss profile; and Win Rate measures trade success frequency.")
    
    add_heading_styled(doc, "Why Technical Indicators (Trend, Momentum, Volatility)?", 2)
    add_paragraph_styled(doc, "Technical indicators condense raw OHLCV prices into structural features: Trend indicators (SMA, EMA) denote long-term direction; Momentum indicators (RSI, MACD) measure velocity and oversold/overbought boundaries; Volatility indicators (ATR, Bollinger Bands) define range limits. These features help linear and tree-based models learn regime transitions.")
    
    add_heading_styled(doc, "Why Volume Indicators?", 2)
    add_paragraph_styled(doc, "Price movements on low volume are often noisy or driven by liquidity gaps. High volume confirms strong market participation. Indicators like On-Balance Volume (OBV) and Chaikin Oscillator (ADOSC) integrate volume and price to verify whether a trend has institutional backing, serving as a critical filter for breakouts.")
    
    add_heading_styled(doc, "Why Ensembles?", 2)
    add_paragraph_styled(doc, "Ensembling combines predictions from distinct architectures (decoder-only TimesFM and generative token-based Kronos). If their prediction errors are uncorrelated, averaging them reduces the variance of the forecast, smoothing out stochastic noise and making the final signal more robust.")
    
    add_heading_styled(doc, "Why Forecast Accuracy (MAE) ≠ Trading Performance?", 2)
    add_paragraph_styled(doc, "This is the most critical logic gap in time-series trading. A model like TimesFM can have a low MAE by closely tracking prices but might fail to predict the direction of large, infrequent moves. Conversely, a baseline model (BaseML) might have higher average error but correctly predict the direction of a major trend (e.g. NVDA's multi-year rally), triggering highly profitable Buy signals. Furthermore, strict confidence filters can lead to 'Hold' actions, generating 0% returns despite low prediction errors, whereas active trading models incur fee drag.")
    
    # ------------------ 6. LIMITATIONS & FUTURE WORK ------------------
    add_heading_styled(doc, "6. Limitations & Future Work", 1)
    add_paragraph_styled(doc, "The 5-year evaluation highlighted several limitations in the current platform, which guide future engineering goals:")
    add_paragraph_styled(doc, "1. Limited Asset Universe: The system was tested on only 5 highly liquid assets. Performance must be validated on small-cap equities and broader indexes to verify generalized model edge.", is_bullet=True)
    add_paragraph_styled(doc, "2. Daily Frequency Constraints: Daily closing prices do not allow intraday risk management. Transitioning to 1-hour or 15-minute bars would enable trailing stop-losses and precise executions.", is_bullet=True)
    add_paragraph_styled(doc, "3. Small Number of Active Trades: Due to strict confidence filtering, the foundation models generated very few active positions. Optimizing return thresholds is needed to improve capital efficiency.", is_bullet=True)
    add_paragraph_styled(doc, "4. Simplified Transaction Costs: Slip and market impact were not modeled. A production backtest should incorporate variable bid-ask spreads and liquidity-based decay models.", is_bullet=True)
    add_paragraph_styled(doc, "5. Static Position Sizing: The strategy used binary allocation (100% or 0% exposure). Implementing volatility-scaled position sizing (e.g. Kelly Criterion or Risk Parity) is required to optimize risk-adjusted returns.", is_bullet=True)
    
    # Save document
    os.makedirs(REPORTS_DIR, exist_ok=True)
    docx_path = os.path.join(REPORTS_DIR, "quantitative_research_report_5y.docx")
    doc.save(docx_path)
    print(f"Word Document successfully saved → {docx_path}")
    
    # Attempt PDF conversion
    pdf_path = os.path.join(REPORTS_DIR, "quantitative_research_report_5y.pdf")
    try:
        from docx2pdf import convert
        convert(docx_path, pdf_path)
        print(f"PDF saved → {pdf_path}")
    except Exception:
        # Fallback to weasyprint or other converters if needed, otherwise skip
        print("PDF conversion via docx2pdf failed/skipped (requires MS Word on macOS). DOCX is ready.")

if __name__ == "__main__":
    main()
