import pandas as pd
import numpy as np
import cufflinks as cf
import yfinance as yf
import ipywidgets as wd
import plotly.graph_objs as go
import plotly.io as pio
from IPython.display import display
from _plotly_utils.basevalidators import CompoundValidator, ColorValidator
import re  # Added for precise string replacement

# --- 1. THE REPAIRED COMPATIBILITY PATCHES ---

# Patch 1: Fix Pandas Index.format()
if not hasattr(pd.Index, 'format'):
    pd.Index.format = lambda self: self.astype(str).tolist()

# Patch 2: Fix 'titlefont' Layout Bug
original_compound_validate = CompoundValidator.validate_coerce
def patched_compound_validate(self, v, skip_invalid=False, _validate=True):
    if isinstance(v, dict):
        if 'titlefont' in v:
            v['title'] = v.get('title', {})
            if isinstance(v['title'], str): v['title'] = {'text': v['title']}
            v['title']['font'] = v.pop('titlefont')
        if 'title' in v and isinstance(v['title'], str):
            v['title'] = {'text': v['title']}
    return original_compound_validate(self, v, skip_invalid, _validate)
CompoundValidator.validate_coerce = patched_compound_validate

# Patch 3: Precision fix for 'rgba(...np.float64(1.0)...)'
original_color_validate = ColorValidator.validate_coerce
def patched_color_validate(self, v, should_raise=True):
    if isinstance(v, str) and 'np.float64' in v:
        # This regex finds 'np.float64(NUMBER)' and replaces it with just 'NUMBER'
        # It preserves the outer rgba(...) parentheses
        v = re.sub(r'np\.float64\(([\d\.]+)\)', r'\1', v)
    return original_color_validate(self, v)
ColorValidator.validate_coerce = patched_color_validate

# --- 2. INITIALIZATION & LOGIC ---
pio.renderers.default = "browser" 
cf.go_offline()

def ta_dashboard(asset, indicator, start_date, end_date, bb_k, bb_n, rsi_periods):
    # Download
    df = yf.download(asset, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty: return
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # QuantFig
    qf = cf.QuantFig(df, title=f'Dashboard - {asset}', legend='right', name=asset)
    if 'Bollinger Bands' in indicator: 
        qf.add_bollinger_bands(periods=bb_n, boll_std=bb_k)
    if 'RSI' in indicator: 
        qf.add_rsi(periods=rsi_periods, showbands=True)

    # Render
    fig = qf.iplot(asFigure=True)
    pio.show(fig)

# --- 3. UI SETUP ---
stocks_sel = wd.Dropdown(options=['META', 'MSFT', 'GOOGL', 'AAPL', 'NVDA'], value='META', description='Asset')
ind_sel = wd.SelectMultiple(options=['Bollinger Bands', 'RSI'], value=['Bollinger Bands'], description='Indicator')
start_dt = wd.DatePicker(description='Start', value=pd.to_datetime('2024-01-01'))
end_dt = wd.DatePicker(description='End', value=pd.to_datetime('2025-01-01'))
bb_n_sld = wd.IntSlider(value=20, min=5, max=40, description='BB N', continuous_update=False)
bb_k_sld = wd.FloatSlider(value=2, min=1, max=4, step=0.5, description='BB k', continuous_update=False)

ui = wd.VBox([wd.HBox([stocks_sel, ind_sel]), wd.HBox([start_dt, end_dt]), wd.HBox([bb_n_sld, bb_k_sld])])
out = wd.interactive_output(ta_dashboard, {
    'asset': stocks_sel, 'indicator': ind_sel, 'start_date': start_dt, 
    'end_date': end_dt, 'bb_k': bb_k_sld, 'bb_n': bb_n_sld, 'rsi_periods': wd.fixed(14)
})

display(ui, out)