from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from dash import Dash, Input, Output, dcc, html
from neo4j import GraphDatabase
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.dataset import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, build_split, signal_size
from model.stgnn import STGNN
from model.baselines import GRUBaseline, MLPBaseline, PersistenceBaseline

def mesh_info_from_run(run_id: str) -> tuple[str, int, int]:
    if run_id.startswith('mesh_8x8'):
        return ('mesh_8x8', 8, 64)
    if run_id.startswith('mesh_4x4'):
        return ('mesh_4x4', 4, 16)
    return ('mesh_4x4', 4, 16)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'results' / 'models'
METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'

def fetch_run_dataframe(run_id: str) -> pd.DataFrame:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    rows = []
    with driver.session() as session:
        for r in session.run('\n            MATCH (rs:RouterState {run_id: $rid})-[:OBSERVED_AT]->(r:Router)\n            RETURN rs.sample_idx AS t, r.id AS router_id, r.x AS x, r.y AS y,\n                   rs.buffer_occupancy_norm AS occ,\n                   rs.stored_total AS stored,\n                   rs.injected AS inj, rs.received_total AS rec\n            ORDER BY t, router_id\n            ', rid=run_id):
            rows.append(dict(r))
    driver.close()
    return pd.DataFrame(rows)

def list_runs() -> list[str]:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run('MATCH (s:Snapshot) RETURN DISTINCT s.run_id AS rid ORDER BY rid')
        runs = [r['rid'] for r in result]
    driver.close()
    return runs

def load_models(window: int, in_features: int, hidden: int=32, mesh: str='mesh_4x4'):
    suffix = '_8x8' if mesh == 'mesh_8x8' else ''
    models = {'Persistence': PersistenceBaseline(window=window), 'MLP': None, 'GRU': None, 'STGNN': None}
    mlp_ck = MODELS_DIR / f'mlp{suffix}.pt'
    if mlp_ck.is_file():
        mlp = MLPBaseline(in_features=in_features, hidden=hidden, dropout=0.2)
        try:
            mlp.load_state_dict(torch.load(mlp_ck, map_location='cpu'))
            mlp.eval()
            models['MLP'] = mlp
        except Exception as exc:
            print(f'[load_models] MLP {mlp_ck.name} load failed: {exc}')
    gru_ck = MODELS_DIR / f'gru{suffix}.pt'
    if gru_ck.is_file():
        gru = GRUBaseline(window=window, hidden=hidden, dropout=0.15)
        try:
            gru.load_state_dict(torch.load(gru_ck, map_location='cpu'))
            gru.eval()
            models['GRU'] = gru
        except Exception as exc:
            print(f'[load_models] GRU {gru_ck.name} load failed: {exc}')
    stgnn_ck = MODELS_DIR / f'stgnn{suffix}.pt'
    if stgnn_ck.is_file():
        stgnn = STGNN(window=window, gcn_hidden=hidden, gru_hidden=hidden, dropout=0.1)
        try:
            stgnn.load_state_dict(torch.load(stgnn_ck, map_location='cpu'))
            stgnn.eval()
            models['STGNN'] = stgnn
        except Exception as exc:
            print(f'[load_models] STGNN {stgnn_ck.name} load failed: {exc}')
    return models

def predict_run(models: dict, run_id: str, *, window: int, target_scale: float=100.0) -> dict[str, np.ndarray]:
    mesh_id, k, n_nodes_expected = mesh_info_from_run(run_id)
    all_same_mesh = [r for r in list_runs() if r.startswith(mesh_id + '_') and r != run_id and (r != 'mesh_4x4_uniform_smoke')]
    if not all_same_mesh:
        return {}
    val_runs = (all_same_mesh[0],)
    excl = ('mesh_4x4_uniform_smoke',) if mesh_id == 'mesh_4x4' else ()
    split = build_split(window=window, mesh_id=mesh_id, val_runs=val_runs, test_runs=(run_id,), exclude_runs=excl)
    n_nodes = split.num_nodes
    df = fetch_run_dataframe(run_id)
    T = int(df['t'].max()) + 1
    preds_by_model: dict[str, np.ndarray] = {}
    for name, model in models.items():
        if model is None:
            continue
        preds = np.full((n_nodes, T), np.nan, dtype=np.float32)
        snapshot_iter = iter(split.test)
        ts_filled = []
        for i, snapshot in enumerate(snapshot_iter):
            with torch.no_grad():
                out = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            if name == 'Persistence':
                out_orig = out.detach().cpu().numpy()
            else:
                out_orig = (out / target_scale).detach().cpu().numpy()
            t_target = window - 1 + i + 1
            if 0 <= t_target < T:
                preds[:, t_target] = out_orig
                ts_filled.append(t_target)
        preds_by_model[name] = preds
    return preds_by_model
COLORS = {'bg': '#f5f7fb', 'card': '#ffffff', 'border': '#e1e6ef', 'text': '#1f2937', 'muted': '#6b7280', 'primary': '#1e40af', 'accent': '#d62728', 'success': '#059669', 'warning': '#d97706', 'shadow': '0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)', 'shadow_lg': '0 4px 6px rgba(0,0,0,0.05), 0 2px 4px rgba(0,0,0,0.06)'}
S_BODY = {'fontFamily': "-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif", 'backgroundColor': COLORS['bg'], 'color': COLORS['text'], 'minHeight': '100vh', 'padding': '0', 'margin': '0'}
S_CONTAINER = {'maxWidth': '1200px', 'margin': '0 auto', 'padding': '24px 32px 48px 32px'}
S_CARD = {'backgroundColor': COLORS['card'], 'borderRadius': '12px', 'padding': '20px 24px', 'boxShadow': COLORS['shadow'], 'border': f"1px solid {COLORS['border']}", 'marginBottom': '20px'}
S_SECTION_TITLE = {'fontSize': '16px', 'fontWeight': '600', 'color': COLORS['text'], 'margin': '0 0 4px 0', 'letterSpacing': '-0.01em'}
S_SECTION_SUB = {'fontSize': '13px', 'color': COLORS['muted'], 'margin': '0 0 16px 0'}

def hero_header() -> html.Div:
    pill_style = {'display': 'inline-block', 'padding': '4px 12px', 'borderRadius': '999px', 'fontSize': '12px', 'fontWeight': '500', 'marginRight': '8px'}
    return html.Div([html.Div([html.Div('NoC Congestion Prediction', style={'fontSize': '26px', 'fontWeight': '700', 'color': COLORS['primary'], 'letterSpacing': '-0.02em'}), html.Div('BI Dashboard - Spatio-Temporal Graph Neural Networks', style={'fontSize': '14px', 'color': COLORS['muted'], 'marginTop': '2px'})]), html.Div([html.Span('44 runs', style={**pill_style, 'background': '#dbeafe', 'color': '#1e40af'}), html.Span('4 models', style={**pill_style, 'background': '#fee2e2', 'color': '#991b1b'}), html.Span('Neo4j', style={**pill_style, 'background': '#dcfce7', 'color': '#166534'}), html.Span('4×4 + 8×8', style={**pill_style, 'background': '#fef3c7', 'color': '#92400e'})], style={'marginTop': '12px'})], style={**S_CARD, 'background': 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)', 'padding': '24px 28px', 'marginBottom': '24px'})

def build_app() -> Dash:
    app = Dash(__name__, title='NoC Congestion BI', external_stylesheets=['https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'])
    app.index_string = '\n    <!DOCTYPE html>\n    <html>\n        <head>\n            {%metas%}\n            <title>{%title%}</title>\n            {%favicon%}\n            {%css%}\n            <style>\n                body { margin: 0; background-color: ' + COLORS['bg'] + '; }\n                .Select-control { border-radius: 8px !important;\n                                  border: 1px solid ' + COLORS['border'] + ' !important; }\n                .Select-menu-outer { border-radius: 8px !important; }\n            </style>\n        </head>\n        <body>\n            {%app_entry%}\n            <footer>\n                {%config%}\n                {%scripts%}\n                {%renderer%}\n            </footer>\n        </body>\n    </html>\n    '
    runs = list_runs()
    default_run = next((r for r in runs if 'ir040' in r), runs[0] if runs else '')
    WINDOW = 5
    NUM_FEATURES = 5
    HIDDEN = 32
    models_4x4 = load_models(WINDOW, in_features=WINDOW * NUM_FEATURES, hidden=HIDDEN, mesh='mesh_4x4')
    models_8x8 = load_models(WINDOW, in_features=WINDOW * NUM_FEATURES, hidden=HIDDEN, mesh='mesh_8x8')
    has_trained_any = any((m is not None and (not isinstance(m, PersistenceBaseline)) for mset in (models_4x4, models_8x8) for n, m in mset.items() if n != 'Persistence'))

    def models_for(run_id: str) -> dict:
        return models_8x8 if run_id.startswith('mesh_8x8') else models_4x4
    metrics_table_div = build_metrics_table()
    label_style = {'fontSize': '13px', 'fontWeight': '500', 'color': COLORS['muted'], 'display': 'block', 'marginBottom': '6px', 'textTransform': 'uppercase', 'letterSpacing': '0.04em'}
    app.layout = html.Div([html.Div([hero_header(), html.Div([html.Div([html.Label('Chọn run mô phỏng', style=label_style), dcc.Dropdown(id='run-dropdown', options=[{'label': r, 'value': r} for r in runs], value=default_run, clearable=False, style={'width': '100%'})], style={'flex': '1', 'marginRight': '20px'}), html.Div([html.Label('Mesh size', style=label_style), html.Div(id='mesh-badge', style={'padding': '8px 14px', 'borderRadius': '8px', 'background': '#eff6ff', 'color': '#1e40af', 'fontWeight': '600', 'fontSize': '14px', 'textAlign': 'center', 'border': '1px solid #bfdbfe'})], style={'width': '150px'})], style={**S_CARD, 'display': 'flex', 'alignItems': 'flex-end'}), html.Div([html.Div('Bảng so sánh hiệu năng - mesh 4×4', style=S_SECTION_TITLE), html.Div(['Số liệu cố định trên mesh 4×4 từ ', html.Code('results/metrics/comparison.csv', style={'background': '#f3f4f6', 'padding': '1px 6px', 'borderRadius': '4px', 'fontSize': '12px'}), '. Trên 8×8, ST-GNN tinh chỉnh đạt ', html.Strong('0.00861', style={'color': COLORS['accent']}), ' (vượt GRU 0.00875) - xem paper §IV.F.'], style=S_SECTION_SUB), metrics_table_div], style=S_CARD), html.Div([html.Div('Bản đồ nhiệt buffer occupancy theo thời gian', style=S_SECTION_TITLE), html.Div('Kéo slider để xem trạng thái mạng tại từng sample period (1 sample = 100 cycles).', style=S_SECTION_SUB), html.Div([html.Label('Sample index (t)', style=label_style), dcc.Slider(id='sample-slider', min=0, max=100, step=1, value=10, marks=None, tooltip={'placement': 'bottom', 'always_visible': True})], style={'padding': '0 8px 8px 8px'}), dcc.Loading(dcc.Graph(id='heatmap-actual', config={'displayModeBar': False}), type='default', color=COLORS['primary'])], style=S_CARD), html.Div([html.Div('Dự đoán vs thực tế theo thời gian', style=S_SECTION_TITLE), html.Div(['So sánh 4 mô hình: ', html.Span('Persistence', style={'color': '#6b7280', 'fontWeight': '500'}), ' · ', html.Span('MLP', style={'color': '#1f77b4', 'fontWeight': '500'}), ' · ', html.Span('GRU', style={'color': '#2ca02c', 'fontWeight': '500'}), ' · ', html.Span('ST-GNN (đề xuất)', style={'color': COLORS['accent'], 'fontWeight': '600'})], style=S_SECTION_SUB), html.Div([html.Label('Router ID', style=label_style), dcc.Dropdown(id='router-dropdown', value=5, clearable=False, style={'width': '180px'})], style={'marginBottom': '12px'}), dcc.Loading(dcc.Graph(id='timeseries', config={'displayModeBar': False}), type='default', color=COLORS['primary'])], style=S_CARD), html.Div([html.Hr(style={'border': 'none', 'borderTop': f"1px solid {COLORS['border']}", 'margin': '32px 0 16px 0'}), html.Div([html.Span('FAIR 2026 - ', style={'fontWeight': '600'}), 'Lê Tiến Hiếu', html.Br(), html.Span('Stack: BookSim2 + Neo4j + PyTorch Geometric Temporal + Plotly Dash', style={'color': COLORS['muted'], 'fontSize': '12px'})], style={'fontSize': '13px', 'color': COLORS['muted'], 'textAlign': 'center'})])], style=S_CONTAINER)], style=S_BODY)

    @app.callback(Output('sample-slider', 'max'), Output('sample-slider', 'value'), Output('router-dropdown', 'options'), Output('router-dropdown', 'value'), Output('mesh-badge', 'children'), Input('run-dropdown', 'value'))
    def update_run_state(run_id):
        if not run_id:
            return (100, 10, [{'label': f'R{i}', 'value': i} for i in range(16)], 5, '4×4')
        _, k, num_nodes = mesh_info_from_run(run_id)
        df = fetch_run_dataframe(run_id)
        T = int(df['t'].max())
        router_opts = [{'label': f'R{i}', 'value': i} for i in range(num_nodes)]
        default_router = 5 if num_nodes >= 6 else 0
        badge = f'{k}×{k} ({num_nodes} routers)'
        return (T, min(10, T // 2), router_opts, default_router, badge)

    @app.callback(Output('heatmap-actual', 'figure'), Input('run-dropdown', 'value'), Input('sample-slider', 'value'))
    def update_heatmap(run_id, sample_idx):
        df = fetch_run_dataframe(run_id)
        _, k, _ = mesh_info_from_run(run_id)
        sub = df[df['t'] == sample_idx]
        z = np.full((k, k), np.nan)
        for _, row in sub.iterrows():
            z[int(row['y']), int(row['x'])] = row['occ']
        fig = px.imshow(z, color_continuous_scale='Viridis', origin='lower', zmin=0, zmax=max(0.1, float(df['occ'].max())), labels=dict(x='X', y='Y', color='occupancy'))
        if k <= 4:
            for y in range(k):
                for x in range(k):
                    v = z[y, x]
                    if np.isnan(v):
                        continue
                    fig.add_annotation(x=x, y=y, text=f'{v:.3f}', showarrow=False, font=dict(color='white' if v > 0.05 else 'black'))
        fig.update_layout(title={'text': f'<b>{run_id}</b> @ t={sample_idx}', 'font': {'size': 14, 'color': COLORS['text']}, 'x': 0.5, 'xanchor': 'center'}, height=440 if k == 4 else 560, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor=COLORS['card'], plot_bgcolor=COLORS['card'], font={'family': 'Inter, sans-serif', 'color': COLORS['text']})
        return fig

    @app.callback(Output('timeseries', 'figure'), Input('run-dropdown', 'value'), Input('router-dropdown', 'value'))
    def update_timeseries(run_id, router_id):
        df = fetch_run_dataframe(run_id)
        sub = df[df['router_id'] == router_id].sort_values('t')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub['t'], y=sub['occ'], mode='lines', name='Actual', line=dict(color='#222', width=2)))
        models = models_for(run_id)
        has_models = any((m is not None and (not isinstance(m, PersistenceBaseline)) for n, m in models.items() if n != 'Persistence'))
        if has_models:
            try:
                preds = predict_run(models, run_id, window=WINDOW)
                for name, arr in preds.items():
                    if name == 'Persistence':
                        color = '#888'
                    elif name == 'MLP':
                        color = '#1f77b4'
                    elif name == 'GRU':
                        color = '#2ca02c'
                    else:
                        color = '#d62728'
                    y_pred = arr[router_id]
                    ts = list(range(len(y_pred)))
                    fig.add_trace(go.Scatter(x=ts, y=y_pred, mode='lines', name=f'{name} pred', line=dict(color=color, dash='dash')))
            except Exception as exc:
                fig.add_annotation(x=0, y=1, xref='paper', yref='paper', text=f'Predict error: {exc}', showarrow=False, font=dict(color='red'))
        fig.update_layout(title={'text': f'<b>Router {router_id}</b> - {run_id}', 'font': {'size': 14, 'color': COLORS['text']}, 'x': 0.5, 'xanchor': 'center'}, xaxis_title='Sample index (t)', yaxis_title='Buffer occupancy', height=420, margin=dict(l=20, r=20, t=50, b=50), paper_bgcolor=COLORS['card'], plot_bgcolor='#fafbfc', font={'family': 'Inter, sans-serif', 'color': COLORS['text']}, legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1, 'bgcolor': 'rgba(255,255,255,0.8)'}, xaxis={'gridcolor': COLORS['border'], 'showline': True, 'linecolor': COLORS['border']}, yaxis={'gridcolor': COLORS['border'], 'showline': True, 'linecolor': COLORS['border']})
        return fig
    return app

def build_metrics_table():
    csv_path = METRICS_DIR / 'comparison.csv'
    if not csv_path.is_file():
        return html.Div('Chưa có results/metrics/comparison.csv - chạy `python -m model.train` trước.', style={'color': COLORS['warning'], 'padding': '12px', 'background': '#fef3c7', 'borderRadius': '8px'})
    df = pd.read_csv(csv_path)
    pivot = df.pivot(index='model', columns='split', values='rmse').round(5)
    pivot.columns = [f'RMSE/{c}' for c in pivot.columns]
    pivot_mae = df.pivot(index='model', columns='split', values='mae').round(5)
    pivot_mae.columns = [f'MAE/{c}' for c in pivot_mae.columns]
    combined = pd.concat([pivot, pivot_mae], axis=1).reset_index()
    order = ['Persistence', 'MLP', 'GRU', 'STGNN']
    combined['__order'] = combined['model'].apply(lambda m: order.index(m) if m in order else len(order))
    combined = combined.sort_values('__order').drop(columns='__order').reset_index(drop=True)
    metric_cols = [c for c in combined.columns if c != 'model']
    min_per_col = {c: combined[c].min() for c in metric_cols}
    th_style = {'padding': '10px 14px', 'background': '#f8fafc', 'color': COLORS['muted'], 'fontWeight': '600', 'fontSize': '11px', 'textTransform': 'uppercase', 'letterSpacing': '0.05em', 'borderBottom': f"2px solid {COLORS['border']}", 'textAlign': 'right'}
    td_style_base = {'padding': '10px 14px', 'fontSize': '13px', 'borderBottom': f"1px solid {COLORS['border']}", 'textAlign': 'right', 'fontVariantNumeric': 'tabular-nums'}

    def model_badge(name):
        colors = {'Persistence': '#9ca3af', 'MLP': '#1f77b4', 'GRU': '#2ca02c', 'STGNN': COLORS['accent']}
        return html.Td(html.Span(name, style={'background': colors.get(name, '#9ca3af'), 'color': 'white', 'padding': '3px 10px', 'borderRadius': '6px', 'fontSize': '12px', 'fontWeight': '600'}), style={**td_style_base, 'textAlign': 'left'})
    rows = []
    for i in range(len(combined)):
        cells = [model_badge(combined.iloc[i]['model'])]
        for c in metric_cols:
            v = combined.iloc[i][c]
            is_best = abs(v - min_per_col[c]) < 1e-09
            style = {**td_style_base}
            if is_best:
                style['fontWeight'] = '700'
                style['color'] = COLORS['success']
                style['background'] = '#ecfdf5'
            cells.append(html.Td(f'{v:.5f}', style=style))
        rows.append(html.Tr(cells))
    return html.Table([html.Thead(html.Tr([html.Th('Model', style={**th_style, 'textAlign': 'left'})] + [html.Th(c, style=th_style) for c in metric_cols])), html.Tbody(rows)], style={'width': '100%', 'borderCollapse': 'collapse', 'borderRadius': '8px', 'overflow': 'hidden', 'border': f"1px solid {COLORS['border']}"})
if __name__ == '__main__':
    app = build_app()
    app.run(debug=True, host='127.0.0.1', port=int(os.environ.get('DASH_PORT', '8060')))
