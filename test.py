import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import uuid
import os
from datetime import datetime

# Initialize the Dash app with a modern theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Modern color scheme
COLORS = {
    'background': '#F8F9FA',
    'text': '#212529',
    'primary': '#0D6EFD',
    'success': '#198754',
    'danger': '#DC3545',
    'warning': '#FFC107',
    'card': '#FFFFFF',
}

def create_card(title, content, color_class="primary"):
    """Helper function to create consistent card styling"""
    return dbc.Card(
        [
            dbc.CardHeader(html.H4(title, className=f"text-{color_class}")),
            dbc.CardBody(content),
        ],
        className="shadow-sm mb-4",
    )

# Layout with modern styling
app.layout = dbc.Container(
    [
        # Header
        html.Div(
            [
                html.H1("CloudGaurdian", className="display-4 mb-2"),
                html.P(
                    "Real-time anomaly detection and threat analysis",
                    className="lead text-muted",
                ),
                html.Hr(),
            ],
            className="text-center my-4",
        ),
        
        # Control Panel
        dbc.Row(
            [
                dbc.Col(
                    create_card(
                        "Control Panel",
                        [
                            dbc.Button(
                                "Start Scanning",
                                id="start-btn",
                                color="success",
                                className="me-2",
                            ),
                            dbc.Button(
                                "Stop Scanning",
                                id="stop-btn",
                                color="danger",
                            ),
                            html.Div(
                                id="processing-info",
                                className="mt-3 text-muted",
                            ),
                        ],
                    ),
                    md=12,
                ),
            ]
        ),

        # Data Tables
        dbc.Row(
            [
                # Normal Events
                dbc.Col(
                    create_card(
                        "Normal Events",
                        dash_table.DataTable(
                            id="normal-table",
                            columns=[
                                {"name": "Event ID", "id": "row_id"},
                                {"name": "Event Type", "id": "eventId"},
                                {"name": "Arguments", "id": "argsNum"},
                                {"name": "Timestamp", "id": "timestamp"},
                            ],
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "textAlign": "left",
                                "padding": "15px",
                                "backgroundColor": COLORS["card"],
                            },
                            style_header={
                                "backgroundColor": COLORS["primary"],
                                "color": "white",
                                "fontWeight": "bold",
                            },
                            style_data_conditional=[
                                {
                                    "if": {"row_index": "odd"},
                                    "backgroundColor": "#f8f9fa",
                                }
                            ],
                            page_size=5,
                        ),
                        "success",
                    ),
                    md=6,
                ),
                
                # Suspicious Events
                dbc.Col(
                    create_card(
                        "Suspicious Events",
                        dash_table.DataTable(
                            id="suspicious-table",
                            columns=[
                                {"name": "Event ID", "id": "row_id"},
                                {"name": "Event Type", "id": "eventId"},
                                {"name": "Arguments", "id": "argsNum"},
                                {"name": "Timestamp", "id": "timestamp"},
                            ],
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "textAlign": "left",
                                "padding": "15px",
                                "backgroundColor": COLORS["card"],
                            },
                            style_header={
                                "backgroundColor": COLORS["danger"],
                                "color": "white",
                                "fontWeight": "bold",
                            },
                            style_data_conditional=[
                                {
                                    "if": {"row_index": "odd"},
                                    "backgroundColor": "#f8f9fa",
                                }
                            ],
                            page_size=5,
                        ),
                        "danger",
                    ),
                    md=6,
                ),
            ]
        ),

        # Event Details
        dbc.Row(
            [
                dbc.Col(
                    create_card(
                        "Event Details",
                        html.Div(
                            id="selected-row-details",
                            className="p-3",
                        ),
                    ),
                    md=12,
                ),
            ]
        ),

        # Hidden components
        dcc.Interval(id="stream-interval", interval=1000, disabled=True),
        dcc.Store(id="scanning", data=False),
        dcc.Store(id="current-index", data=0),
        dcc.Store(id="normal-rows", data=[]),
        dcc.Store(id="suspicious-rows", data=[]),
    ],
    fluid=True,
    className="px-4 py-3",
)

# Callbacks remain mostly the same, but with added timestamp
@app.callback(
    Output("stream-interval", "disabled"),
    [Input("start-btn", "n_clicks"),
     Input("stop-btn", "n_clicks")],
    [State("stream-interval", "disabled")]
)
def toggle_interval(start_n, stop_n, current_state):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_state
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "start-btn":
        return False
    elif button_id == "stop-btn":
        return True
    return current_state

@app.callback(
    Output("scanning", "data"),
    [Input("stream-interval", "disabled")]
)
def update_scanning_state(disabled):
    return not disabled

@app.callback(
    [Output("current-index", "data"),
     Output("normal-rows", "data"),
     Output("suspicious-rows", "data"),
     Output("processing-info", "children")],
    [Input("stream-interval", "n_intervals"),
     Input("scanning", "data")],
    [State("current-index", "data"),
     State("normal-rows", "data"),
     State("suspicious-rows", "data")]
)
def process_row(n_intervals, scanning, current_idx, normal_data, suspicious_data):
    if not scanning:
        return current_idx, normal_data, suspicious_data, ""

    # Simulate data processing
    item = {
        "row_id": str(uuid.uuid4())[:6],
        "eventId": np.random.randint(1000, 9999),
        "argsNum": np.random.randint(1, 10),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "sus": np.random.choice([0, 1], p=[0.8, 0.2])
    }

    if item["sus"] == 0:
        normal_data.append(item)
        normal_data = normal_data[-5:]  # Keep last 5 rows
    else:
        suspicious_data.append(item)
        suspicious_data = suspicious_data[-5:]  # Keep last 5 rows

    info_msg = f"Processing events... Last event ID: {item['eventId']}"
    return current_idx + 1, normal_data, suspicious_data, info_msg

@app.callback(
    [Output("normal-table", "data"),
     Output("suspicious-table", "data")],
    [Input("normal-rows", "data"),
     Input("suspicious-rows", "data")]
)
def update_tables(normal_rows, suspicious_rows):
    return normal_rows, suspicious_rows

@app.callback(
    Output("selected-row-details", "children"),
    [Input("normal-table", "active_cell"),
     Input("suspicious-table", "active_cell")],
    [State("normal-table", "data"),
     State("suspicious-table", "data")]
)
def show_row_details(normal_active, suspicious_active, normal_data, suspicious_data):
    if not normal_active and not suspicious_active:
        return html.P("Select a row to view details", className="text-muted")

    ctx = dash.callback_context
    if not ctx.triggered:
        return html.P("Select a row to view details", className="text-muted")

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "normal-table" and normal_active:
        row_idx = normal_active["row"]
        row_info = normal_data[row_idx]
        alert_color = "success"
    elif trigger_id == "suspicious-table" and suspicious_active:
        row_idx = suspicious_active["row"]
        row_info = suspicious_data[row_idx]
        alert_color = "danger"
    else:
        return html.P("Select a row to view details", className="text-muted")

    return dbc.Alert(
        [
            html.H4(f"Event ID: {row_info['row_id']}", className="alert-heading"),
            html.Hr(),
            html.P(f"Event Type: {row_info['eventId']}", className="mb-0"),
            html.P(f"Arguments: {row_info['argsNum']}", className="mb-0"),
            html.P(f"Timestamp: {row_info['timestamp']}", className="mb-0"),
            html.P(f"Status: {'Normal' if row_info['sus'] == 0 else 'Suspicious'}", className="mb-0"),
        ],
        color=alert_color,
        className="mt-3",
    )

if __name__ == "__main__":
    app.run_server(debug=True)