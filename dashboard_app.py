import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import uuid
import os
from datetime import datetime
import pickle

# Initialize data and model


def load_data_and_model():
    sample_df = pd.read_csv("data/sample_data.csv")
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return sample_df, model


sample_df, model = load_data_and_model()
current_row_index = 0

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport",
                "content": "width=device-width, initial-scale=1"}],
)

COLORS = {
    'background': '#F8F9FA',
    'text': '#2C3E50',
    'primary': '#3498DB',
    'success': '#2ECC71',
    'danger': '#E74C3C',
    'warning': '#F1C40F',
    'card': '#FFFFFF',
    'accent': '#9B59B6'
}


def create_card(title, content, color_class="primary"):
    """Helper function to create consistent card styling"""
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H4(
                    title,
                    className=f"text-{color_class} text-center"
                )
            ),
            dbc.CardBody(content),
        ],
        className="shadow-sm mb-4",
    )


app.layout = dbc.Container(
    [
        # Top Row with Header and Control Panel
        dbc.Row(
            [
                # Header on Left
                dbc.Col(
                    html.Div(
                        [
                            html.H1("CloudGuardian",
                                    className="display-4 fw-bold"),
                            html.P(
                                "Real-time anomaly detection and threat analysis",
                                className="lead text-muted",
                            ),
                        ],
                    ),
                    md=8,
                ),

                # Control Panel on Right
                dbc.Col(
                    create_card(
                        "Control Panel",
                        [
                            dbc.Button(
                                "Start Scanning",
                                id="start-btn",
                                color="success",
                                className="me-2 px-4 py-2 rounded-pill",
                                size="lg",
                            ),
                            dbc.Button(
                                "Stop Scanning",
                                id="stop-btn",
                                color="danger",
                                className="px-4 py-2 rounded-pill",
                                size="lg",
                            ),
                            dbc.Spinner(
                                html.Div(
                                    id="processing-info",
                                    className="mt-3 text-muted",
                                ),
                                color="primary",
                                type="grow",
                            ),
                        ],
                    ),
                    md=4,
                ),
            ],
            className="mb-4",
        ),

        # Main Content Row
        dbc.Row(
            [
                # Left Column for Tables
                dbc.Col(
                    [
                        # Normal Events Table
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
                                style_table={
                                    "overflowX": "auto",
                                    "overflowY": "auto",
                                    "borderRadius": "8px",
                                    "height": "250px",
                                },
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "12px",
                                    "backgroundColor": COLORS["card"],
                                    "fontFamily": "system-ui",
                                },
                                style_header={
                                    "backgroundColor": COLORS["success"],
                                    "color": "white",
                                    "fontWeight": "bold",
                                    "textTransform": "uppercase",
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

                        # Suspicious Events Table
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
                                style_table={
                                    "overflowX": "auto",
                                    "overflowY": "auto",
                                    "borderRadius": "8px",
                                    "height": "250px",
                                },
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "12px",
                                    "backgroundColor": COLORS["card"],
                                    "fontFamily": "system-ui",
                                },
                                style_header={
                                    "backgroundColor": COLORS["danger"],
                                    "color": "white",
                                    "fontWeight": "bold",
                                    "textTransform": "uppercase",
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
                    ],
                    md=6,
                ),

                # Right Column for Event Details
                dbc.Col(
                    create_card(
                        "Event Details",
                        html.Div(
                            id="selected-row-details",
                            className="p-3",
                            style={
                                "height": "calc(100vh - 250px)", "overflowY": "auto"}
                        ),
                    ),
                    md=6,
                ),
            ],
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
    style={"maxWidth": "1800px", "margin": "0 auto", "height": "100vh"}
)

# Callbacks


@app.callback(
    Output("stream-interval", "disabled"),
    [Input("start-btn", "n_clicks"), Input("stop-btn", "n_clicks")],
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
    global current_row_index, sample_df

    if not scanning:
        return current_idx, normal_data, suspicious_data, ""

    if current_row_index >= len(sample_df):
        current_row_index = 0
        return current_idx, normal_data, suspicious_data, "Completed scanning all rows."

    row = sample_df.iloc[current_row_index]
    features = pd.DataFrame([[
        row["processId"],
        row["parentProcessId"],
        row["userId"],
        row["mountNamespace"],
        row["eventId"],
        row["argsNum"],
        row["returnValue"]
    ]], columns=["processId", "parentProcessId", "userId", "mountNamespace",
                 "eventId", "argsNum", "returnValue"])

    prediction = model.predict(features)[0]

    item = {
        "row_id": str(current_row_index),
        "processId": int(row["processId"]),
        "parentProcessId": int(row["parentProcessId"]),
        "threadId": int(row["threadId"]),
        "processName": str(row["processName"]),
        "userId": int(row["userId"]),
        "mountNamespace": int(row["mountNamespace"]),
        "hostName": str(row["hostName"]),
        "eventId": int(row["eventId"]),
        "argsNum": int(row["argsNum"]),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "sus": prediction
    }

    if prediction == 0:
        normal_data.append(item)
        normal_data = normal_data[-5:]
    else:
        suspicious_data.append(item)
        suspicious_data = suspicious_data[-5:]

    info_msg = f"Processing row {current_row_index +
                                 1}/{len(sample_df)} - Event ID: {item['eventId']}"

    current_row_index += 1
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

    return dbc.Card(
        [
            dbc.CardHeader(
                html.H4(f"Event ID: {row_info['row_id']}",
                        className="text-center mb-0"),
                className=f"bg-{alert_color} text-white"
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Process Information",
                                           className="fw-bold"),
                            dbc.CardBody([
                                html.H6(f"Process ID: {row_info['processId']}",
                                        className="mb-2"),
                                html.H6(f"Parent Process ID: {row_info['parentProcessId']}",
                                        className="mb-2"),
                                html.H6(f"Thread ID: {row_info['threadId']}",
                                        className="mb-2"),
                                html.H6(f"Process Name: {row_info['processName']}",
                                        className="mb-2"),
                            ])
                        ], className="h-100")
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("System Information",
                                           className="fw-bold"),
                            dbc.CardBody([
                                html.H6(f"User ID: {row_info['userId']}",
                                        className="mb-2"),
                                html.H6(f"Mount Namespace: {row_info['mountNamespace']}",
                                        className="mb-2"),
                                html.H6(f"Host Name: {row_info['hostName']}",
                                        className="mb-2"),
                                html.H6(f"Event ID: {row_info['eventId']}",
                                        className="mb-2"),
                            ])
                        ], className="h-100")
                    ], md=6),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Event Status",
                                           className="fw-bold"),
                            dbc.CardBody([
                                html.H6(f"Timestamp: {row_info['timestamp']}",
                                        className="mb-2"),
                                html.H6(
                                    f"Status: {
                                        'Normal' if row_info['sus'] == 0 else 'Suspicious'}",
                                    className="mb-2",
                                    style={
                                        "color": COLORS['success']
                                        if row_info['sus'] == 0
                                        else COLORS['danger'],
                                        "fontWeight": "bold"
                                    }
                                ),
                            ])
                        ])
                    ])
                ])
            ])
        ],
        className="mt-3 shadow-sm",
        style={"minWidth": "800px"}
    )


if __name__ == "__main__":
    app.run_server(debug=True)
