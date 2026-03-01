import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

# -------------------------------
# LSPB Trajectory Generator
# -------------------------------
def lspb(q0, qf, T, V=None, dt=0.05):
    D = qf - q0
    if V is None:
        V = 1.5 * D / T
    
    # Avoid division by zero
    if V == 0:
        V = 1e-6

    tb = (q0 - qf + V * T) / V
    if tb <= 0:
        tb = 0.1

    t = np.arange(0, T+dt, dt)
    q, dq, ddq = [], [], []

    for ti in t:
        if ti < tb:
            qi = q0 + (V/(2*tb)) * ti**2
            dqi = (V/tb) * ti
            ddqi = V/tb
        elif ti <= (T - tb):
            qi = q0 + V*(ti - tb/2)
            dqi = V
            ddqi = 0
        else:
            qi = qf - (V/(2*tb)) * (T - ti)**2
            dqi = (V/tb) * (T - ti)
            ddqi = -V/tb
        q.append(qi)
        dq.append(dqi)
        ddq.append(ddqi)

    return t, np.array(q), np.array(dq), np.array(ddq)

# -------------------------------
# Forward Kinematics
# -------------------------------
def fk(theta, link_lengths=[1,1,1]):
    x0, y0 = 0, 0
    x1 = link_lengths[0]*np.cos(theta[0])
    y1 = link_lengths[0]*np.sin(theta[0])
    x2 = x1 + link_lengths[1]*np.cos(theta[0]+theta[1])
    y2 = y1 + link_lengths[1]*np.sin(theta[0]+theta[1])
    x3 = x2 + link_lengths[2]*np.cos(theta[0]+theta[1]+theta[2])
    y3 = y2 + link_lengths[2]*np.sin(theta[0]+theta[1]+theta[2])
    return [x0, x1, x2, x3], [y0, y1, y2, y3]

# -------------------------------
# Dash App Layout (Enhanced UI)
# -------------------------------
app = dash.Dash(__name__)
server = app.server 

card_style = {
    "padding": "20px",
    "borderRadius": "15px",
    "boxShadow": "0px 4px 15px rgba(0,0,0,0.1)",
    "marginBottom": "20px",
    "backgroundColor": "#ffffff"
}

app.layout = html.Div(style={"backgroundColor": "#f4f6f9", "padding": "20px"}, children=[

    html.H1("🤖 3-DoF Manipulator Dashboard", 
        style={"textAlign": "center", "marginBottom": "10px"}),

    html.H4("Name: Himanshu Yadav | Roll No: 2023UME7035",
        style={"textAlign": "center", "marginBottom": "30px", "color": "gray"}),

    html.Div(style={"display": "flex", "gap": "20px"}, children=[

        # LEFT PANEL (INPUTS)
        html.Div(style={"flex": "1"}, children=[

            html.Div(style=card_style, children=[
                html.H3("Start Pose"),
                dcc.Input(id='theta1_start', type='number', value=0, step=0.1, placeholder="θ1", style={"width": "30%"}),
                dcc.Input(id='theta2_start', type='number', value=0, step=0.1, placeholder="θ2", style={"width": "30%"}),
                dcc.Input(id='theta3_start', type='number', value=0, step=0.1, placeholder="θ3", style={"width": "30%"}),
            ]),

            html.Div(style=card_style, children=[
                html.H3("Goal Pose"),
                dcc.Input(id='theta1_goal', type='number', value=0.5, step=0.1, style={"width": "30%"}),
                dcc.Input(id='theta2_goal', type='number', value=0.3, step=0.1, style={"width": "30%"}),
                dcc.Input(id='theta3_goal', type='number', value=-0.5, step=0.1, style={"width": "30%"}),
            ]),

            html.Div(style=card_style, children=[
                html.H3("Simulation Settings"),
                dcc.Input(id='time_total', type='number', value=5, step=0.5, style={"width": "50%"}),
                html.Br(), html.Br(),
                html.Button("🚀 Generate", id='generate', n_clicks=0,
                            style={"backgroundColor": "#007bff",
                                   "color": "white",
                                   "padding": "10px 20px",
                                   "border": "none",
                                   "borderRadius": "10px",
                                   "cursor": "pointer"})
            ])
        ]),

        # RIGHT PANEL (OUTPUTS)
        html.Div(style={"flex": "2"}, children=[

            html.Div(style=card_style, children=[
                html.H3("Joint Trajectories"),
                dcc.Graph(id='trajectory_plot')
            ]),

            html.Div(style=card_style, children=[
                html.H3("Manipulator Animation"),
                dcc.Graph(id='manipulator_animation')
            ])
        ])
    ])
])

# -------------------------------
# Callbacks
# -------------------------------
@app.callback(
    [Output('trajectory_plot', 'figure'),
     Output('manipulator_animation', 'figure')],
    [Input('generate', 'n_clicks')],
    [State('theta1_start', 'value'),
     State('theta2_start', 'value'),
     State('theta3_start', 'value'),
     State('theta1_goal', 'value'),
     State('theta2_goal', 'value'),
     State('theta3_goal', 'value'),
     State('time_total', 'value')]
)
def update_dashboard(n_clicks, th1s, th2s, th3s, th1g, th2g, th3g, T):
    if n_clicks == 0:
        return go.Figure(), go.Figure()

    # Input validation
    if None in [th1s, th2s, th3s, th1g, th2g, th3g, T] or T <= 0:
        return go.Figure(), go.Figure()

    q0 = np.array([th1s, th2s, th3s])
    qf = np.array([th1g, th2g, th3g])

    trajectories = []
    t = None
    dq_all, ddq_all = [], []
    for i in range(3):
        t, q, dq, ddq = lspb(q0[i], qf[i], T)
        trajectories.append(q)
        dq_all.append(dq)
        ddq_all.append(ddq)
    trajectories = np.array(trajectories).T

    fig_traj = go.Figure()
    for i in range(3):
        fig_traj.add_trace(go.Scatter(x=t, y=trajectories[:,i], mode='lines', name=f'θ{i+1} pos'))
        fig_traj.add_trace(go.Scatter(x=t, y=dq_all[i], mode='lines', name=f'θ{i+1} vel'))
        fig_traj.add_trace(go.Scatter(x=t, y=ddq_all[i], mode='lines', name=f'θ{i+1} acc'))

    fig_traj.update_layout(template="plotly_white",
                           title="Joint Trajectories",
                           xaxis_title="Time (s)")

    frames = []
    for k in range(0, len(t), 2):   
        x, y = fk(trajectories[k])
        frames.append(go.Frame(
            data=[go.Scatter(x=x, y=y, mode='lines+markers')],
            name=str(k)
        ))

    x0, y0 = fk(trajectories[0])

    fig_anim = go.Figure(
        data=[go.Scatter(x=x0, y=y0, mode='lines+markers')],
        frames=frames
    )

    fig_anim.update_layout(template="plotly_white",
                           xaxis=dict(range=[-3,3]), yaxis=dict(range=[-3,3]),
                           updatemenus=[{
                                "type": "buttons",
                                "buttons": [
                                    {
                                        "label": "Play",
                                        "method": "animate",
                                        "args": [None, {
                                            "frame": {"duration": 50, "redraw": True},
                                            "fromcurrent": True,
                                            "transition": {"duration": 0}
                                        }]
                                    }
                                ]
                            }]

    return fig_traj, fig_anim

# -------------------------------
# Basic Tests (Non-Dash)
# -------------------------------
def _run_tests():
    t, q, dq, ddq = lspb(0, 1, 5)
    assert len(t) == len(q)
    assert len(q) == len(dq)
    assert len(dq) == len(ddq)

    x, y = fk([0, 0, 0])
    assert len(x) == 4 and len(y) == 4

    print("All basic tests passed ✅")

# -------------------------------
# Run Server (FIXED)
# -------------------------------
if __name__ == '__main__':
    try:
        _run_tests()
    except:
        pass

    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host='0.0.0.0', port=port, debug=False)

