"""Interactive 3D view of a point cloud (Plotly: rotate/zoom in the browser, no WebGL code of our own)."""

from __future__ import annotations

import numpy as np


def point_cloud_figure(points: np.ndarray, height: int = 480):
    import plotly.graph_objects as go

    p = np.asarray(points, dtype=np.float32)
    # model frame is Y-up; Plotly's 3D scene is Z-up, so plot (x, z, y)
    x, y, z = p[:, 0], p[:, 2], p[:, 1]
    fig = go.Figure(go.Scatter3d(
        x=x, y=y, z=z, mode="markers",
        marker=dict(size=2.2, color=z, colorscale="Viridis", opacity=0.9),
        hovertemplate="x %{x:.3f}<br>y %{z:.3f}<br>z %{y:.3f}<extra></extra>"))
    axis = dict(range=[-0.55, 0.55], showbackground=False, showticklabels=False, title="", zeroline=False)
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube",
                   camera=dict(eye=dict(x=1.35, y=1.35, z=0.85))),
        paper_bgcolor="rgba(0,0,0,0)")
    return fig
