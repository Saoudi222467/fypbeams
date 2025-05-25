#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Beam Analysis Module
Contains functions for analyzing stress in a simple beam with one fixed end.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_stress_profile(b, h, F, d, beam_model):
    """
    Uses the trained model to predict stress at the single fixed point (fixed side of beam)
    where d is perpendicular distance of force from center
    
    Args:
        b (float): Beam width (m)
        h (float): Beam thickness/height (m)
        F (float): Applied force (N)
        d (float): Distance from fixed end where force is applied (m)
        beam_model: Model for stress prediction or None for analytical solution
        
    Returns:
        tuple: (stress value, boolean indicating if analytical solution was used)
    """
    if beam_model is None:
        # Calculate second moment of area
        I = b * h**3 / 12
        # Calculate maximum stress at fixed end
        stress = F * d * (h/2) / I
        return stress, True  # Second value indicates analytical solution was used
    
    df = pd.DataFrame({
        'b (m)': [b],
        'h (m)': [h],
        'F (N)': [F],
        'd (m)': [d]
    })
    stress = beam_model.predict(df)[0]
    return stress, False  # Second value indicates model was used


def generate_beam_plot(b, h, F, d, beam_model, output_path=None, log_callback=None):
    """
    Generate a 3D plot of the simple beam with just a stress point visualization
    
    Args:
        b (float): Beam width (m)
        h (float): Beam thickness/height (m)
        F (float): Applied force (N)
        d (float): Distance from fixed end where force is applied (m)
        beam_model: Model for stress prediction or None for analytical solution
        output_path (str, optional): Path to save HTML file. If None, plot is not saved.
        log_callback (function, optional): Function to log output
        
    Returns:
        tuple: (figure object, maximum stress value, bool indicator if analytical solution was used)
    """
    # Calculate stress at fixed point (single point analysis)
    stress_at_fixed, analytical_used = get_stress_profile(b, h, F, d, beam_model)
    
    if log_callback:
        solution_type = "analytical" if analytical_used else "ML model"
        log_callback(f"Using {solution_type} for stress calculation")
        log_callback(f"Calculated stress at fixed point: {stress_at_fixed:.2e} Pa")
    
    # Create figure with 3D axis
    fig = make_subplots(specs=[[{'type': 'scene'}]])
    
    # Beam properties
    length = 1.0
    
    # Create a simple beam visualization without stress coloring
    # Define the vertices of the beam
    x = [0, length, length, 0, 0, length, length, 0]
    y = [-b/2, -b/2, b/2, b/2, -b/2, -b/2, b/2, b/2]
    z = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]
    
    # Define the faces (triangles) for the beam
    i = [0, 0, 7, 0, 0, 0, 6]
    j = [1, 2, 3, 4, 1, 5, 5]
    k = [2, 3, 4, 7, 5, 6, 7]
    
    # Create a beam using mesh3d (for a solid appearance)
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z, 
        i=i, j=j, k=k,
        color='lightgray',
        opacity=0.9,
        name='Beam'
    ))
    
    # Add beam edges for better definition
    edge_x, edge_y, edge_z = [], [], []
    
    # Bottom rectangle (z = -h/2)
    edge_x.extend([0, length, length, 0, 0])
    edge_y.extend([-b/2, -b/2, b/2, b/2, -b/2])
    edge_z.extend([-h/2, -h/2, -h/2, -h/2, -h/2])
    
    # Top rectangle (z = h/2)
    edge_x.extend([0, length, length, 0, 0])
    edge_y.extend([-b/2, -b/2, b/2, b/2, -b/2])
    edge_z.extend([h/2, h/2, h/2, h/2, h/2])
    
    # Connect top to bottom at x = 0
    edge_x.extend([0, 0, 0, 0])
    edge_y.extend([-b/2, -b/2, b/2, b/2])
    edge_z.extend([-h/2, h/2, h/2, -h/2])
    
    # Connect top to bottom at x = length
    edge_x.extend([length, length, length, length])
    edge_y.extend([-b/2, -b/2, b/2, b/2])
    edge_z.extend([-h/2, h/2, h/2, -h/2])
    
    # Add beam edges
    fig.add_trace(go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(
            color='black',
            width=3
        ),
        name="Beam Edges",
        showlegend=False
    ))
    
    # Add a marker at the fixed end (x=0) to indicate fixed support
    fig.add_trace(go.Scatter3d(
        x=[0, 0, 0, 0, 0], 
        y=[-b/2, b/2, b/2, -b/2, -b/2],
        z=[-h/2, -h/2, h/2, h/2, -h/2],
        mode='lines',
        line=dict(
            color='blue',
            width=6
        ),
        name="Fixed Support"
    ))
    
    # Add a red marker at the point of maximum stress (fixed end, center)
    fig.add_trace(go.Scatter3d(
        x=[0],  # Fixed end
        y=[0],  # Middle width
        z=[0],  # Middle height
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            symbol='circle',
            line=dict(
                color='black',
                width=1
            ),
            opacity=1.0
        ),
        name=f'Max Stress Point: {stress_at_fixed:.2e} Pa'
    ))
    
    # Show the force application point
    fig.add_trace(go.Scatter3d(
        x=[d],
        y=[0],
        z=[0],
        mode='markers+text',
        marker=dict(
            size=8,
            color='green',
            symbol='circle',  # Using a valid symbol
            opacity=0.8
        ),
        text=['F'],
        textposition='top center',
        name=f'Force Point: {F} N'
    ))
    
    # Add a force arrow (represented by a line pointing down)
    fig.add_trace(go.Scatter3d(
        x=[d, d],
        y=[0, 0],
        z=[h/4, -h/4],  # Arrow pointing down
        mode='lines',
        line=dict(
            color='green',
            width=5
        ),
        name=f"Force: {F} N"
    ))
    
    # Update the layout
    camera = dict(
        eye=dict(x=1.5, y=-1.5, z=1.2),  # Adjusted for good viewpoint
        up=dict(x=0, y=0, z=1)
    )
    
    # Set axis labels
    fig.update_layout(
        scene=dict(
            xaxis_title='Length (m)',
            yaxis_title='Width (m)',
            zaxis_title='Height (m)',
            aspectmode='data',
            camera=camera,
            xaxis=dict(range=[0, length], dtick=0.25),
            yaxis=dict(range=[-b/2, b/2]),
            zaxis=dict(range=[-h/2, h/2])
        ),
        title=dict(
            text=f'Simple Beam Stress Analysis (b={b:.3f}m, h={h:.3f}m, F={F:.1f}N, d={d:.3f}m)',
            x=0.5
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0, y=0),
        height=700,
        width=1000,
    )
    
    # Add annotation to emphasize stress analysis
    fig.add_annotation(
        x=0.5, 
        y=0.95,
        text=f"Maximum Stress: {stress_at_fixed:.2e} Pa",
        showarrow=False,
        xref="paper", 
        yref="paper",
        font=dict(size=14, color="black"),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="red",
        borderwidth=2,
        borderpad=4
    )
    
    if log_callback:
        log_callback("Plot creation complete")
    
    # Save the plot if an output path is provided
    if output_path:
        fig.write_html(output_path)
        if log_callback:
            log_callback(f"Plot saved to {output_path}")
    
    return fig, stress_at_fixed, analytical_used