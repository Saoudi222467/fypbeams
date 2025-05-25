    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Column Stress Analysis Module
Contains functions for analyzing stress in a column with eccentric loading.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_column_stress(h, b, L, F, eccentricity, x, y, z, column_model, column_scaler):
    """
    Predict stress at a point in the column.
    
    Args:
        h: Column height/thickness (m)
        b: Column width (m)
        L: Column length (m)
        F: Axial force (N)
        eccentricity: Eccentricity of the load (m)
        x, y, z: Position in the column for stress calculation
        column_model: Model for stress prediction
        column_scaler: Scaler for input normalization
        
    Returns:
        float: Combined compressive and bending stress at the specified point
    """
    # If model is not available, use analytical solution
    if column_model is None or column_scaler is None:
        # Calculate cross-sectional area
        A = b * h
        
        # Calculate moment of inertia for bending about z-axis
        I_z = b * h**3 / 12
        
        # Calculate direct stress from axial load
        direct_stress = -F / A  # Negative for compression
        
        # Calculate bending moment due to eccentricity
        M = F * eccentricity
        
        # Calculate bending stress (varies with y position)
        bending_stress = -M * y / I_z
        
        # Calculate total stress (compression + bending)
        total_stress = direct_stress + bending_stress
        
        return total_stress
    
    try:
        # Create a dataframe with the input parameters
        features = pd.DataFrame({
            'h': [h],
            'b': [b],
            'L': [L],
            'F': [F],
            'e': [eccentricity],
            'x': [x],
            'y': [y],
            'z': [z]
        })
        
        # Scale the inputs
        features_scaled = column_scaler.transform(features)
        
        # Make prediction
        stress = column_model.predict(features_scaled)[0][0]
        
        return stress
        
    except Exception as e:
        # Fallback to analytical solution
        return get_column_stress(h, b, L, F, eccentricity, x, y, z, None, None)


def update_column_surface_stresses(xx, yy, zz, h, b, L, F, eccentricity, column_model, column_scaler, analytical_stress_cache=None, log_callback=None):
    """
    Helper function to calculate stress at column surface points
    
    Args:
        xx, yy, zz: Surface coordinate matrices
        h, b, L: Column dimensions
        F: Applied force
        eccentricity: Load eccentricity
        column_model: Model for predictions
        column_scaler: Scaler for input normalization
        analytical_stress_cache: Cache of analytical solutions (optional)
        log_callback: Function for logging (optional)
        
    Returns:
        ndarray: Surface stress values
    """
    stress_surface = np.zeros_like(xx)
    
    # Debug surface coordinates
    if log_callback:
        log_callback(f"Surface coordinate ranges: x: {np.min(xx):.3f} to {np.max(xx):.3f}, y: {np.min(yy):.3f} to {np.max(yy):.3f}, z: {np.min(zz):.3f} to {np.max(zz):.3f}")
    
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            x, y, z = xx[i,j], yy[i,j], zz[i,j]
            try:
                # Check if we can use the cached value
                if analytical_stress_cache is not None and y in analytical_stress_cache:
                    stress_surface[i,j] = analytical_stress_cache[y]
                else:
                    # Calculate the stress at this point
                    stress_surface[i,j] = get_column_stress(h, b, L, F, eccentricity, x, y, z, column_model, column_scaler)
            except Exception as e:
                if log_callback:
                    log_callback(f"Error calculating stress at point ({x}, {y}, {z}): {e}")
                # Fallback to analytical solution
                A = b * h
                I_z = b * h**3 / 12
                direct_stress = -F / A
                M = F * eccentricity
                bending_stress = -M * y / I_z
                stress_surface[i,j] = direct_stress + bending_stress
    
    # Debug stress values
    if log_callback:
        log_callback(f"Stress range for this surface: {np.min(stress_surface):.2e} to {np.max(stress_surface):.2e}")
    return stress_surface


def generate_column_stress_plot(h, b, L, F, eccentricity, column_model, column_scaler, num_points=40, output_path=None, log_callback=None):
    """
    Generate a 3D plot of a column with stress visualization
    
    Args:
        h (float): Column height/thickness (m)
        b (float): Column width (m)
        L (float): Column length (m)
        F (float): Axial force (N)
        eccentricity (float): Eccentricity of load (m)
        column_model: Model for stress prediction
        column_scaler: Scaler for input normalization
        num_points (int, optional): Resolution for visualization. Defaults to 40.
        output_path (str, optional): Path to save HTML file. If None, plot is not saved.
        log_callback (function, optional): Function to log output
        
    Returns:
        tuple: (figure object, maximum stress value, bool indicator if analytical solution was used)
    """
    try:
        # Create figure with 3D axis
        fig = make_subplots(specs=[[{'type': 'scene'}]])
        
        if log_callback:
            log_callback("Creating 3D column visualization with stress contours...")
        
        # Create a higher resolution grid for smoother visualization
        # More points for length to show stress variation clearly
        x_points = np.linspace(0, L, num_points)
        # Points for cross-section
        y_points = np.linspace(-h/2, h/2, 15)
        z_points = np.linspace(-b/2, b/2, 15)
        
        # Using volumes for the more accurate and contiguous visualization
        # Create a 3D structured grid for smooth visualization
        X, Y, Z = np.meshgrid(x_points, y_points, z_points)
        
        # Reshape for processing
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        
        # Calculate stress at each point using the column model
        stresses = np.zeros(len(points))
        if log_callback:
            log_callback(f"Calculating stresses at {len(points)} points...")
        
        # Check if using model or analytical solution
        analytical_used = (column_model is None or column_scaler is None)
        
        if log_callback:
            if analytical_used:
                log_callback("Using analytical solution for column stress")
            else:
                log_callback("Using ML model for column stress prediction")
        
        # Pre-calculate analytical stress values for y positions
        analytical_stress_cache = {}
        
        # Calculate cross-sectional area and moment of inertia once
        A = b * h
        I_z = b * h**3 / 12
        
        # Calculate direct stress from axial load
        direct_stress = -F / A  # Negative for compression
        
        # Calculate bending moment due to eccentricity
        M = F * eccentricity
        
        # Cache the stresses for each y position
        for y in y_points:
            # Calculate bending stress (varies with y position)
            bending_stress = -M * y / I_z
            # Calculate total stress (compression + bending)
            analytical_stress_cache[y] = direct_stress + bending_stress
        
        # Calculate stresses for all points
        for i, point in enumerate(points):
            x, y, z = point
            stresses[i] = get_column_stress(h, b, L, F, eccentricity, x, y, z, column_model, column_scaler)
        
        # Find maximum compressive (negative) stress
        min_stress_idx = np.argmin(stresses)
        max_comp_stress_point = points[min_stress_idx]
        max_comp_stress = abs(stresses[min_stress_idx])  # Convert to positive for display
        
        # Find maximum tensile (positive) stress if any
        max_stress_idx = np.argmax(stresses)
        max_tens_stress = stresses[max_stress_idx]
        
        # Get actual min/max for color scaling
        min_stress = np.min(stresses)
        max_stress = np.max(stresses)
        
        # Print stress ranges for verification
        if log_callback:
            log_callback(f"Stress range: {min_stress:.2e} Pa (compression) to {max_stress:.2e} Pa (tension)")
            log_callback(f"Maximum compressive stress: {max_comp_stress:.2e} Pa at point {max_comp_stress_point}")
            if max_tens_stress > 0:
                log_callback(f"Maximum tensile stress: {max_tens_stress:.2e} Pa")
            else:
                log_callback("No tensile stress detected (all compression)")
        
        # Ensure we get a symmetric colorscale centered at zero for clear visualization
        # This makes compression blue and tension red with green at zero stress
        abs_max = max(abs(min_stress), abs(max_stress))
        
        # Update stress limits for symmetric colorscale if needed
        if abs_max > 0:
            cmin = -abs_max  # Blue for maximum compression
            cmax = abs_max   # Red for maximum tension
            if log_callback:
                log_callback(f"Using symmetric color range: {cmin:.2e} Pa (blue) → 0 Pa (green) → {cmax:.2e} Pa (red)")
        else:
            cmin = min_stress
            cmax = max_stress
        
        # Create a colorscale matching the reference image
        colorscale = [
            [0.0, 'rgb(0,0,128)'],      # Dark blue (maximum compression) 
            [0.125, 'rgb(0,0,255)'],    # Blue
            [0.25, 'rgb(0,128,255)'],   # Light blue
            [0.375, 'rgb(0,255,255)'],  # Cyan
            [0.5, 'rgb(0,255,0)'],      # Green (middle/zero)
            [0.625, 'rgb(255,255,0)'],  # Yellow
            [0.75, 'rgb(255,128,0)'],   # Orange
            [0.875, 'rgb(255,0,0)'],    # Red
            [1.0, 'rgb(128,0,0)']       # Dark red (maximum tension)
        ]
        
        # Create solid column visualization
        # Use six separate surface plots for each face of the column
        
        # Create higher resolution surfaces for smoother visualization
        surf_res = 50
        surf_x = np.linspace(0, L, surf_res)
        surf_y = np.linspace(-h/2, h/2, max(int(surf_res*h/L), 10))
        surf_z = np.linspace(-b/2, b/2, max(int(surf_res*b/L), 10))
        
        # Print dimensions of surface arrays for debugging
        if log_callback:
            log_callback(f"Surface array dimensions:")
            log_callback(f"surf_x: {surf_x.shape}, surf_y: {surf_y.shape}, surf_z: {surf_z.shape}")
    
        # Create and plot top surface (y = h/2)
        xx_top, zz_top = np.meshgrid(surf_x, surf_z)
        yy_top = np.full_like(xx_top, h/2)
        
        # Calculate stress for all surfaces using the column helper function
        stress_top = update_column_surface_stresses(xx_top, yy_top, zz_top, h, b, L, F, eccentricity, 
                                                  column_model, column_scaler, analytical_stress_cache, log_callback)
        
        # Create and plot bottom surface (y = -h/2)
        xx_bottom, zz_bottom = np.meshgrid(surf_x, surf_z)
        yy_bottom = np.full_like(xx_bottom, -h/2)
        stress_bottom = update_column_surface_stresses(xx_bottom, yy_bottom, zz_bottom, h, b, L, F, eccentricity, 
                                                     column_model, column_scaler, analytical_stress_cache, log_callback)
        
        # Create and plot front surface (z = -b/2)
        xx_front, yy_front = np.meshgrid(surf_x, surf_y)
        zz_front = np.full_like(xx_front, -b/2)
        stress_front = update_column_surface_stresses(xx_front, yy_front, zz_front, h, b, L, F, eccentricity, 
                                                    column_model, column_scaler, analytical_stress_cache, log_callback)
        
        # Create and plot back surface (z = b/2)
        xx_back, yy_back = np.meshgrid(surf_x, surf_y)
        zz_back = np.full_like(xx_back, b/2)
        stress_back = update_column_surface_stresses(xx_back, yy_back, zz_back, h, b, L, F, eccentricity, 
                                                   column_model, column_scaler, analytical_stress_cache, log_callback)
        
        # Create and plot left surface (x = 0)
        yy_left, zz_left = np.meshgrid(surf_y, surf_z)
        xx_left = np.full_like(yy_left, 0)
        stress_left = update_column_surface_stresses(xx_left, yy_left, zz_left, h, b, L, F, eccentricity, 
                                                   column_model, column_scaler, analytical_stress_cache, log_callback)
        
        # Create and plot right surface (x = L)
        yy_right, zz_right = np.meshgrid(surf_y, surf_z)
        xx_right = np.full_like(yy_right, L)
        stress_right = update_column_surface_stresses(xx_right, yy_right, zz_right, h, b, L, F, eccentricity, 
                                                    column_model, column_scaler, analytical_stress_cache, log_callback)
    
        # Add a simple marker at the origin to ensure something is always visible
        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            marker=dict(
                size=6,
                color='black',
                opacity=0.8
            ),
            name="Column Base",
            showlegend=False
        ))
    
        if log_callback:
            log_callback("Adding all surfaces to plot...")
        
        # Add all surfaces to plot
        # Top surface
        fig.add_trace(go.Surface(
            x=xx_top, y=yy_top, z=zz_top,
            surfacecolor=stress_top,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=True,
            colorbar=dict(
                title="Stress (Pa)",
                thickness=20,
                len=0.75,
                tickformat=".2e"
            ),
            opacity=1.0,
            lighting=dict(ambient=0.8, diffuse=0.9, roughness=0.4, fresnel=0.2)
        ))
        
        # Bottom surface
        fig.add_trace(go.Surface(
            x=xx_bottom, y=yy_bottom, z=zz_bottom,
            surfacecolor=stress_bottom,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            opacity=1.0,
            lighting=dict(ambient=0.8, diffuse=0.9, roughness=0.4, fresnel=0.2)
        ))
        
        # Front surface
        fig.add_trace(go.Surface(
            x=xx_front, y=yy_front, z=zz_front,
            surfacecolor=stress_front,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            opacity=1.0,
            lighting=dict(ambient=0.8, diffuse=0.9, roughness=0.4, fresnel=0.2)
        ))
        
        # Back surface
        fig.add_trace(go.Surface(
            x=xx_back, y=yy_back, z=zz_back,
            surfacecolor=stress_back,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            opacity=1.0,
            lighting=dict(ambient=0.8, diffuse=0.9, roughness=0.4, fresnel=0.2)
        ))
        
        # Left surface (bottom end)
        fig.add_trace(go.Surface(
            x=xx_left, y=yy_left, z=zz_left,
            surfacecolor=stress_left,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            opacity=1.0,
            lighting=dict(ambient=0.8, diffuse=0.9, roughness=0.4, fresnel=0.2)
        ))
        
        # Right surface (top end)
        fig.add_trace(go.Surface(
            x=xx_right, y=yy_right, z=zz_right,
            surfacecolor=stress_right,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=False, 
            opacity=1.0,
            lighting=dict(ambient=0.8, diffuse=0.9, roughness=0.4, fresnel=0.2)
        ))
        
        # Add column edges to make it look more solid and continuous
        # Create the 12 edges of the column
        edge_x, edge_y, edge_z = [], [], []
        
        # Bottom rectangle (y = -h/2)
        edge_x.extend([0, L, L, 0, 0])
        edge_y.extend([-h/2, -h/2, -h/2, -h/2, -h/2])
        edge_z.extend([-b/2, -b/2, b/2, b/2, -b/2])
        
        # Top rectangle (y = h/2)
        edge_x.extend([0, L, L, 0, 0])
        edge_y.extend([h/2, h/2, h/2, h/2, h/2])
        edge_z.extend([-b/2, -b/2, b/2, b/2, -b/2])
        
        # Connect top to bottom at x = 0
        edge_x.extend([0, 0, 0, 0])
        edge_y.extend([-h/2, h/2, h/2, -h/2])
        edge_z.extend([-b/2, -b/2, b/2, b/2])
        
        # Connect top to bottom at x = L
        edge_x.extend([L, L, L, L])
        edge_y.extend([-h/2, h/2, h/2, -h/2])
        edge_z.extend([-b/2, -b/2, b/2, b/2])
        
        # Add column edges
        fig.add_trace(go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(
                color='black',
                width=3
            ),
            name="Column Edges",
            showlegend=False
        ))
        
        # Add a force indicator at the top with eccentricity shown
        fig.add_trace(go.Scatter3d(
            x=[L, L],
            y=[eccentricity, eccentricity],
            z=[0, -b/3],  # Arrow pointing down
            mode='lines',
            line=dict(
                color='red',
                width=5
            ),
            name=f"Force: {F} N"
        ))
        
        # Add a marker for the force application point
        fig.add_trace(go.Scatter3d(
            x=[L],
            y=[eccentricity],
            z=[0],
            mode='markers+text',
            marker=dict(
                size=8,
                color='red',
                symbol='circle',
                opacity=0.8
            ),
            text=['F'],
            textposition='top center',
            name=f'Force: {F} N'
        ))
        
        # Add marker for maximum compressive stress point
        max_stress_x, max_stress_y, max_stress_z = max_comp_stress_point
        fig.add_trace(go.Scatter3d(
            x=[max_stress_x],
            y=[max_stress_y],
            z=[max_stress_z],
            mode='markers',
            marker=dict(
                size=8,
                color='blue',
                symbol='x',
                line=dict(
                    color='black',
                    width=1
                ),
                opacity=1.0
            ),
            name=f'Max Compressive Stress: {max_comp_stress:.2e} Pa'
        ))
        
        if log_callback:
            log_callback("Setting up camera and layout...")
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title='Length (m)',
                yaxis_title='Height (m)',
                zaxis_title='Width (m)',
                aspectmode='data',  # This preserves the actual dimensions
                camera=dict(
                    eye=dict(x=1.5, y=-1.8, z=1.2),
                    up=dict(x=0, y=0, z=1)
                ),
                # Axis font sizes
                xaxis=dict(
                    title=dict(
                        font=dict(size=10)
                    )
                ),
                yaxis=dict(
                    title=dict(
                        font=dict(size=10)
                    )
                ),
                zaxis=dict(
                    title=dict(
                        font=dict(size=10)
                    )
                ),
                bgcolor='rgba(255,255,255,1)'  # White background
            ),
            title=dict(
                text=f'Column Stress Analysis (L={L:.2f}m, h={h:.2f}m, b={b:.2f}m, F={F:.1f}N, e={eccentricity:.3f}m)',
                x=0.5,
                font=dict(size=14)
            ),
            margin=dict(l=0, r=0, b=20, t=40),
            legend=dict(x=0, y=0, font=dict(size=10)),
            height=700,
            width=1000,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Update colorbar font size
        for trace in fig.data:
            if isinstance(trace, go.Surface) and hasattr(trace, 'colorbar'):
                if trace.colorbar:
                    trace.colorbar.title.font.size = 10
                    trace.colorbar.tickfont.size = 9
        
        # Add stress range annotation
        min_formatted = f"{min_stress:.2e}"
        max_formatted = f"{max_stress:.2e}"
        fig.add_annotation(
            x=0.5,
            y=0.02,
            text=f"Stress Range: {min_formatted} Pa to {max_formatted} Pa",
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=10, color="black"),
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        )
        
        if log_callback:
            log_callback("Column stress plot completed successfully!")
        
        # Save the plot if an output path is provided
        if output_path:
            fig.write_html(output_path)
            if log_callback:
                log_callback(f"Plot saved to {output_path}")
            
        return fig, max_comp_stress, analytical_used
    
    except Exception as e:
        if log_callback:
            log_callback(f"Error in plot generation: {e}")
        # Create a simple placeholder figure if visualization fails
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=f"Error generating plot: {e}",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        
        if output_path:
            fig.write_html(output_path)
            if log_callback:
                log_callback(f"Error plot saved to {output_path}")
            
        return fig, 0, True