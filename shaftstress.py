#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shaft Stress Analysis Module
Contains functions for analyzing stress in shafts due to torsion and bending.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_shaft_stress(d, L, T, M, theta, r, z, shaft_model, shaft_scaler):
    """
    Predict stress at a point in the shaft.
    
    Args:
        d: Shaft diameter (m)
        L: Shaft length (m)
        T: Torque (N·m)
        M: Bending moment (N·m)
        theta: Angular position (radians)
        r: Radial position (m)
        z: Axial position (m)
        shaft_model: ML model for stress prediction
        shaft_scaler: Scaler for input normalization
        
    Returns:
        float: Combined stress at the specified point (Pa)
    """
    # If model is not available, use analytical solution
    if shaft_model is None or shaft_scaler is None:
        # Calculate polar moment of inertia
        J = np.pi * d**4 / 32
        
        # Calculate second moment of area
        I = np.pi * d**4 / 64
        
        # Calculate torsional stress
        torsional_stress = T * r / J
        
        # Calculate bending stress (depends on angular position)
        bending_stress = M * r * np.cos(theta) / I
        
        # Combined stress (using simple superposition)
        # In reality, we might use von Mises for complex loading
        total_stress = torsional_stress + bending_stress
        
        return total_stress
    
    try:
        # Create a dataframe with the input parameters
        features = pd.DataFrame({
            'd': [d],        # Diameter
            'L': [L],        # Length
            'T': [T],        # Torque
            'M': [M],        # Bending moment
            'theta': [theta],  # Angular position
            'r': [r],        # Radial position
            'z': [z]         # Axial position
        })
        
        # Scale the inputs
        features_scaled = shaft_scaler.transform(features)
        
        # Make prediction
        stress = shaft_model.predict(features_scaled)[0][0]
        
        return stress
        
    except Exception as e:
        # Fallback to analytical solution
        return get_shaft_stress(d, L, T, M, theta, r, z, None, None)


def update_shaft_surface_stresses(xx, yy, zz, d, L, T, M, shaft_model, shaft_scaler, analytical_stress_cache=None, log_callback=None):
    """
    Helper function to calculate stress at shaft surface points
    
    Args:
        xx, yy, zz: Surface coordinate matrices
        d: Shaft diameter
        L: Shaft length
        T: Torque
        M: Bending moment
        shaft_model: Model for predictions
        shaft_scaler: Scaler for input normalization
        analytical_stress_cache: Cache of analytical solutions (optional)
        log_callback: Function for logging (optional)
        
    Returns:
        ndarray: Surface stress values
    """
    stress_surface = np.zeros_like(xx)
    
    # Debug surface coordinates
    if log_callback:
        log_callback(f"Surface coordinate ranges: x: {np.min(xx):.3f} to {np.max(xx):.3f}, y: {np.min(yy):.3f} to {np.max(yy):.3f}, z: {np.min(zz):.3f} to {np.max(zz):.3f}")
    
    radius = d/2
    
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            x, y, z = xx[i,j], yy[i,j], zz[i,j]
            
            # Convert from Cartesian to cylindrical coordinates for points on the surface
            # For surface points, r is approximately radius
            r = radius
            theta = np.arctan2(y, z)
            
            try:
                # Convert from surface-specific coordinates to shaft coordinates
                # In our system, x is along the shaft axis (z in standard cylindrical)
                if analytical_stress_cache is not None and x in analytical_stress_cache and theta in analytical_stress_cache[x]:
                    stress_surface[i,j] = analytical_stress_cache[x][theta]
                else:
                    # Calculate the stress at this point
                    stress_surface[i,j] = get_shaft_stress(d, L, T, M, theta, r, x, shaft_model, shaft_scaler)
            except Exception as e:
                if log_callback:
                    log_callback(f"Error calculating stress at point ({x}, {y}, {z}): {e}")
                # Fallback to analytical solution
                J = np.pi * d**4 / 32
                I = np.pi * d**4 / 64
                torsional_stress = T * r / J
                bending_stress = M * r * np.cos(theta) / I
                stress_surface[i,j] = torsional_stress + bending_stress
    
    # Debug stress values
    if log_callback:
        log_callback(f"Stress range for this surface: {np.min(stress_surface):.2e} to {np.max(stress_surface):.2e}")
    return stress_surface


def generate_shaft_stress_plot(d, L, T, M, shaft_model, shaft_scaler, num_points=40, output_path=None, log_callback=None):
    """
    Generate a 3D plot of a shaft with stress visualization (solid appearance)
    
    Args:
        d (float): Shaft diameter (m)
        L (float): Shaft length (m)
        T (float): Torque (N·m)
        M (float): Bending moment (N·m)
        shaft_model: Model for stress prediction
        shaft_scaler: Scaler for input normalization
        num_points (int): Resolution for visualization (defaults to 40)
        output_path (str): Path to save HTML file (if None, plot is not saved)
        log_callback (function): Function for logging output
        
    Returns:
        tuple: (figure object, maximum stress value, bool indicator if analytical solution was used)
    """
    try:
        # Create figure with 3D axis
        fig = make_subplots(specs=[[{'type': 'scene'}]])
        
        if log_callback:
            log_callback("Creating 3D shaft visualization with solid appearance and stress contours...")
        
        # Create a cylindrical grid for the shaft
        # The shaft will be oriented along the X axis
        theta = np.linspace(0, 2*np.pi, num_points)
        x = np.linspace(0, L, num_points)  # Points along the shaft axis
        r = d/2  # Radius of the shaft
        
        # Create grid for calculating stresses
        THETA, X = np.meshgrid(theta, x)
        R = np.ones_like(THETA) * r
        
        # Convert to Cartesian coordinates for plotting
        # X remains the same (shaft axis)
        Y = R * np.sin(THETA)  # Y is vertical
        Z = R * np.cos(THETA)  # Z is horizontal
        
        # Check if using model or analytical solution
        analytical_used = (shaft_model is None or shaft_scaler is None)
        
        if log_callback:
            if analytical_used:
                log_callback("Using analytical solution for shaft stress")
            else:
                log_callback("Using ML model for shaft stress prediction")
        
        # Calculate stresses over the surface of the shaft
        stresses = np.zeros_like(THETA)
        if log_callback:
            log_callback(f"Calculating stresses at {THETA.size} points...")
        
        # Pre-calculate analytical stress values for angular positions
        analytical_stress_cache = {}
        
        # Calculate polar moment of inertia and second moment of area once
        J = np.pi * d**4 / 32
        I = np.pi * d**4 / 64
        
        # Cache the stresses for each x and theta position
        for x_val in x:
            analytical_stress_cache[x_val] = {}
            for theta_val in theta:
                # For points on the surface, r = radius
                torsional_stress = T * r / J
                bending_stress = M * r * np.cos(theta_val) / I
                analytical_stress_cache[x_val][theta_val] = torsional_stress + bending_stress
        
        # Calculate stresses for all surface points
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x_val = X[i,j]
                theta_val = THETA[i,j]
                r_val = r  # All points are on the surface
                
                if analytical_used:
                    # Use cached analytical values
                    stresses[i,j] = analytical_stress_cache[x_val][theta_val]
                else:
                    # Use the ML model
                    try:
                        stresses[i,j] = get_shaft_stress(d, L, T, M, theta_val, r_val, x_val, shaft_model, shaft_scaler)
                    except Exception as e:
                        if log_callback:
                            log_callback(f"Error in model prediction: {e}. Using analytical solution.")
                        # Fallback to analytical
                        stresses[i,j] = analytical_stress_cache[x_val][theta_val]
        
        # Find maximum stress value
        max_stress = np.max(np.abs(stresses))
        max_stress_idx = np.argmax(np.abs(stresses.flatten()))
        max_stress_x = X.flatten()[max_stress_idx]
        max_stress_y = Y.flatten()[max_stress_idx]
        max_stress_z = Z.flatten()[max_stress_idx]
        
        # Get actual min/max for color scaling
        min_stress = np.min(stresses)
        max_stress = np.max(stresses)
        
        # Print stress ranges for verification
        if log_callback:
            log_callback(f"Stress range: {min_stress:.2e} Pa to {max_stress:.2e} Pa")
            log_callback(f"Maximum absolute stress: {max_stress:.2e} Pa at position ({max_stress_x:.2f}, {max_stress_y:.2f}, {max_stress_z:.2f})")
        
        # Ensure we get a symmetric colorscale centered at zero for clear visualization
        abs_max = max(abs(min_stress), abs(max_stress))
        
        # Update stress limits for symmetric colorscale if needed
        if abs_max > 0:
            cmin = -abs_max
            cmax = abs_max
            if log_callback:
                log_callback(f"Using symmetric color range: {cmin:.2e} Pa (blue) → 0 Pa (green) → {cmax:.2e} Pa (red)")
        else:
            cmin = min_stress
            cmax = max_stress
        
        # Create a colorscale similar to the other modules
        colorscale = [
            [0.0, 'rgb(0,0,128)'],      # Dark blue (minimum stress) 
            [0.125, 'rgb(0,0,255)'],    # Blue
            [0.25, 'rgb(0,128,255)'],   # Light blue
            [0.375, 'rgb(0,255,255)'],  # Cyan
            [0.5, 'rgb(0,255,0)'],      # Green (middle/zero)
            [0.625, 'rgb(255,255,0)'],  # Yellow
            [0.75, 'rgb(255,128,0)'],   # Orange
            [0.875, 'rgb(255,0,0)'],    # Red
            [1.0, 'rgb(128,0,0)']       # Dark red (maximum stress)
        ]
        
        # Add the shaft surface with stress coloring
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=stresses,
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
        
        # Add circular end caps to make the shaft look solid
        if log_callback:
            log_callback("Adding end caps for solid appearance...")
        
        # Create end cap at x = 0 (left end)
        theta_cap = np.linspace(0, 2*np.pi, num_points)
        r_cap = np.linspace(0, r, 15)  # Radial points from center to edge
        THETA_CAP, R_CAP = np.meshgrid(theta_cap, r_cap)
        
        # Convert to Cartesian for left end cap
        X_left = np.zeros_like(THETA_CAP)  # x = 0
        Y_left = R_CAP * np.sin(THETA_CAP)
        Z_left = R_CAP * np.cos(THETA_CAP)
        
        # Calculate stress for left end cap
        stress_left = np.zeros_like(THETA_CAP)
        for i in range(THETA_CAP.shape[0]):
            for j in range(THETA_CAP.shape[1]):
                theta_val = THETA_CAP[i,j]
                r_val = R_CAP[i,j]
                if r_val > 0:  # Avoid division by zero at center
                    stress_left[i,j] = get_shaft_stress(d, L, T, M, theta_val, r_val, 0, shaft_model, shaft_scaler)
        
        # Add left end cap
        fig.add_trace(go.Surface(
            x=X_left, y=Y_left, z=Z_left,
            surfacecolor=stress_left,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            opacity=1.0,
            lighting=dict(ambient=0.8, diffuse=0.9, roughness=0.4, fresnel=0.2)
        ))
        
        # Create end cap at x = L (right end)
        X_right = np.full_like(THETA_CAP, L)  # x = L
        Y_right = R_CAP * np.sin(THETA_CAP)
        Z_right = R_CAP * np.cos(THETA_CAP)
        
        # Calculate stress for right end cap
        stress_right = np.zeros_like(THETA_CAP)
        for i in range(THETA_CAP.shape[0]):
            for j in range(THETA_CAP.shape[1]):
                theta_val = THETA_CAP[i,j]
                r_val = R_CAP[i,j]
                if r_val > 0:  # Avoid division by zero at center
                    stress_right[i,j] = get_shaft_stress(d, L, T, M, theta_val, r_val, L, shaft_model, shaft_scaler)
        
        # Add right end cap
        fig.add_trace(go.Surface(
            x=X_right, y=Y_right, z=Z_right,
            surfacecolor=stress_right,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            opacity=1.0,
            lighting=dict(ambient=0.8, diffuse=0.9, roughness=0.4, fresnel=0.2)
        ))
        
        # Add circular edges along the shaft for better visual definition
        n_circles = 10
        for i in range(n_circles + 1):
            x_pos = i * L / n_circles
            circle_x = np.ones(num_points+1) * x_pos
            circle_y = r * np.sin(np.linspace(0, 2*np.pi, num_points+1))
            circle_z = r * np.cos(np.linspace(0, 2*np.pi, num_points+1))
            
            fig.add_trace(go.Scatter3d(
                x=circle_x, y=circle_y, z=circle_z,
                mode='lines',
                line=dict(
                    color='black',
                    width=2
                ),
                showlegend=False
            ))
        
        # Add longitudinal lines along the shaft
        n_long_lines = 8
        for i in range(n_long_lines):
            angle = i * 2*np.pi / n_long_lines
            line_x = np.linspace(0, L, num_points)
            line_y = r * np.sin(angle) * np.ones(num_points)
            line_z = r * np.cos(angle) * np.ones(num_points)
            
            fig.add_trace(go.Scatter3d(
                x=line_x, y=line_y, z=line_z,
                mode='lines',
                line=dict(
                    color='black',
                    width=2
                ),
                showlegend=False
            ))
        
        # Add force/moment indicators
        # Torque indicator (circular arrow at the end)
        if T != 0:
            arrow_radius = r * 1.2
            arrow_x = np.ones(num_points+1) * L
            arrow_y = arrow_radius * np.sin(np.linspace(0, 2*np.pi, num_points+1))
            arrow_z = arrow_radius * np.cos(np.linspace(0, 2*np.pi, num_points+1))
            
            fig.add_trace(go.Scatter3d(
                x=arrow_x, y=arrow_y, z=arrow_z,
                mode='lines',
                line=dict(
                    color='red',
                    width=4
                ),
                name=f"Torque: {T} N·m"
            ))
            
            # Add arrow head
            arrow_head_x = [L, L, L]
            arrow_head_y = [arrow_radius, arrow_radius*1.1, arrow_radius*0.9]
            arrow_head_z = [0, 0, 0]
            
            fig.add_trace(go.Scatter3d(
                x=arrow_head_x, y=arrow_head_y, z=arrow_head_z,
                mode='lines',
                line=dict(
                    color='red',
                    width=4
                ),
                showlegend=False
            ))
        
        # Bending moment indicator (arrow at the end)
        if M != 0:
            # Add a vertical arrow for bending moment
            moment_x = [L, L]
            moment_y = [0, r*2]
            moment_z = [0, 0]
            
            fig.add_trace(go.Scatter3d(
                x=moment_x, y=moment_y, z=moment_z,
                mode='lines',
                line=dict(
                    color='blue',
                    width=4
                ),
                name=f"Bending Moment: {M} N·m"
            ))
            
            # Add arrow head
            arrow_head_x = [L, L, L]
            arrow_head_y = [r*2, r*1.8, r*1.8]
            arrow_head_z = [0, 0.1*r, -0.1*r]
            
            fig.add_trace(go.Scatter3d(
                x=arrow_head_x, y=arrow_head_y, z=arrow_head_z,
                mode='lines',
                line=dict(
                    color='blue',
                    width=4
                ),
                showlegend=False
            ))
        
        # Add marker for maximum stress point
        fig.add_trace(go.Scatter3d(
            x=[max_stress_x],
            y=[max_stress_y],
            z=[max_stress_z],
            mode='markers',
            marker=dict(
                size=8,
                color='purple',
                symbol='diamond',
                line=dict(
                    color='black',
                    width=1
                ),
                opacity=1.0
            ),
            name=f'Max Stress: {max_stress:.2e} Pa'
        ))
        
        if log_callback:
            log_callback("Setting up camera and layout...")
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title='Length (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',  # This preserves the actual dimensions
                camera=dict(
                    eye=dict(x=1.8, y=1.2, z=0.8),
                    up=dict(x=0, y=0, z=1)
                ),
                # Reduce axis title font size
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
                text=f'Shaft Stress Analysis (d={d:.3f}m, L={L:.2f}m, T={T:.1f}N·m, M={M:.1f}N·m)',
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
            log_callback("Solid shaft stress plot completed successfully!")
        
        # Save the plot if an output path is provided
        if output_path:
            fig.write_html(output_path)
            if log_callback:
                log_callback(f"Plot saved to {output_path}")
            
        return fig, max_stress, analytical_used
    
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