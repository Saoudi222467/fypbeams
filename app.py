#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beam Stress Analysis - Web Application
Main Flask application that integrates all analysis modules.
"""

import os
import uuid
import joblib
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler  # Added import
from flask import Flask, render_template, request, jsonify, send_from_directory
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import tempfile

# Import module-specific functions
from simplebeam import generate_beam_plot
from cantileverbeam import generate_cantilever_beam_plot, CantileverBeamModel  # Import the class
from columnstress import generate_column_stress_plot
from shaftstress import generate_shaft_stress_plot  # Import the new shaft stress module

# Filter specific scikit-learn warnings about feature names
warnings.filterwarnings("ignore", category=UserWarning, 
                        message="X does not have valid feature names, but StandardScaler was fitted with feature names")

# Initialize Flask app
app = Flask(__name__)

# Create necessary directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(CURRENT_DIR, 'uploads')
MODEL_FOLDER = CURRENT_DIR  # Changed to use the current directory
PLOT_FOLDER = os.path.join(CURRENT_DIR, 'static', 'plots')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(PLOT_FOLDER, 'simple'), exist_ok=True)
os.makedirs(os.path.join(PLOT_FOLDER, 'cantilever'), exist_ok=True)
os.makedirs(os.path.join(PLOT_FOLDER, 'column'), exist_ok=True)
os.makedirs(os.path.join(PLOT_FOLDER, 'shaft'), exist_ok=True)  # New directory for shaft plots

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['PLOT_FOLDER'] = PLOT_FOLDER

# Global variables for models
beam_model = None
cantilever_model = None
column_model = None
column_scaler = None
shaft_model = None  # New variable for shaft model
shaft_scaler = None  # New variable for shaft scaler
model_status_messages = []


def load_models(model_dir=None):
    """Load all beam and column models from the specified directory"""
    global beam_model, cantilever_model, column_model, column_scaler, shaft_model, shaft_scaler, model_status_messages
    
    if model_dir is None:
        model_dir = app.config['MODEL_FOLDER']
    
    beam_model = None
    cantilever_model = None
    column_model = None
    column_scaler = None
    shaft_model = None
    shaft_scaler = None
    
    # Check if model files exist
    beam_model_path = os.path.join(model_dir, 'beam_model.pkl')
    cantilever_model_path = os.path.join(model_dir, 'cantilever_beam_model.pkl')
    column_model_path = os.path.join(model_dir, 'column_stress_model.h5')
    column_scaler_path = os.path.join(model_dir, 'column_scaler.pkl')
    shaft_model_path = os.path.join(model_dir, 'shaft_stress_model.h5')  # New shaft model path
    shaft_scaler_path = os.path.join(model_dir, 'shaft_scaler.pkl')  # New shaft scaler path
    
    model_status_messages = []
    
    try:
        if os.path.exists(beam_model_path):
            beam_model = joblib.load(beam_model_path)
            model_status_messages.append(f"✅ Beam model loaded successfully from: {beam_model_path}")
        else:
            model_status_messages.append(f"⚠️ beam_model.pkl not found at: {beam_model_path}")
            model_status_messages.append("   Simple beam analysis will use analytical solution.")
    except Exception as e:
        model_status_messages.append(f"❌ Error loading beam model: {e}")
    
    try:
        if os.path.exists(cantilever_model_path):
            loaded_model = joblib.load(cantilever_model_path)
            # Check if the loaded model is already a CantileverBeamModel instance
            if isinstance(loaded_model, CantileverBeamModel):
                cantilever_model = loaded_model
            else:
                # If it's a different type of model (e.g., scikit-learn model),
                # wrap it in a CantileverBeamModel
                cantilever_model = CantileverBeamModel(
                    model=loaded_model,
                    scaler=None,  # Adjust if you have a scaler
                    feature_names=['L', 'h', 'b', 'I_z', 'c', 'Fy', 'x', 'y', 'z']  # Default feature names
                )
            model_status_messages.append(f"✅ Cantilever beam model loaded successfully from: {cantilever_model_path}")
        else:
            model_status_messages.append(f"⚠️ cantilever_beam_model.pkl not found at: {cantilever_model_path}")
            model_status_messages.append("   Cantilever analysis will use analytical solution.")
    except Exception as e:
        model_status_messages.append(f"❌ Error loading cantilever model: {e}")
    
    # Load column model and scaler
    try:
        if os.path.exists(column_model_path) and os.path.exists(column_scaler_path):
            # Load TensorFlow model (h5 format)
            column_model = tf.keras.models.load_model(column_model_path)
            column_scaler = joblib.load(column_scaler_path)
            model_status_messages.append(f"✅ Column stress model loaded successfully from: {column_model_path}")
            model_status_messages.append(f"✅ Column scaler loaded successfully from: {column_scaler_path}")
        else:
            if not os.path.exists(column_model_path):
                model_status_messages.append(f"⚠️ column_stress_model.h5 not found at: {column_model_path}")
            if not os.path.exists(column_scaler_path):
                model_status_messages.append(f"⚠️ column_scaler.pkl not found at: {column_scaler_path}")
            model_status_messages.append("   Column stress analysis will use analytical solution.")
    except Exception as e:
        model_status_messages.append(f"❌ Error loading column stress model: {e}")
    
    # Load shaft model and scaler
    try:
        if os.path.exists(shaft_model_path) and os.path.exists(shaft_scaler_path):
            # Load TensorFlow model (h5 format)
            shaft_model = tf.keras.models.load_model(shaft_model_path)
            shaft_scaler = joblib.load(shaft_scaler_path)
            model_status_messages.append(f"✅ Shaft stress model loaded successfully from: {shaft_model_path}")
            model_status_messages.append(f"✅ Shaft scaler loaded successfully from: {shaft_scaler_path}")
        else:
            if not os.path.exists(shaft_model_path):
                model_status_messages.append(f"⚠️ shaft_stress_model.h5 not found at: {shaft_model_path}")
            if not os.path.exists(shaft_scaler_path):
                model_status_messages.append(f"⚠️ shaft_scaler.pkl not found at: {shaft_scaler_path}")
            model_status_messages.append("   Shaft stress analysis will use analytical solution.")
    except Exception as e:
        model_status_messages.append(f"❌ Error loading shaft stress model: {e}")
        
    return beam_model, cantilever_model, column_model, column_scaler, shaft_model, shaft_scaler, model_status_messages


# Flask routes and view functions
@app.route('/')
def index():
    """Render the main page with tabs for beam analysis"""
    return render_template('index.html', model_status=model_status_messages)


@app.route('/simple_beam')
def simple_beam():
    """Render the simple beam analysis page"""
    return render_template('simple_beam.html', model_status=model_status_messages)


@app.route('/cantilever_beam')
def cantilever_beam():
    """Render the cantilever beam analysis page"""
    return render_template('cantilever_beam.html', model_status=model_status_messages)


@app.route('/column_stress')
def column_stress():
    """Render the column stress analysis page"""
    return render_template('column_stress.html', model_status=model_status_messages)


@app.route('/shaft_stress')
def shaft_stress():
    """Render the shaft stress analysis page"""
    return render_template('shaft_stress.html', model_status=model_status_messages)


@app.route('/analyze_simple_beam', methods=['POST'])
def analyze_simple_beam():
    """Handle simple beam analysis request"""
    try:
        # Get parameters from form
        b = float(request.form.get('width', 0.1))
        h = float(request.form.get('height', 0.1))
        F = float(request.form.get('force', 1000))
        d = float(request.form.get('distance', 0.5))
        
        # Generate a unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Create output path for the plot
        output_dir = os.path.join(app.config['PLOT_FOLDER'], 'simple')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"simple_beam_{analysis_id}.html")
        
        # Store log messages
        log_messages = []
        
        # Run analysis
        fig, stress, analytical_used = generate_beam_plot(
            b, h, F, d, beam_model,
            output_path=output_path,
            log_callback=lambda msg: log_messages.append(msg)
        )
        
        # Prepare results
        solution_type = "Analytical Solution" if analytical_used else "ML Model"
        results = {
            'max_stress': f"{stress:.2e}",
            'solution_type': solution_type,
            'plot_url': f"/plots/simple/simple_beam_{analysis_id}.html",
            'log_messages': log_messages,
            'analysis_id': analysis_id
        }
        
        return jsonify(results)
    
    except Exception as e:
        app.logger.error(f"Error in simple beam analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_cantilever_beam', methods=['POST'])
def analyze_cantilever_beam():
    """Handle cantilever beam analysis request"""
    try:
        # Get parameters from form
        L = float(request.form.get('length', 1.0))
        h = float(request.form.get('height', 0.1))
        b = float(request.form.get('width', 0.05))
        Fy = float(request.form.get('force', 100))
        
        # Get resolution parameter with default and bounds
        try:
            num_points = int(request.form.get('resolution', 30))
            num_points = max(10, min(num_points, 100))  # Clamp between 10 and 100
        except:
            num_points = 30
        
        # Generate a unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Create output path for the plot
        output_dir = os.path.join(app.config['PLOT_FOLDER'], 'cantilever')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"cantilever_beam_{analysis_id}.html")
        
        # Store log messages
        log_messages = []
        
        # Run analysis
        fig, max_stress, analytical_used = generate_cantilever_beam_plot(
            L, h, b, Fy, 
            cantilever_model, 
            num_points,
            output_path=output_path,
            log_callback=lambda msg: log_messages.append(msg)
        )
        
        # Prepare results
        solution_type = "Analytical Solution" if analytical_used else "ML Model"
        results = {
            'max_stress': f"{max_stress:.2e}",
            'solution_type': solution_type,
            'plot_url': f"/plots/cantilever/cantilever_beam_{analysis_id}.html",
            'log_messages': log_messages,
            'analysis_id': analysis_id
        }
        
        return jsonify(results)
    
    except Exception as e:
        app.logger.error(f"Error in cantilever beam analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_column_stress', methods=['POST'])
def analyze_column_stress():
    """Handle column stress analysis request"""
    try:
        # Get parameters from form
        h = float(request.form.get('height', 0.1))
        b = float(request.form.get('width', 0.1))
        L = float(request.form.get('length', 1.0))
        F = float(request.form.get('force', 10000))
        eccentricity = float(request.form.get('eccentricity', 0.01))
        
        # Get resolution parameter with default and bounds
        try:
            num_points = int(request.form.get('resolution', 30))
            num_points = max(10, min(num_points, 100))  # Clamp between 10 and 100
        except:
            num_points = 30
        
        # Generate a unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Create output path for the plot
        output_dir = os.path.join(app.config['PLOT_FOLDER'], 'column')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"column_stress_{analysis_id}.html")
        
        # Store log messages
        log_messages = []
        
        # Run analysis
        fig, max_stress, analytical_used = generate_column_stress_plot(
            h, b, L, F, eccentricity,
            column_model, column_scaler,
            num_points,
            output_path=output_path,
            log_callback=lambda msg: log_messages.append(msg)
        )
        
        # Prepare results
        solution_type = "Analytical Solution" if analytical_used else "ML Model"
        results = {
            'max_stress': f"{max_stress:.2e}",
            'solution_type': solution_type,
            'plot_url': f"/plots/column/column_stress_{analysis_id}.html",
            'log_messages': log_messages,
            'analysis_id': analysis_id
        }
        
        return jsonify(results)
    
    except Exception as e:
        app.logger.error(f"Error in column stress analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_shaft_stress', methods=['POST'])
def analyze_shaft_stress():
    """Handle shaft stress analysis request"""
    try:
        # Get parameters from form
        d = float(request.form.get('diameter', 0.05))
        L = float(request.form.get('length', 0.5))
        T = float(request.form.get('torque', 100))
        M = float(request.form.get('moment', 50))
        
        # Get resolution parameter with default and bounds
        try:
            num_points = int(request.form.get('resolution', 30))
            num_points = max(10, min(num_points, 100))  # Clamp between 10 and 100
        except:
            num_points = 30
        
        # Generate a unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Create output path for the plot
        output_dir = os.path.join(app.config['PLOT_FOLDER'], 'shaft')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"shaft_stress_{analysis_id}.html")
        
        # Store log messages
        log_messages = []
        
        # Run analysis
        fig, max_stress, analytical_used = generate_shaft_stress_plot(
            d, L, T, M,
            shaft_model, shaft_scaler,
            num_points,
            output_path=output_path,
            log_callback=lambda msg: log_messages.append(msg)
        )
        
        # Prepare results
        solution_type = "Analytical Solution" if analytical_used else "ML Model"
        results = {
            'max_stress': f"{max_stress:.2e}",
            'solution_type': solution_type,
            'plot_url': f"/plots/shaft/shaft_stress_{analysis_id}.html",
            'log_messages': log_messages,
            'analysis_id': analysis_id
        }
        
        return jsonify(results)
    
    except Exception as e:
        app.logger.error(f"Error in shaft stress analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/upload_model', methods=['POST'])
def upload_model():
    """Handle model upload"""
    try:
        if 'model_file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['model_file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            # Save the file
            filename = file.filename
            file_path = os.path.join(app.config['MODEL_FOLDER'], filename)
            file.save(file_path)
            
            # Reload models
            global beam_model, cantilever_model, column_model, column_scaler, shaft_model, shaft_scaler, model_status_messages
            beam_model, cantilever_model, column_model, column_scaler, shaft_model, shaft_scaler, model_status_messages = load_models()
            
            return jsonify({
                'success': True,
                'message': f"Model {filename} uploaded and loaded successfully",
                'model_status': model_status_messages
            })
    
    except Exception as e:
        app.logger.error(f"Error uploading model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/plots/<path:filename>')
def get_plot(filename):
    """Serve plot files"""
    return send_from_directory(app.config['PLOT_FOLDER'], filename)


@app.route('/get_logs/<analysis_type>/<analysis_id>')
def get_logs(analysis_type, analysis_id):
    """Get logs for a specific analysis"""
    # This is a placeholder - in a real app, you'd store logs in a database or file
    return jsonify({'logs': f"Logs for {analysis_type} analysis {analysis_id} not available"})


# HTML templates as functions to avoid separate files
def create_templates():
    """Create template files in the templates folder"""
    os.makedirs('templates', exist_ok=True)
    
    # Base template - explicitly using UTF-8 encoding
    with open('templates/base.html', 'w', encoding='utf-8') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Beam Stress Analysis Tool{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            padding-top: 20px;
        }
        .nav-tabs {
            margin-bottom: 20px;
        }
        .log-container {
            height: 300px;
            overflow-y: auto;
            background-color: #f8f9fa;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-family: monospace;
        }
        .results-panel {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f8f9fa;
        }
        .plot-container {
            height: 700px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-top: 20px;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .model-status {
            margin-top: 15px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f8f9fa;
        }
    </style>
    {% block head %}{% endblock %}
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">Beam Stress Analysis Tool</h1>
        
        <ul class="nav nav-tabs">
            <li class="nav-item">
                <a class="nav-link {% block simple_active %}{% endblock %}" href="/simple_beam">Simple Beam Analysis</a>
            </li>
            <li class="nav-item">
                <a class="nav-link {% block cantilever_active %}{% endblock %}" href="/cantilever_beam">Cantilever Beam Analysis</a>
            </li>
            <li class="nav-item">
                <a class="nav-link {% block column_active %}{% endblock %}" href="/column_stress">Column Stress Analysis</a>
            </li>
            <li class="nav-item">
                <a class="nav-link {% block shaft_active %}{% endblock %}" href="/shaft_stress">Shaft Stress Analysis</a>
            </li>
        </ul>
        
        {% block content %}{% endblock %}
        
        <footer class="mt-5 text-center text-muted">
            <p>Beam Stress Analysis Tool &copy; 2025</p>
        </footer>
    </div>
    
    {% block scripts %}{% endblock %}
</body>
</html>
        ''')
    
    # Index template - explicitly using UTF-8 encoding
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write('''
{% extends "base.html" %}

{% block content %}
<div class="row mt-4">
    <div class="col-md-12">
        <div class="card">
            <div class="card-header">
                <h2>Welcome to Beam Stress Analysis Web Tool</h2>
            </div>
            <div class="card-body">
                <p class="lead">This web application allows you to perform stress analysis for various structural elements.</p>
                
                <p>Use the tabs above to navigate between analysis types:</p>
                <ul>
                    <li><strong>Simple Beam Analysis</strong> - Analyze stress in a simple beam with a fixed end and applied force</li>
                    <li><strong>Cantilever Beam Analysis</strong> - Analyze stress in a cantilever beam with a fixed end and force at the free end</li>
                    <li><strong>Column Stress Analysis</strong> - Analyze stress in a column with axial load and eccentricity</li>
                    <li><strong>Shaft Stress Analysis</strong> - Analyze stress in a shaft subjected to torsion and bending</li>
                </ul>
                
                <h3 class="mt-4">Model Status</h3>
                <div class="model-status">
                    {% if model_status %}
                        {% for msg in model_status %}
                            <div>{{ msg }}</div>
                        {% endfor %}
                    {% else %}
                        <div>No model status information available. Models will be loaded when you first access an analysis page.</div>
                    {% endif %}
                </div>
                
                <h3 class="mt-4">Upload Models</h3>
                <p>You can upload trained beam stress models (.pkl or .h5 files):</p>
                <form id="model-upload-form" enctype="multipart/form-data" class="mb-3">
                    <div class="mb-3">
                        <label for="model_file" class="form-label">Select model file (.pkl or .h5)</label>
                        <input class="form-control" type="file" id="model_file" name="model_file" accept=".pkl,.h5">
                    </div>
                    <button type="submit" class="btn btn-primary">Upload Model</button>
                </form>
                <div id="upload-result" class="alert" style="display: none;"></div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
    $(document).ready(function() {
        $('#model-upload-form').submit(function(e) {
            e.preventDefault();
            
            var formData = new FormData(this);
            
            $.ajax({
                url: '/upload_model',
                type: 'POST',
                data: formData,
                contentType: false,
                processData: false,
                success: function(response) {
                    $('#upload-result').removeClass('alert-danger').addClass('alert-success')
                        .text(response.message).show();
                    
                    // Update model status
                    var statusHtml = '';
                    if (response.model_status) {
                        for (var i = 0; i < response.model_status.length; i++) {
                            statusHtml += '<div>' + response.model_status[i] + '</div>';
                        }
                        $('.model-status').html(statusHtml);
                    }
                },
                error: function(xhr) {
                    var errorMsg = 'Error uploading model';
                    if (xhr.responseJSON && xhr.responseJSON.error) {
                        errorMsg = xhr.responseJSON.error;
                    }
                    $('#upload-result').removeClass('alert-success').addClass('alert-danger')
                        .text(errorMsg).show();
                }
            });
        });
    });
</script>
{% endblock %}
        ''')
    
    # Create simple_beam.html template
    with open('templates/simple_beam.html', 'w', encoding='utf-8') as f:
        f.write('''
{% extends "base.html" %}

{% block title %}Simple Beam Analysis{% endblock %}

{% block simple_active %}active{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h3>Parameters</h3>
            </div>
            <div class="card-body">
                <form id="simple-beam-form">
                    <div class="mb-3">
                        <label for="width" class="form-label">Width (m)</label>
                        <input type="number" class="form-control" id="width" name="width" value="0.05" step="0.01" min="0.01" required>
                        <small class="form-text text-muted">Beam width</small>
                    </div>
                    <div class="mb-3">
                        <label for="height" class="form-label">Height (m)</label>
                        <input type="number" class="form-control" id="height" name="height" value="0.1" step="0.01" min="0.01" required>
                        <small class="form-text text-muted">Beam height/thickness</small>
                    </div>
                    <div class="mb-3">
                        <label for="force" class="form-label">Force (N)</label>
                        <input type="number" class="form-control" id="force" name="force" value="1000" step="100" required>
                        <small class="form-text text-muted">Applied force</small>
                    </div>
                    <div class="mb-3">
                        <label for="distance" class="form-label">Distance (m)</label>
                        <input type="number" class="form-control" id="distance" name="distance" value="0.5" step="0.1" min="0.1" required>
                        <small class="form-text text-muted">Distance from fixed end where force is applied</small>
                    </div>
                    <button type="submit" class="btn btn-primary">Calculate Stress</button>
                </form>
                
                <div id="results" class="results-panel" style="display: none;">
                    <h4>Results</h4>
                    <div id="max-stress"></div>
                    <div id="solution-type"></div>
                </div>
                
                <div class="model-status mt-3">
                    <h4>Model Status</h4>
                    {% if model_status %}
                        {% for msg in model_status %}
                            <div>{{ msg }}</div>
                        {% endfor %}
                    {% else %}
                        <div>No model information available</div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
    
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h3>Visualization</h3>
            </div>
            <div class="card-body">
                <div id="plot-message" class="text-center py-5">
                    <h4>Visualization will appear here</h4>
                    <p>Click "Calculate Stress" to generate a plot</p>
                </div>
                <div id="loading" class="loading">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Calculating stress and generating visualization...</p>
                </div>
                <div id="plot-container" class="plot-container" style="display: none;"></div>
                
                <div class="mt-3" id="plot-buttons" style="display: none;">
                    <a id="view-in-browser" href="#" target="_blank" class="btn btn-success">View in Browser</a>
                    <a id="download-plot" href="#" download class="btn btn-secondary">Download Plot</a>
                </div>
                
                <div class="mt-4">
                    <h4>Log</h4>
                    <div id="log-container" class="log-container"></div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
    $(document).ready(function() {
        $('#simple-beam-form').submit(function(e) {
            e.preventDefault();
            
            // Show loading indicator
            $('#plot-message').hide();
            $('#plot-container').hide();
            $('#plot-buttons').hide();
            $('#loading').show();
            $('#log-container').empty();
            $('#results').hide();
            
            // Get form data
            var formData = $(this).serialize();
            
            // Send AJAX request
            $.ajax({
                url: '/analyze_simple_beam',
                type: 'POST',
                data: formData,
                success: function(response) {
                    // Hide loading indicator
                    $('#loading').hide();
                    
                    // Display results
                    $('#max-stress').text('Maximum Stress: ' + response.max_stress + ' Pa');
                    $('#solution-type').text('Solution Type: ' + response.solution_type);
                    $('#results').show();
                    
                    // Update log
                    if (response.log_messages) {
                        for (var i = 0; i < response.log_messages.length; i++) {
                            $('#log-container').append('<div>' + response.log_messages[i] + '</div>');
                        }
                        // Scroll to bottom of log
                        $('#log-container').scrollTop($('#log-container')[0].scrollHeight);
                    }
                    
                    // Set up plot buttons
                    $('#view-in-browser').attr('href', response.plot_url);
                    $('#download-plot').attr('href', response.plot_url);
                    
                    // Display plot in iframe
                    $('#plot-container').html('<iframe src="' + response.plot_url + '" width="100%" height="100%" frameborder="0"></iframe>');
                    $('#plot-container').show();
                    $('#plot-buttons').show();
                },
                error: function(xhr) {
                    // Hide loading indicator
                    $('#loading').hide();
                    
                    // Show error message
                    var errorMsg = 'Error in analysis';
                    if (xhr.responseJSON && xhr.responseJSON.error) {
                        errorMsg = xhr.responseJSON.error;
                    }
                    
                    $('#log-container').append('<div class="text-danger">' + errorMsg + '</div>');
                    $('#plot-message').show();
                    $('#plot-message').html('<div class="alert alert-danger">' + errorMsg + '</div>');
                }
            });
        });
    });
</script>
{% endblock %}
        ''')
        
    # Create cantilever_beam.html template
    with open('templates/cantilever_beam.html', 'w', encoding='utf-8') as f:
        f.write('''
{% extends "base.html" %}

{% block title %}Cantilever Beam Analysis{% endblock %}

{% block cantilever_active %}active{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h3>Parameters</h3>
            </div>
            <div class="card-body">
                <form id="cantilever-beam-form">
                    <div class="mb-3">
                        <label for="length" class="form-label">Length (m)</label>
                        <input type="number" class="form-control" id="length" name="length" value="1.0" step="0.1" min="0.1" required>
                        <small class="form-text text-muted">Beam length</small>
                    </div>
                    <div class="mb-3">
                        <label for="height" class="form-label">Height (m)</label>
                        <input type="number" class="form-control" id="height" name="height" value="0.1" step="0.01" min="0.01" required>
                        <small class="form-text text-muted">Beam height/thickness</small>
                    </div>
                    <div class="mb-3">
                        <label for="width" class="form-label">Width (m)</label>
                        <input type="number" class="form-control" id="width" name="width" value="0.05" step="0.01" min="0.01" required>
                        <small class="form-text text-muted">Beam width</small>
                    </div>
                    <div class="mb-3">
                        <label for="force" class="form-label">Force (N)</label>
                        <input type="number" class="form-control" id="force" name="force" value="100" step="10" required>
                        <small class="form-text text-muted">Applied force at free end</small>
                    </div>
                    <div class="mb-3">
                        <label for="resolution" class="form-label">Resolution (10-100)</label>
                        <input type="number" class="form-control" id="resolution" name="resolution" value="30" step="1" min="10" max="100" required>
                        <small class="form-text text-muted">Higher values give smoother visualization but slower performance</small>
                    </div>
                    <button type="submit" class="btn btn-primary">Calculate Stress</button>
                </form>
                
                <div id="results" class="results-panel" style="display: none;">
                    <h4>Results</h4>
                    <div id="max-stress"></div>
                    <div id="solution-type"></div>
                </div>
                
                <div class="model-status mt-3">
                    <h4>Model Status</h4>
                    {% if model_status %}
                        {% for msg in model_status %}
                            <div>{{ msg }}</div>
                        {% endfor %}
                    {% else %}
                        <div>No model information available</div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
    
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h3>Visualization</h3>
            </div>
            <div class="card-body">
                <div id="plot-message" class="text-center py-5">
                    <h4>Visualization will appear here</h4>
                    <p>Click "Calculate Stress" to generate a plot</p>
                </div>
                <div id="loading" class="loading">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Calculating stress and generating visualization...</p>
                </div>
                <div id="plot-container" class="plot-container" style="display: none;"></div>
                
                <div class="mt-3" id="plot-buttons" style="display: none;">
                    <a id="view-in-browser" href="#" target="_blank" class="btn btn-success">View in Browser</a>
                    <a id="download-plot" href="#" download class="btn btn-secondary">Download Plot</a>
                </div>
                
                <div class="mt-4">
                    <h4>Log</h4>
                    <div id="log-container" class="log-container"></div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
    $(document).ready(function() {
        $('#cantilever-beam-form').submit(function(e) {
            e.preventDefault();
            
            // Show loading indicator
            $('#plot-message').hide();
            $('#plot-container').hide();
            $('#plot-buttons').hide();
            $('#loading').show();
            $('#log-container').empty();
            $('#results').hide();
            
            // Get form data
            var formData = $(this).serialize();
            
            // Send AJAX request
            $.ajax({
                url: '/analyze_cantilever_beam',
                type: 'POST',
                data: formData,
                success: function(response) {
                    // Hide loading indicator
                    $('#loading').hide();
                    
                    // Display results
                    $('#max-stress').text('Maximum Stress: ' + response.max_stress + ' Pa');
                    $('#solution-type').text('Solution Type: ' + response.solution_type);
                    $('#results').show();
                    
                    // Update log
                    if (response.log_messages) {
                        for (var i = 0; i < response.log_messages.length; i++) {
                            $('#log-container').append('<div>' + response.log_messages[i] + '</div>');
                        }
                        // Scroll to bottom of log
                        $('#log-container').scrollTop($('#log-container')[0].scrollHeight);
                    }
                    
                    // Set up plot buttons
                    $('#view-in-browser').attr('href', response.plot_url);
                    $('#download-plot').attr('href', response.plot_url);
                    
                    // Display plot in iframe
                    $('#plot-container').html('<iframe src="' + response.plot_url + '" width="100%" height="100%" frameborder="0"></iframe>');
                    $('#plot-container').show();
                    $('#plot-buttons').show();
                },
                error: function(xhr) {
                    // Hide loading indicator
                    $('#loading').hide();
                    
                    // Show error message
                    var errorMsg = 'Error in analysis';
                    if (xhr.responseJSON && xhr.responseJSON.error) {
                        errorMsg = xhr.responseJSON.error;
                    }
                    
                    $('#log-container').append('<div class="text-danger">' + errorMsg + '</div>');
                    $('#plot-message').show();
                    $('#plot-message').html('<div class="alert alert-danger">' + errorMsg + '</div>');
                }
            });
        });
    });
</script>
{% endblock %}
        ''')
    
    # Create column_stress.html template
    with open('templates/column_stress.html', 'w', encoding='utf-8') as f:
        f.write('''
{% extends "base.html" %}

{% block title %}Column Stress Analysis{% endblock %}

{% block column_active %}active{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h3>Parameters</h3>
            </div>
            <div class="card-body">
                <form id="column-stress-form">
                    <div class="mb-3">
                        <label for="height" class="form-label">Height/Thickness (m)</label>
                        <input type="number" class="form-control" id="height" name="height" value="0.1" step="0.01" min="0.01" required>
                        <small class="form-text text-muted">Column height/thickness</small>
                    </div>
                    <div class="mb-3">
                        <label for="width" class="form-label">Width (m)</label>
                        <input type="number" class="form-control" id="width" name="width" value="0.1" step="0.01" min="0.01" required>
                        <small class="form-text text-muted">Column width</small>
                    </div>
                    <div class="mb-3">
                        <label for="length" class="form-label">Length (m)</label>
                        <input type="number" class="form-control" id="length" name="length" value="1.0" step="0.1" min="0.1" required>
                        <small class="form-text text-muted">Column length</small>
                    </div>
                    <div class="mb-3">
                        <label for="force" class="form-label">Force (N)</label>
                        <input type="number" class="form-control" id="force" name="force" value="10000" step="1000" required>
                        <small class="form-text text-muted">Applied axial force</small>
                    </div>
                    <div class="mb-3">
                        <label for="eccentricity" class="form-label">Eccentricity (m)</label>
                        <input type="number" class="form-control" id="eccentricity" name="eccentricity" value="0.01" step="0.001" min="0" required>
                        <small class="form-text text-muted">Eccentricity of the load</small>
                    </div>
                    <div class="mb-3">
                        <label for="resolution" class="form-label">Resolution (10-100)</label>
                        <input type="number" class="form-control" id="resolution" name="resolution" value="30" step="1" min="10" max="100" required>
                        <small class="form-text text-muted">Higher values give smoother visualization but slower performance</small>
                    </div>
                    <button type="submit" class="btn btn-primary">Calculate Stress</button>
                </form>
                
                <div id="results" class="results-panel" style="display: none;">
                    <h4>Results</h4>
                    <div id="max-stress"></div>
                    <div id="solution-type"></div>
                </div>
                
                <div class="model-status mt-3">
                    <h4>Model Status</h4>
                    {% if model_status %}
                        {% for msg in model_status %}
                            <div>{{ msg }}</div>
                        {% endfor %}
                    {% else %}
                        <div>No model information available</div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
    
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h3>Visualization</h3>
            </div>
            <div class="card-body">
                <div id="plot-message" class="text-center py-5">
                    <h4>Visualization will appear here</h4>
                    <p>Click "Calculate Stress" to generate a plot</p>
                </div>
                <div id="loading" class="loading">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Calculating stress and generating visualization...</p>
                </div>
                <div id="plot-container" class="plot-container" style="display: none;"></div>
                
                <div class="mt-3" id="plot-buttons" style="display: none;">
                    <a id="view-in-browser" href="#" target="_blank" class="btn btn-success">View in Browser</a>
                    <a id="download-plot" href="#" download class="btn btn-secondary">Download Plot</a>
                </div>
                
                <div class="mt-4">
                    <h4>Log</h4>
                    <div id="log-container" class="log-container"></div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
    $(document).ready(function() {
        $('#column-stress-form').submit(function(e) {
            e.preventDefault();
            
            // Show loading indicator
            $('#plot-message').hide();
            $('#plot-container').hide();
            $('#plot-buttons').hide();
            $('#loading').show();
            $('#log-container').empty();
            $('#results').hide();
            
            // Get form data
            var formData = $(this).serialize();
            
            // Send AJAX request
            $.ajax({
                url: '/analyze_column_stress',
                type: 'POST',
                data: formData,
                success: function(response) {
                    // Hide loading indicator
                    $('#loading').hide();
                    
                    // Display results
                    $('#max-stress').text('Maximum Stress: ' + response.max_stress + ' Pa');
                    $('#solution-type').text('Solution Type: ' + response.solution_type);
                    $('#results').show();
                    
                    // Update log
                    if (response.log_messages) {
                        for (var i = 0; i < response.log_messages.length; i++) {
                            $('#log-container').append('<div>' + response.log_messages[i] + '</div>');
                        }
                        // Scroll to bottom of log
                        $('#log-container').scrollTop($('#log-container')[0].scrollHeight);
                    }
                    
                    // Set up plot buttons
                    $('#view-in-browser').attr('href', response.plot_url);
                    $('#download-plot').attr('href', response.plot_url);
                    
                    // Display plot in iframe
                    $('#plot-container').html('<iframe src="' + response.plot_url + '" width="100%" height="100%" frameborder="0"></iframe>');
                    $('#plot-container').show();
                    $('#plot-buttons').show();
                },
                error: function(xhr) {
                    // Hide loading indicator
                    $('#loading').hide();
                    
                    // Show error message
                    var errorMsg = 'Error in analysis';
                    if (xhr.responseJSON && xhr.responseJSON.error) {
                        errorMsg = xhr.responseJSON.error;
                    }
                    
                    $('#log-container').append('<div class="text-danger">' + errorMsg + '</div>');
                    $('#plot-message').show();
                    $('#plot-message').html('<div class="alert alert-danger">' + errorMsg + '</div>');
                }
            });
        });
    });
</script>
{% endblock %}
        ''')
    
    # Shaft stress template - explicitly using UTF-8 encoding
    with open('templates/shaft_stress.html', 'w', encoding='utf-8') as f:
        f.write('''
{% extends "base.html" %}

{% block title %}Shaft Stress Analysis{% endblock %}

{% block shaft_active %}active{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h3>Parameters</h3>
            </div>
            <div class="card-body">
                <form id="shaft-stress-form">
                    <div class="mb-3">
                        <label for="diameter" class="form-label">Diameter (m)</label>
                        <input type="number" class="form-control" id="diameter" name="diameter" value="0.05" step="0.001" min="0.001" required>
                        <small class="form-text text-muted">Shaft diameter</small>
                    </div>
                    <div class="mb-3">
                        <label for="length" class="form-label">Length (m)</label>
                        <input type="number" class="form-control" id="length" name="length" value="0.5" step="0.01" min="0.01" required>
                        <small class="form-text text-muted">Shaft length</small>
                    </div>
                    <div class="mb-3">
                        <label for="torque" class="form-label">Torque (N-m)</label>
                        <input type="number" class="form-control" id="torque" name="torque" value="100" step="10" required>
                        <small class="form-text text-muted">Applied torque</small>
                    </div>
                    <div class="mb-3">
                        <label for="moment" class="form-label">Bending Moment (N-m)</label>
                        <input type="number" class="form-control" id="moment" name="moment" value="50" step="10" required>
                        <small class="form-text text-muted">Applied bending moment</small>
                    </div>
                    <div class="mb-3">
                        <label for="resolution" class="form-label">Resolution (10-100)</label>
                        <input type="number" class="form-control" id="resolution" name="resolution" value="30" step="1" min="10" max="100" required>
                        <small class="form-text text-muted">Higher values give smoother visualization but slower performance</small>
                    </div>
                    <button type="submit" class="btn btn-primary">Calculate Stress</button>
                </form>
                
                <div id="results" class="results-panel" style="display: none;">
                    <h4>Results</h4>
                    <div id="max-stress"></div>
                    <div id="solution-type"></div>
                </div>
                
                <div class="model-status mt-3">
                    <h4>Model Status</h4>
                    {% if model_status %}
                        {% for msg in model_status %}
                            <div>{{ msg }}</div>
                        {% endfor %}
                    {% else %}
                        <div>No model information available</div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
    
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h3>Visualization</h3>
            </div>
            <div class="card-body">
                <div id="plot-message" class="text-center py-5">
                    <h4>Visualization will appear here</h4>
                    <p>Click "Calculate Stress" to generate a plot</p>
                </div>
                <div id="loading" class="loading">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Calculating stress and generating visualization...</p>
                </div>
                <div id="plot-container" class="plot-container" style="display: none;"></div>
                
                <div class="mt-3" id="plot-buttons" style="display: none;">
                    <a id="view-in-browser" href="#" target="_blank" class="btn btn-success">View in Browser</a>
                    <a id="download-plot" href="#" download class="btn btn-secondary">Download Plot</a>
                </div>
                
                <div class="mt-4">
                    <h4>Log</h4>
                    <div id="log-container" class="log-container"></div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
    $(document).ready(function() {
        $('#shaft-stress-form').submit(function(e) {
            e.preventDefault();
            
            // Show loading indicator
            $('#plot-message').hide();
            $('#plot-container').hide();
            $('#plot-buttons').hide();
            $('#loading').show();
            $('#log-container').empty();
            $('#results').hide();
            
            // Get form data
            var formData = $(this).serialize();
            
            // Send AJAX request
            $.ajax({
                url: '/analyze_shaft_stress',
                type: 'POST',
                data: formData,
                success: function(response) {
                    // Hide loading indicator
                    $('#loading').hide();
                    
                    // Display results
                    $('#max-stress').text('Maximum Stress: ' + response.max_stress + ' Pa');
                    $('#solution-type').text('Solution Type: ' + response.solution_type);
                    $('#results').show();
                    
                    // Update log
                    if (response.log_messages) {
                        for (var i = 0; i < response.log_messages.length; i++) {
                            $('#log-container').append('<div>' + response.log_messages[i] + '</div>');
                        }
                        // Scroll to bottom of log
                        $('#log-container').scrollTop($('#log-container')[0].scrollHeight);
                    }
                    
                    // Set up plot buttons
                    $('#view-in-browser').attr('href', response.plot_url);
                    $('#download-plot').attr('href', response.plot_url);
                    
                    // Display plot in iframe
                    $('#plot-container').html('<iframe src="' + response.plot_url + '" width="100%" height="100%" frameborder="0"></iframe>');
                    $('#plot-container').show();
                    $('#plot-buttons').show();
                },
                error: function(xhr) {
                    // Hide loading indicator
                    $('#loading').hide();
                    
                    // Show error message
                    var errorMsg = 'Error in analysis';
                    if (xhr.responseJSON && xhr.responseJSON.error) {
                        errorMsg = xhr.responseJSON.error;
                    }
                    
                    $('#log-container').append('<div class="text-danger">' + errorMsg + '</div>');
                    $('#plot-message').show();
                    $('#plot-message').html('<div class="alert alert-danger">' + errorMsg + '</div>');
                }
            });
        });
    });
</script>
{% endblock %}
        ''')


# Main execution
if __name__ == '__main__':
    # Create template directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create template files before starting the app
    create_templates()
    
    # Ensure all plot directories exist
    os.makedirs(os.path.join(PLOT_FOLDER, 'simple'), exist_ok=True)
    os.makedirs(os.path.join(PLOT_FOLDER, 'cantilever'), exist_ok=True)
    os.makedirs(os.path.join(PLOT_FOLDER, 'column'), exist_ok=True)
    os.makedirs(os.path.join(PLOT_FOLDER, 'shaft'), exist_ok=True)
    
    # Load models
    load_models()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)

# Streamlit app
def main():
    st.set_page_config(
        page_title="Cantilever Beam Visualization",
        page_icon="🏗️",
        layout="wide"
    )

    # Add custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stSlider {
            padding: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

    # Model class
    class CantileverBeamModel(nn.Module):
        def __init__(self, input_dim=5, hidden_dim=128, output_dim=3):
            super(CantileverBeamModel, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, x):
            return self.network(x)

    # Helper functions
    def create_mesh(length, height, width, num_points=20):
        x = np.linspace(0, length, num_points)
        y = np.linspace(-height/2, height/2, num_points)
        z = np.linspace(-width/2, width/2, num_points)
        X, Y, Z = np.meshgrid(x, y, z)
        return X, Y, Z

    def analytical_solution(x, y, z, length, height, width, force):
        # Simple analytical solution for demonstration
        # This is a simplified version - you should replace with actual beam theory
        stress = np.zeros_like(x)
        for i in range(len(x)):
            if x[i] <= length:
                stress[i] = force * (length - x[i]) * y[i] / (height**3 * width)
        return stress

    def visualize_beam(length, height, width, force, use_mesh=True, use_model=False, model_path=None):
        if use_mesh:
            # Create mesh grid
            X, Y, Z = create_mesh(length, height, width)
            points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        else:
            # Create point cloud
            num_points = 1000
            x = np.random.uniform(0, length, num_points)
            y = np.random.uniform(-height/2, height/2, num_points)
            z = np.random.uniform(-width/2, width/2, num_points)
            points = np.column_stack((x, y, z))

        # Calculate stresses
        if use_model and model_path:
            try:
                model = CantileverBeamModel()
                model.load_state_dict(torch.load(model_path))
                model.eval()
                with torch.no_grad():
                    inputs = torch.tensor(points, dtype=torch.float32)
                    stresses = model(inputs).numpy()
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                stresses = analytical_solution(points[:, 0], points[:, 1], points[:, 2], 
                                            length, height, width, force)
        else:
            stresses = analytical_solution(points[:, 0], points[:, 1], points[:, 2], 
                                        length, height, width, force)

        # Create 3D visualization
        fig = go.Figure(data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=stresses,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title='Stress')
                )
            )
        ])

        # Update layout
        fig.update_layout(
            title='Cantilever Beam Stress Visualization',
            scene=dict(
                xaxis_title='Length (m)',
                yaxis_title='Height (m)',
                zaxis_title='Width (m)',
                aspectmode='data'
            ),
            width=800,
            height=600
        )

        return fig

    # Main Streamlit app
    st.title("Cantilever Beam Visualization")
    st.write("Interactive 3D visualization of cantilever beam stress analysis")

    # Sidebar controls
    st.sidebar.header("Beam Parameters")
    length = st.sidebar.slider("Length (m)", 1.0, 10.0, 5.0)
    height = st.sidebar.slider("Height (m)", 0.1, 2.0, 0.5)
    width = st.sidebar.slider("Width (m)", 0.1, 2.0, 0.5)
    force = st.sidebar.slider("Force (N)", 100, 10000, 1000)
    
    # Visualization options
    st.sidebar.header("Visualization Options")
    use_mesh = st.sidebar.checkbox("Use Mesh Visualization", value=True)
    use_model = st.sidebar.checkbox("Use Trained Model", value=False)
    
    # Model upload
    if use_model:
        uploaded_file = st.sidebar.file_uploader("Upload Model File", type=['pth'])
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                model_path = tmp_file.name
        else:
            model_path = None
            st.sidebar.warning("Please upload a model file to use the trained model")
    else:
        model_path = None

    # Create visualization
    fig = visualize_beam(length, height, width, force, use_mesh, use_model, model_path)
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)

    # Add information about the visualization
    st.markdown("""
    ### About the Visualization
    - The color scale represents the stress distribution in the beam
    - Red indicates higher stress regions
    - Blue indicates lower stress regions
    - The beam is fixed at x=0 and loaded at x=length
    """)

if __name__ == "__main__":
    main()