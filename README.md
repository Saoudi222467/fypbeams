# Beam Stress Analysis

A web application for beam structural analysis, focusing on stress prediction for both simple and cantilever beams.

## Description

This application allows engineers and students to:
- Select between simple beam and cantilever beam for analysis
- Input geometric and load parameters
- Visualize the 3D beam with stress distribution 
- Get precise stress values at critical points

## Features

- Interactive web interface with real-time analysis
- 3D visualization with stress clearly highlighted
- Support for multiple beam types (simple and cantilever)
- Based on trained machine learning models for fast, accurate stress prediction

## Requirements

- Python 3.7+
- Flask
- NumPy
- Pandas
- Plotly
- Joblib
- Scikit-learn (for the ML models)

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install flask numpy pandas plotly scikit-learn joblib
   ```
3. **Important**: Download the cantilever beam model separately
   - The cantilever beam model (`cantilever_beam_model.pkl`) is a large file (~424MB) and is not included in this repository
   - Please contact the repository owner to obtain this file
   - Place the downloaded file in the root directory of the project
4. Run the application:
   ```
   python app.py
   ```
5. Open your browser and navigate to `http://127.0.0.1:5000/`

## Usage

1. Select the structure type (simple beam or cantilever beam)
2. For simple beam:
   - Enter width (b), thickness (h), force (F), and distance (d)
   - Click "Analyze Single Point Stress"
3. For cantilever beam:
   - Enter length (L), height (h), width (b), and transverse force (Fy)
   - Click "Analyze Cantilever Beam"
4. The visualization will show the 3D beam with stress visualization
5. Interact with the 3D model by rotating, zooming, and panning

## Project Structure

- `app.py`: Main Flask application with stress prediction logic
- `beam_model.pkl`: Trained machine learning model for simple beam analysis
- `templates/`: HTML templates
- `static/`: CSS and JavaScript files

## Future Enhancements

- Additional structural elements (columns, frames, trusses)
- Support for different loading conditions
- Export of analysis results
- User accounts and saved analysis history 

# Cantilever Beam Visualization

Interactive 3D visualization of cantilever beam stress analysis using Streamlit.

## Features

- Interactive 3D visualization of beam stress distribution
- Adjustable beam parameters (length, height, width, force)
- Support for both mesh-based and point-based visualization
- Option to use trained model or analytical solution
- Real-time updates with interactive sliders

## Deployment Instructions

### Option 1: Streamlit Cloud (Recommended)

1. Create a GitHub repository and push this code
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and main file (app.py)
6. Click "Deploy"

### Option 2: Hugging Face Spaces

1. Create a Hugging Face account
2. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
3. Click "Create new Space"
4. Choose "Streamlit" as the SDK
5. Push your code to the repository
6. The app will automatically deploy

### Option 3: Render

1. Create a Render account
2. Create a new Web Service
3. Connect your GitHub repository
4. Set the following:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py`
5. Click "Create Web Service"

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

## Usage

1. Adjust beam parameters using the sidebar sliders
2. Choose between mesh and point visualization
3. Optionally upload a trained model file
4. The visualization will update in real-time

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt