# Ateg 

Ateg is an intelligent system that automatically generates gaming highlight reels by detecting kill events in gameplay footage. Using computer vision powered by YOLOv8, Ateg creates professional-quality montages with smooth transitions and background music - all with minimal human intervention.

## Features

- **AI-Powered Kill Detection**: Trained YOLOv8 model identifies kill events in gaming footage
- **Automatic Clip Extraction**: Extracts the most exciting moments from hours of gameplay
- **Editing**: Adds transitions and effects
- **Background Music**: Integrates music tracks with proper audio mixing
- **Web Interface**: User-friendly Flask-based frontend for configuration and preview

## How It Works

### Technical Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Video Input    │────▶│  Kill Detection │────▶│  Clip Extraction│
│                 │     │  (YOLOv8)       │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Final Output   │◀────│  Audio Mixing   │◀────│  Add Transitions│
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Workflow in Detail

1. **Kill Detection with YOLOv8**:
   - A YOLOv8 model was trained on labeled gaming images
   - The model recognizes specific visual patterns that indicate a kill event

2. **Clip Extraction**:
   - Once kills are detected, the system extracts video segments around these events
   - Configurable time windows (e.g., 5 seconds before and after the kill)
   - Smart merging of consecutive kills to avoid choppy clips

3. **Video Processing**:
   - MoviePy handles the core video manipulation tasks
   - Integration with [video-editing-py-script](https://github.com/salaheddinek/video-editing-py-script) enables advanced transitions

4. **Audio Processing**:
   - Game audio is preserved during important moments
   - Background music is dynamically adjusted to match the intensity of gameplay

5. **Web Interface**:
   - Flask-based frontend for ease of use
   - Upload gameplay videos directly from the browser

## Installation


### Setup

```bash
# Clone the repository
git clone https://github.com/ShikharSomething/Ateg.git
cd Ateg

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Web Interface

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`


## Technical Details

### Video Processing Pipeline

The video processing uses a combination of libraries:

- **OpenCV**: Frame extraction and initial processing
- **MoviePy**: Core video manipulation and composition
- **video-editing-py-script**: Advanced transitions and effects
- **PyTorch**: Running the YOLOv8 model for detection

### Flask Implementation

The web interface is built with Flask and includes:

- RESTful API for processing requests
- WebSocket for real-time progress updates
- Form-based configuration of highlight parameters
