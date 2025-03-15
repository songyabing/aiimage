# AI Image Generator

An AI-powered image captioning application using BLIP (Bootstrapping Language-Image Pre-training) model.

## Features

- Upload images (supports PNG, JPG, JPEG, WEBP formats)
- Generate AI-powered captions for images
- Clean and modern user interface
- Inspiration gallery
- Pricing information

## Technical Stack

- Python 3.x
- Flask
- Transformers (BLIP model)
- PyTorch
- PIL (Python Imaging Library)

## Setup

1. Install dependencies:
```bash
pip install flask transformers torch pillow
```

2. Run the application:
```bash
python app.py
```

The server will start on http://localhost:8080

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates
- `static/`: Static files (images, CSS)
- `config.ini`: Configuration file

## Note

This is a development server. For production deployment, use a production WSGI server.
