# Salus.ai — Privacy-First Multimodal Health & Nutrition Assistant

**Salus.ai** is a full-stack, locally hosted AI web and Progressive Web App (PWA) that combines computer vision and natural language processing to estimate meal calories from photos and answer general health questions—without sending sensitive user health data to third-party cloud APIs.

## Demo Video
> **[Watch the 60-Second Mobile Demo Video Here](PASTE_YOUR_YOUTUBE_LINK_HERE)**

## System Architecture
Salus.ai runs a dual-model inference pipeline built on Python, Flask, and PyTorch:
* **Vision Pipeline (~350 MB):** A pretrained image classification model processes uploaded or camera-captured meal photos locally to identify food items and estimate average caloric values.
* **Conversational Health Engine (~3 GB):** A local large language model (LLM) handles context-aware follow-up questions and maintains session history.
* **Privacy-by-Design:** Because health and dietary logs are sensitive, all model weights and inference run locally on Apple Silicon hardware rather than external commercial APIs.
* **Progressive Web App (PWA) Layer:** Configured with a custom `manifest.json`, SVG vector icon, and mobile viewport meta tags so users can install Salus.ai full-screen onto iOS and Android home screens.

## Engineering Challenges Solved
1. **Local-to-Public HTTPS Routing:** Exposed a local Flask server (`127.0.0.1:5055`) over encrypted HTTPS tunnels (`HTTP/2`) with `caffeinate` power-management hooks to prevent hardware sleep during active inference.
2. **SSL & Environment Isolation:** Resolved macOS Python certificate verification (`certifi`) and dependency isolation using a dedicated `.venv` virtual environment.
3. **Safety & Responsible AI:** Implemented clear medical disclaimers reminding users that outputs are informational estimates rather than clinical advice, alongside a one-click memory reset (`/reset`) for user data control.

## Local Setup & Installation
1. Clone the repository:
   ```bash
   git clone https://github.com
   cd salus-ai
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Launch the application:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5055` in your browser.

## Disclaimer
*Salus.ai is an educational software project. Nutritional estimates are rough averages and AI responses do not constitute medical advice.*
