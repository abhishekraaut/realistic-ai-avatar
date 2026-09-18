# Avatar Engine (Backend)

This is the Python backend that will act as the orchestrator for the Custom Interactive Avatar Engine.
It uses **FastAPI** for HTTP endpoints and **LiveKit** for real-time WebRTC streaming.

## 1. Prerequisites

- Python 3.10+
- [Docker](https://docs.docker.com/get-docker/) (to run the local LiveKit server)

## 2. Start the LiveKit Server

We use LiveKit to handle the low-latency WebRTC connections between the browser and our engine. 
Run the development server locally via Docker:

```bash
docker run -d --name livekit \
  -p 7880:7880 \
  -p 7881:7881 \
  -p 7882:7882/udp \
  livekit/livekit-server \
  --dev
```

*(This starts LiveKit with default dev credentials: `devkey` / `secret`)*

## 3. Setup the Python Environment

Create a virtual environment and install the dependencies:

```bash
cd engine
python -m venv venv
# Activate the venv:
# On Windows: venv\Scripts\activate
# On Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
```

## 4. Run the Engine API

Start the FastAPI server:

```bash
python main.py
```

The API will run on `http://localhost:8000`. 
It exposes a `/token` endpoint that your Next.js frontend will call to get permission to join the WebRTC room.
