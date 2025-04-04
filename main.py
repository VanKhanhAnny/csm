from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
import torch
import torchaudio
import json
import os

# Import your RealTimeVoiceSystem from its module (adjust the import as needed)
from test3 import RealTimeVoiceSystem

app = FastAPI()

# Initialize your voice system
device = "cuda" if torch.cuda.is_available() else "cpu"
voice_system = RealTimeVoiceSystem(device=device)

# Pre-load a voice prompt for voice_id=0 here:
voice_system.add_voice_prompt(
    voice_id=0,
    name="Voice A",
    text="This is a sample voice prompt for the first speaker.",
    audio_path="audio.wav"
)

# Ensure uploads folder exists
os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ---------------------------
# Placeholder Conversational Function (Updated)
# ---------------------------
def generate_conversational_reply(user_text: str) -> str:
    """
    Generates a conversational reply without echoing the user's text.
    Currently, this is a placeholder that randomly selects from preset responses.
    Replace or expand this function with a conversational model as needed.
    """
    import random
    responses = [
        "I'm doing well, thank you! How can I help you today?",
        "All good here. What can I do for you?",
        "I'm fine and ready to assist you. What's on your mind?",
        "I'm here to help! Let me know if you have any questions."
    ]
    return random.choice(responses)

# ---------------------------
# 1. Upload Voice Prompt Route
# ---------------------------
@app.post("/upload_prompt")
async def upload_prompt(
    voice_id: int = Form(...),
    name: str = Form(...),
    text: str = Form(...),
    file: UploadFile = File(...)
):
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        voice_system.add_voice_prompt(voice_id, name, text, file_location)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    
    return JSONResponse(content={"message": "Voice prompt uploaded successfully."})

# ---------------------------
# 2. Generate Speech Route
# ---------------------------
@app.post("/generate_speech")
async def generate_speech(voice_id: int = Form(...), text: str = Form(...)):
    try:
        audio = voice_system.generate_speech(text, voice_id, streaming=False)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    output_file = "output/generated_audio.wav"
    torchaudio.save(output_file, audio.unsqueeze(0).cpu(), voice_system.sample_rate)

    return JSONResponse(content={"message": "Speech generated successfully.", "audio_file": output_file})

# ---------------------------
# 3. Chat Endpoint (Conversational)
# ---------------------------
@app.post("/chat")
async def chat(voice_id: int = Form(...), text: str = Form(...)):
    # Generate a conversational reply (without echoing the input)
    response_text = generate_conversational_reply(text)
    
    try:
        audio = voice_system.generate_speech(response_text, voice_id, streaming=False)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    
    output_file = "output/generated_response.wav"
    torchaudio.save(output_file, audio.unsqueeze(0).cpu(), voice_system.sample_rate)
    
    return JSONResponse(content={
        "message": "Chat response generated.",
        "audio_file": output_file,
        "response": response_text
    })

# ---------------------------
# 4. WebSocket for Real-Time Audio Streaming
# ---------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                message = {"action": data}
            
            if message.get("action") == "speak":
                text = message.get("text", "")
                voice_id = message.get("voice", 0)
                try:
                    audio = voice_system.generate_speech(text, voice_id, streaming=False)
                    audio_bytes = audio.cpu().numpy().tobytes()
                    await websocket.send_bytes(audio_bytes)
                except Exception as e:
                    await websocket.send_text(f"Error generating speech: {str(e)}")
            else:
                await websocket.send_text("Unknown action")
    except WebSocketDisconnect:
        print("Client disconnected")

# ---------------------------
# Root Route (for testing)
# ---------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Voice Cloning API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
