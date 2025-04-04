import React, { useState, useEffect, useRef } from "react";

function VoiceDemo() {
  const [text, setText] = useState("");
  const [status, setStatus] = useState("Disconnected");
  const ws = useRef(null);
  const audioContext = useRef(null);

  const connectWebSocket = () => {
    console.log("Attempting to connect to WebSocket...");
    ws.current = new WebSocket("ws://127.0.0.1:8000/ws");
    ws.current.binaryType = "arraybuffer";

    ws.current.onopen = () => {
      console.log("WebSocket connected.");
      setStatus("Connected");
    };

    ws.current.onmessage = async (event) => {
      if (event.data instanceof ArrayBuffer) {
        console.log("Received audio data from server.");
        playAudio(event.data);
      } else {
        console.log("Received message:", event.data);
      }
    };

    ws.current.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    ws.current.onclose = () => {
      console.log("WebSocket disconnected. Attempting to reconnect in 2 seconds...");
      setStatus("Disconnected. Reconnecting...");
      setTimeout(connectWebSocket, 2000);
    };
  };

  useEffect(() => {
    // Initialize WebSocket connection
    connectWebSocket();

    // Initialize AudioContext for playback
    audioContext.current = new (window.AudioContext || window.webkitAudioContext)();

    // Clean up on unmount
    return () => {
      if (ws.current) ws.current.close();
      if (audioContext.current) audioContext.current.close();
    };
  }, []);

  // Function to decode and play audio data
  const playAudio = async (arrayBuffer) => {
    try {
      const decodedData = await audioContext.current.decodeAudioData(arrayBuffer);
      const source = audioContext.current.createBufferSource();
      source.buffer = decodedData;
      source.connect(audioContext.current.destination);
      source.start();
      console.log("Playing received audio...");
    } catch (error) {
      console.error("Error playing audio:", error);
    }
  };

  const sendText = (messageText) => {
    console.log("Attempting to send text:", messageText);
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      const message = { action: "speak", text: messageText, voice: 0 };
      console.log("Sending message:", message);
      ws.current.send(JSON.stringify(message));
    } else {
      console.warn("WebSocket not connected. Current state:", ws.current ? ws.current.readyState : "No ws instance");
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim() !== "") {
      sendText(text.trim());
      setText(""); // Clear the input field after sending
    }
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h1>Real-Time Voice Conversation</h1>
      <p>Status: {status}</p>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to speak"
          style={{ padding: "0.5rem", width: "300px" }}
        />
        <button type="submit" style={{ padding: "0.5rem", marginLeft: "1rem" }}>
          Send Text
        </button>
      </form>
    </div>
  );
}

export default VoiceDemo;
