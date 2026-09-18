"use client";

import { useEffect, useState } from "react";
import {
  LiveKitRoom,
  RoomAudioRenderer,
  VideoTrack,
  useTracks,
  useConnectionState,
} from "@livekit/components-react";
import { Track } from "livekit-client";
import "@livekit/components-styles";

export default function EnginePlayerPage() {
  const [token, setToken] = useState("");

  useEffect(() => {
    // Fetch a token from our custom Python backend
    const fetchToken = async () => {
      try {
        const res = await fetch("http://localhost:8000/token", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            room_name: "avatar-room-" + Math.floor(Math.random() * 1000),
            participant_name: "user-" + Math.floor(Math.random() * 1000),
          }),
        });
        const data = await res.json();
        setToken(data.token);
      } catch (e) {
        console.error("Failed to fetch token:", e);
      }
    };
    fetchToken();
  }, []);

  if (token === "") {
    return <div className="flex items-center justify-center h-screen bg-black text-white">Connecting to Custom Engine...</div>;
  }

  return (
    <LiveKitRoom
      video={false}
      audio={true}
      token={token}
      serverUrl="ws://localhost:7880"
      connect={true}
      data-lk-theme="default"
      style={{ height: "100vh", backgroundColor: "black" }}
    >
      <div className="flex flex-col items-center justify-center h-full text-white">
        <h1 className="text-2xl mb-4">Custom AI Engine (Audio Only)</h1>
        <p className="text-gray-400 mb-8">Phase 1: Connected to Local LiveKit Server</p>
        <ConnectionStatus />
      </div>
      <RoomAudioRenderer />
    </LiveKitRoom>
  );
}

function ConnectionStatus() {
  const state = useConnectionState();
  return <div>Status: {state}</div>;
}
