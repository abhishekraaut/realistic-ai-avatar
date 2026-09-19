import numpy as np
import cv2
from typing import Protocol, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.media_types import TurnContext

class RendererProtocol(Protocol):
    def initialize_avatar(self, identity_config: dict):
        ...
        
    def initialize_sequence(self, turn_context: TurnContext):
        ...
        
    def render_motion_window(self, latents: list, pts: float, turn_id: int) -> np.ndarray:
        ...
        
    def update_temporal_context(self, frame: np.ndarray):
        ...
        
    def cancel_turn(self, turn_id: int):
        ...
        
    def reset_sequence(self):
        ...

class DiagnosticRenderer:
    """
    Renders a deterministic visual proxy of the avatar based explicitly on 
    motion latents to validate synchronization and data paths without a GPU.
    """
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.current_turn_id = -1
        
    def initialize_avatar(self, identity_config: dict):
        pass
        
    def initialize_sequence(self, turn_context: TurnContext):
        self.current_turn_id = turn_context.turn_id
        
    def render_motion_window(self, latents: list, pts: float, turn_id: int) -> np.ndarray:
        frame = np.full((self.height, self.width, 4), 200, dtype=np.uint8) # Light gray background
        
        jaw_open = latents[0]
        lip_pucker = latents[1]
        head_pitch = latents[2]
        head_yaw = latents[3]
        
        # Center coordinates
        cx = self.width // 2
        cy = self.height // 2
        
        # Apply head motion (mock)
        cx += int(head_yaw * 100)
        cy += int(head_pitch * 100)
        
        # Draw face (circle)
        cv2.circle(frame, (cx, cy), 300, (230, 230, 230, 255), -1)
        cv2.circle(frame, (cx, cy), 300, (100, 100, 100, 255), 5)
        
        # Draw eyes
        cv2.circle(frame, (cx - 100, cy - 80), 20, (50, 50, 50, 255), -1)
        cv2.circle(frame, (cx + 100, cy - 80), 20, (50, 50, 50, 255), -1)
        
        # Draw explicit mouth (controlled by Latents)
        mouth_y = cy + 120
        # Jaw open controls mouth height (min 10, max 160)
        mouth_height = max(10, int(10 + jaw_open * 150))
        # Lip pucker controls mouth width (base 160, puckers down to 60)
        mouth_width = max(60, int(160 - lip_pucker * 100))
        
        cv2.ellipse(frame, (cx, mouth_y), (mouth_width, mouth_height), 0, 0, 360, (30, 30, 30, 255), -1)
        
        # Draw metrics
        cv2.putText(frame, f"TURN ID: {turn_id}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0, 255), 4)
        cv2.putText(frame, f"PTS: {pts:.3f}s", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0, 255), 4)
        cv2.putText(frame, f"JAW: {jaw_open:.3f}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0, 255), 4)
        cv2.putText(frame, f"PUCKER: {lip_pucker:.3f}", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0, 255), 4)
        
        return frame
        
    def update_temporal_context(self, frame: np.ndarray):
        pass
        
    def cancel_turn(self, turn_id: int):
        if self.current_turn_id == turn_id:
            self.current_turn_id = -1
            
    def reset_sequence(self):
        self.current_turn_id = -1

class NeuralRenderer(DiagnosticRenderer):
    """
    ARCHITECTURE / UNTRAINED
    This is the placeholder for the full Temporal Diffusion Transformer.
    """
    def render_motion_window(self, latents: list, pts: float, turn_id: int) -> np.ndarray:
        # In a real environment, this invokes the PyTorch CUDA pipeline.
        # For now, it falls back to DiagnosticRenderer.
        return super().render_motion_window(latents, pts, turn_id)
