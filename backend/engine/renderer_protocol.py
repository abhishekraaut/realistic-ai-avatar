from typing import Protocol, Optional
import numpy as np
from shared.media_types import MotionWindow

class RendererProtocol(Protocol):
    """
    Contract for rendering frames from motion windows.
    Must be completely independent of network protocols, LiveKit, or LLMs.
    """
    
    def initialize_identity(self, identity_reference: np.ndarray) -> None:
        """
        Loads the deterministic appearance/identity for the avatar.
        Args:
            identity_reference: E.g., an RGB image array [H, W, 3] or latent embedding.
        """
        ...
        
    def initialize_sequence(self, turn_context: dict) -> None:
        """
        Starts a new continuous generation sequence (e.g. user speaking turn).
        Args:
            turn_context: Metadata about the current turn.
        """
        ...
        
    def render_motion_window(self, motion: MotionWindow, temporal_context: Optional[dict] = None) -> np.ndarray:
        """
        Renders a single frame or batch of frames based on the provided motion.
        Args:
            motion: Target facial kinematics and pose.
            temporal_context: Previous frame outputs or latent states to ensure temporal smoothing.
        Returns:
            RGB image frame array (e.g. 512x512x3).
        """
        ...
        
    def update_temporal_context(self, rendered_frame: np.ndarray) -> dict:
        """
        Extracts temporal context from a rendered frame to be passed into the next render_motion_window call.
        """
        ...
        
    def cancel_turn(self, turn_id: str) -> None:
        """
        Aborts any ongoing rendering for a specific turn_id (e.g. due to user interruption).
        """
        ...
        
    def reset_sequence(self) -> None:
        """
        Clears all temporal context and returns the avatar to a neutral resting state.
        """
        ...
