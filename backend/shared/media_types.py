import dataclasses
from typing import Optional

@dataclasses.dataclass
class MediaTimestamp:
    """Represents a precise point in the media timeline."""
    turn_id: int
    sample_position: int
    sample_rate: int
    
    @property
    def time_seconds(self) -> float:
        return self.sample_position / self.sample_rate if self.sample_rate > 0 else 0.0

@dataclasses.dataclass
class AudioChunk:
    """A chunk of audio bound to the timeline."""
    timestamp: MediaTimestamp
    audio_data: bytes
    num_channels: int
    sample_count: int
    is_final: bool

@dataclasses.dataclass
class AudioFeatureWindow:
    """Temporally indexed speech features for a single analysis window."""
    turn_id: int
    sequence: int
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    sample_rate: int
    feature_hop: int
    
    # Features
    rms_energy: float
    pitch: float
    voicing: bool
    log_mel: list  # Using list of floats (or np.ndarray if we assume numpy everywhere)
    
@dataclasses.dataclass
class MotionWindow:
    """Animation window corresponding to a timeline segment."""
    start_timestamp: MediaTimestamp
    end_timestamp: MediaTimestamp
    motion_latents: list # Dummy representation for now
    is_speaking: bool

@dataclasses.dataclass
class TurnContext:
    """Information tracking an active conversational turn."""
    turn_id: int
    is_cancelled: bool = False
    
@dataclasses.dataclass
class AvatarState:
    """State enum wrapper."""
    state: str # IDLE, THINKING, SPEAKING
