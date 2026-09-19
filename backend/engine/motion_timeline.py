import logging
import sys
import os
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.media_types import AudioFeatureWindow, MotionWindow, MediaTimestamp

logger = logging.getLogger("motion-timeline")
logger.setLevel(logging.INFO)

class StreamingMotionPredictor:
    """
    Consumes streaming AudioFeatureWindow objects and projects them into temporally 
    aligned MotionWindow segments. Acts as the CPU-mocked bridge for Latent Diffusion.
    """
    def __init__(self, target_fps=30.0):
        self.target_fps = target_fps
        self.current_turn_id = -1
        self.feature_buffer: List[AudioFeatureWindow] = []
        
        # Exact duration of one video frame mapped to the media timeline
        self.motion_frame_duration = 1.0 / target_fps
        self.last_motion_time = 0.0
        
    def cancel_turn(self, turn_id: int):
        """Discards buffered features for a cancelled turn."""
        if self.current_turn_id == turn_id:
            logger.info(f"[MotionTimeline] Cancelling turn {turn_id}")
            self.feature_buffer.clear()
            self.current_turn_id = -1
            self.last_motion_time = 0.0
            
    def append_features(self, features: List[AudioFeatureWindow]) -> List[MotionWindow]:
        """
        Ingests audio features and outputs discrete 30 FPS MotionWindows 
        when enough features are buffered to cross the next frame boundary.
        """
        if not features:
            return []
            
        # 1. Handle turn isolation / barge-in
        if features[0].turn_id != self.current_turn_id:
            logger.info(f"[MotionTimeline] Processing new turn: {features[0].turn_id}")
            self.current_turn_id = features[0].turn_id
            self.feature_buffer.clear()
            # Anchor to the physical start time of this turn's features
            self.last_motion_time = features[0].start_time
            
        self.feature_buffer.extend(features)
        
        motion_windows = []
        
        # 2. Extract motion frames whenever we have enough buffered future context
        while self.feature_buffer:
            next_motion_end_time = self.last_motion_time + self.motion_frame_duration
            
            # Check if we have enough context to fulfill this frame
            if self.feature_buffer[-1].end_time < next_motion_end_time:
                break
                
            # 3. Spatial/Temporal downsampling: Group all features belonging to this motion frame
            window_features = [
                f for f in self.feature_buffer
                if self.last_motion_time <= f.start_time < next_motion_end_time
            ]
            
            # 4. CPU-Mocked Latent Projection (Replaces DiT/Autoregressive model for now)
            if window_features:
                avg_energy = sum(f.rms_energy for f in window_features) / len(window_features)
                is_speaking = any(f.voicing for f in window_features)
                
                # Mocked Latents: [jaw_open, lip_pucker, head_pitch, head_yaw]
                jaw_open = min(1.0, avg_energy * 20.0) # Highly responsive to energy
                lip_pucker = 0.2 if is_speaking else 0.0
                latents = [jaw_open, lip_pucker, 0.0, 0.0]
            else:
                is_speaking = False
                latents = [0.0, 0.0, 0.0, 0.0]
                
            # 5. Emit synchronized MotionWindow
            sample_rate = features[0].sample_rate
            start_ts = MediaTimestamp(
                turn_id=self.current_turn_id,
                sample_position=int(self.last_motion_time * sample_rate),
                sample_rate=sample_rate
            )
            end_ts = MediaTimestamp(
                turn_id=self.current_turn_id,
                sample_position=int(next_motion_end_time * sample_rate),
                sample_rate=sample_rate
            )
            
            mw = MotionWindow(
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                motion_latents=latents,
                is_speaking=is_speaking
            )
            motion_windows.append(mw)
            
            # 6. Advance local clock and prune
            self.last_motion_time = next_motion_end_time
            # Keep features that might overlap the next frame boundary
            self.feature_buffer = [f for f in self.feature_buffer if f.end_time > self.last_motion_time]
            
        return motion_windows

    def flush(self) -> List[MotionWindow]:
        """Flushes any remaining features as a final motion frame."""
        if not self.feature_buffer:
            return []
            
        avg_energy = sum(f.rms_energy for f in self.feature_buffer) / len(self.feature_buffer)
        is_speaking = any(f.voicing for f in self.feature_buffer)
        jaw_open = min(1.0, avg_energy * 20.0)
        latents = [jaw_open, 0.0, 0.0, 0.0]
        
        sample_rate = self.feature_buffer[0].sample_rate
        start_ts = MediaTimestamp(
            turn_id=self.current_turn_id,
            sample_position=int(self.last_motion_time * sample_rate),
            sample_rate=sample_rate
        )
        end_ts = MediaTimestamp(
            turn_id=self.current_turn_id,
            sample_position=int(self.feature_buffer[-1].end_time * sample_rate),
            sample_rate=sample_rate
        )
        
        mw = MotionWindow(
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            motion_latents=latents,
            is_speaking=is_speaking
        )
        
        self.feature_buffer.clear()
        return [mw]
