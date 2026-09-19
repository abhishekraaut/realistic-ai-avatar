import logging
import sys
import os
import torch
import numpy as np
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.media_types import AudioFeatureWindow, MotionWindow, MediaTimestamp
from models.motion_model import LearnedTemporalMotionModel

logger = logging.getLogger("learned-motion-timeline")
logger.setLevel(logging.INFO)

class LearnedStreamingMotionPredictor:
    """
    Replaces the heuristic StreamingMotionPredictor.
    Consumes streaming AudioFeatureWindow objects, uses a trained Neural Network,
    and projects them into temporally aligned MotionWindow segments.
    """
    def __init__(self, checkpoint_path: str, target_fps=30.0, context_frames=16):
        self.target_fps = target_fps
        self.current_turn_id = -1
        self.feature_buffer: List[AudioFeatureWindow] = []
        
        # Exact duration of one video frame mapped to the media timeline
        self.motion_frame_duration = 1.0 / target_fps
        self.last_motion_time = 0.0
        
        self.context_frames = context_frames
        self.history_mels = [] # To store the latest mel vectors
        
        self.device = torch.device("cpu")
        self.model = LearnedTemporalMotionModel(input_dim=80, hidden_dim=64, output_dim=4).to(self.device)
        self.model.eval()
        
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            logger.info(f"[LearnedMotionTimeline] Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warning(f"[LearnedMotionTimeline] Checkpoint not found at {checkpoint_path}. Using random weights!")

    def cancel_turn(self, turn_id: int):
        if self.current_turn_id == turn_id:
            logger.info(f"[LearnedMotionTimeline] Cancelling turn {turn_id}")
            self.feature_buffer.clear()
            self.history_mels.clear()
            self.current_turn_id = -1
            self.last_motion_time = 0.0
            
    def append_features(self, features: List[AudioFeatureWindow]) -> List[MotionWindow]:
        if not features:
            return []
            
        if features[0].turn_id != self.current_turn_id:
            logger.info(f"[LearnedMotionTimeline] Processing new turn: {features[0].turn_id}")
            self.current_turn_id = features[0].turn_id
            self.feature_buffer.clear()
            self.history_mels.clear()
            self.last_motion_time = features[0].start_time
            
        self.feature_buffer.extend(features)
        
        motion_windows = []
        
        while self.feature_buffer:
            next_motion_end_time = self.last_motion_time + self.motion_frame_duration
            
            if self.feature_buffer[-1].end_time < next_motion_end_time:
                break
                
            window_features = [
                f for f in self.feature_buffer
                if self.last_motion_time <= f.start_time < next_motion_end_time
            ]
            
            if window_features:
                is_speaking = any(f.voicing for f in window_features)
                # Average log-mels
                mels = np.array([f.log_mel for f in window_features])
                avg_mel = np.mean(mels, axis=0) # shape (80,)
                
                self.history_mels.append(avg_mel)
                if len(self.history_mels) > self.context_frames:
                    self.history_mels.pop(0)
                
                # Inference
                with torch.no_grad():
                    # Shape: (1, T_context, 80)
                    X = torch.tensor(np.array(self.history_mels), dtype=torch.float32).unsqueeze(0).to(self.device)
                    preds = self.model(X) # (1, T_context, 4)
                    latents = preds[0, -1, :].tolist() # Take the last timestep prediction
                    
            else:
                is_speaking = False
                latents = [0.0, 0.0, 0.0, 0.0]
                
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
            
            self.last_motion_time = next_motion_end_time
            self.feature_buffer = [f for f in self.feature_buffer if f.end_time > self.last_motion_time]
            
        return motion_windows

    def flush(self) -> List[MotionWindow]:
        if not self.feature_buffer:
            return []
            
        is_speaking = any(f.voicing for f in self.feature_buffer)
        mels = np.array([f.log_mel for f in self.feature_buffer])
        avg_mel = np.mean(mels, axis=0)
        
        self.history_mels.append(avg_mel)
        if len(self.history_mels) > self.context_frames:
            self.history_mels.pop(0)
            
        with torch.no_grad():
            X = torch.tensor(np.array(self.history_mels), dtype=torch.float32).unsqueeze(0).to(self.device)
            preds = self.model(X)
            latents = preds[0, -1, :].tolist()
        
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
        self.history_mels.clear()
        return [mw]
