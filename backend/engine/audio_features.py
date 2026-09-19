import numpy as np
import librosa
from typing import List
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.media_types import AudioChunk, AudioFeatureWindow

logger = logging.getLogger("audio-features")
logger.setLevel(logging.INFO)

class StreamingAudioFeatureExtractor:
    """
    Transforms raw TTS audio stream into temporally indexed speech features.
    Maintains a rolling buffer to output features incrementally.
    """
    def __init__(self, sample_rate=24000, window_size=1024, hop_size=256, n_mels=80):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        
        self.current_turn_id = -1
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_start_sample = 0
        self.sequence = 0
        
    def cancel_turn(self, turn_id: int):
        """Immediately discards buffered audio for a cancelled turn."""
        if self.current_turn_id == turn_id:
            logger.info(f"[AudioFeatures] Cancelling turn {turn_id}")
            self._reset_buffer()
            self.current_turn_id = -1
            
    def _reset_buffer(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_start_sample = 0
        self.sequence = 0
        
    def append(self, chunk: AudioChunk) -> List[AudioFeatureWindow]:
        """
        Appends a new chunk of audio and extracts as many full feature windows 
        as possible from the current rolling buffer.
        """
        # Guard against cancelled/stale turns
        if chunk.timestamp.turn_id != self.current_turn_id:
            logger.info(f"[AudioFeatures] Processing new turn: {chunk.timestamp.turn_id}")
            self.current_turn_id = chunk.timestamp.turn_id
            self._reset_buffer()
            # The chunk's start position anchors the physical timeline
            self.buffer_start_sample = chunk.timestamp.sample_position
            
        # Parse PCM16 to float32
        if isinstance(chunk.audio_data, bytes):
            pcm_data = np.frombuffer(chunk.audio_data, dtype=np.int16)
            float_data = pcm_data.astype(np.float32) / 32768.0
        else:
            float_data = np.array(chunk.audio_data, dtype=np.float32)
            
        self.audio_buffer = np.concatenate([self.audio_buffer, float_data])
        
        windows = []
        # Roll the buffer and extract continuous windows
        while len(self.audio_buffer) >= self.window_size:
            window_audio = self.audio_buffer[:self.window_size]
            
            # 1. RMS Energy
            rms = float(np.sqrt(np.mean(window_audio**2)))
            
            # 2. Log-Mel Spectrogram (Fast CPU cached version)
            # Create Mel filterbank once if not created to avoid massive overhead
            if not hasattr(self, '_mel_basis'):
                self._mel_basis = librosa.filters.mel(sr=self.sample_rate, n_fft=self.window_size, n_mels=self.n_mels)
                # pre-compute Hann window
                self._hann_win = np.hanning(self.window_size).astype(np.float32)
                
            # Apply window, compute FFT magnitude
            fft_mag = np.abs(np.fft.rfft(window_audio * self._hann_win))
            fft_power = fft_mag ** 2
            
            # Apply mel filterbank
            mel_spec = np.dot(self._mel_basis, fft_power)
            mel_spec = np.clip(mel_spec, a_min=1e-10, a_max=None)
            log_mel = (10.0 * np.log10(mel_spec)).tolist()
            
            # 3. Ultra-Fast CPU-compatible Pitch / Voicing (Zero-Crossing Proxy)
            # To strictly guarantee RTF < 1.0 on constrained CPUs, we use ZCR proxy.
            try:
                # Count zero crossings
                zero_crossings = np.nonzero(np.diff(window_audio > 0))[0]
                num_zc = len(zero_crossings)
                
                # Rough approximation: frequency = (num_zc / 2) * (sample_rate / window_size)
                approx_freq = (num_zc / 2.0) * (self.sample_rate / self.window_size)
                
                if 65 <= approx_freq <= 300 and rms > 0.005:
                    pitch = float(approx_freq)
                else:
                    pitch = 0.0
            except Exception:
                pitch = 0.0
                
            # Basic energy & pitch thresholds for voicing flag
            voicing = bool(rms > 0.005 and pitch > 0)
            
            start_s = self.buffer_start_sample
            end_s = start_s + self.window_size
            
            fw = AudioFeatureWindow(
                turn_id=self.current_turn_id,
                sequence=self.sequence,
                start_sample=start_s,
                end_sample=end_s,
                start_time=start_s / self.sample_rate,
                end_time=end_s / self.sample_rate,
                sample_rate=self.sample_rate,
                feature_hop=self.hop_size,
                rms_energy=rms,
                pitch=pitch,
                voicing=voicing,
                log_mel=log_mel
            )
            windows.append(fw)
            
            # Shift buffer by hop_size
            self.audio_buffer = self.audio_buffer[self.hop_size:]
            self.buffer_start_sample += self.hop_size
            self.sequence += 1
            
        return windows

    def flush(self) -> List[AudioFeatureWindow]:
        """Flushes remaining audio if it's smaller than a window, zero-padding to make one final window."""
        windows = []
        if 0 < len(self.audio_buffer) < self.window_size:
            pad_size = self.window_size - len(self.audio_buffer)
            padded_audio = np.pad(self.audio_buffer, (0, pad_size), mode='constant')
            
            # Create a mock full-sized chunk payload to pass into our internal append logic
            # We temporarily override the buffer
            self.audio_buffer = padded_audio
            
            # This will extract exactly one window and shift by hop_size
            # We construct a dummy chunk just to invoke the rest of the append logic
            # safely without breaking turn state
            dummy_chunk = AudioChunk(
                timestamp=self._get_dummy_timestamp(self.current_turn_id),
                audio_data=np.array([], dtype=np.float32), 
                num_channels=1, 
                sample_count=0, 
                is_final=True
            )
            windows = self.append(dummy_chunk)
            self._reset_buffer()
        return windows

    def _get_dummy_timestamp(self, turn_id):
        from shared.media_types import MediaTimestamp
        return MediaTimestamp(turn_id=turn_id, sample_position=self.buffer_start_sample, sample_rate=self.sample_rate)
