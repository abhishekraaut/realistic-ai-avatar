import React, { useRef, useEffect, useState } from 'react';
import { CallState } from './types';

interface AvatarVideoProps {
  src: string;
  callState: CallState;
}

export function AvatarVideo({ src, callState }: AvatarVideoProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (videoRef.current) {
      if (callState === 'active' || callState === 'connecting') {
        videoRef.current.play().catch(() => {
          // Graceful fallback if autoplay fails
        });
      } else {
        videoRef.current.pause();
        videoRef.current.currentTime = 0;
      }
    }
  }, [callState]);

  if (error) {
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-900 text-white z-0">
        <svg className="w-12 h-12 text-gray-500 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <p className="text-lg font-medium">Unable to load avatar</p>
        <button 
          onClick={() => setError(false)}
          className="mt-4 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-full text-sm font-semibold transition-colors"
        >
          Try again
        </button>
      </div>
    );
  }

  return (
    <video 
      ref={videoRef}
      className="absolute inset-0 w-full h-full object-cover z-0"
      autoPlay
      muted
      loop
      playsInline
      onError={() => setError(true)}
    >
      <source src={src} type="video/mp4" />
      <p>Your browser does not support HTML5 video.</p>
    </video>
  );
}
