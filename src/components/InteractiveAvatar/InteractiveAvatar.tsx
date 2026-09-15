'use client';

import React, { useState, useCallback } from 'react';
import { InteractiveAvatarProps, CallState } from './types';
import { AvatarModal } from './AvatarModal';

export default function InteractiveAvatar({ 
  videoSrc = "/videos/avatar.mp4" 
}: InteractiveAvatarProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [callState, setCallState] = useState<CallState>('idle');
  const [isMuted, setIsMuted] = useState(false);

  const handleOpen = useCallback(() => {
    setIsOpen(true);
    setCallState('active'); // Directly to active for this demo, could simulate 'connecting' first
    setIsMuted(false);
  }, []);

  const handleClose = useCallback(() => {
    setIsOpen(false);
    setCallState('ended');
    
    // Reset back to idle after animation
    setTimeout(() => {
      setCallState('idle');
    }, 300);
  }, []);

  const handleToggleMute = useCallback(() => {
    setIsMuted(prev => !prev);
  }, []);

  return (
    <>
      <button 
        onClick={handleOpen}
        className="px-6 py-3 bg-white text-black font-semibold rounded-full hover:bg-gray-100 transition-colors shadow-lg flex items-center gap-2"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
        Start Interactive Avatar Call
      </button>

      <AvatarModal 
        isOpen={isOpen}
        onClose={handleClose}
        videoSrc={videoSrc}
        callState={callState}
        isMuted={isMuted}
        onToggleMute={handleToggleMute}
      />
    </>
  );
}
