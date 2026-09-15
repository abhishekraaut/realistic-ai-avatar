import React from 'react';

interface AvatarControlsProps {
  isMuted: boolean;
  onToggleMute: () => void;
  onEndCall: () => void;
}

export function AvatarControls({ isMuted, onToggleMute, onEndCall }: AvatarControlsProps) {
  return (
    <div className="flex flex-col items-center gap-4 w-full">
      {/* Action Row */}
      <div className="flex items-center justify-center gap-3">
        <button 
          onClick={onToggleMute}
          aria-label={isMuted ? "Unmute microphone" : "Mute microphone"}
          aria-pressed={isMuted}
          className={`w-12 h-12 flex items-center justify-center rounded-full transition-colors ${
            isMuted ? 'bg-red-100 text-red-600 hover:bg-red-200' : 'bg-white text-black hover:bg-gray-100'
          }`}
        >
          {isMuted ? (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" clipRule="evenodd" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2" />
            </svg>
          ) : (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
          )}
        </button>
        
        <button 
          onClick={onEndCall}
          aria-label="End call"
          className="h-12 px-6 bg-red-500 hover:bg-red-600 transition-colors text-white rounded-full text-sm font-semibold shadow-sm"
        >
          End session
        </button>
      </div>

      {/* Mic Selector (Decorative for now as requested by typical visual fidelity) */}
      <button 
        className="flex items-center gap-1 text-white text-sm border-b border-white/50 pb-0.5 opacity-90 hover:opacity-100 transition-opacity focus:outline-none focus:ring-2 focus:ring-white/50 focus:ring-offset-4 focus:ring-offset-black rounded-sm"
        aria-haspopup="listbox"
      >
        Default mic
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
          <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
        </svg>
      </button>
    </div>
  );
}
