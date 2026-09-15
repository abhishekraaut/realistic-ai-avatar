import React, { useEffect } from 'react';
import { CallState } from './types';
import { AvatarVideo } from './AvatarVideo';
import { AvatarControls } from './AvatarControls';
import { CallTimer } from './CallTimer';

interface AvatarModalProps {
  isOpen: boolean;
  onClose: () => void;
  videoSrc: string;
  callState: CallState;
  isMuted: boolean;
  onToggleMute: () => void;
}

export function AvatarModal({
  isOpen,
  onClose,
  videoSrc,
  callState,
  isMuted,
  onToggleMute,
}: AvatarModalProps) {
  const [isRendered, setIsRendered] = React.useState(isOpen);
  const [isAnimatingOut, setIsAnimatingOut] = React.useState(false);

  // Sync state during render
  if (isOpen && (!isRendered || isAnimatingOut)) {
    setIsRendered(true);
    setIsAnimatingOut(false);
  }

  useEffect(() => {
    if (!isOpen && isRendered && !isAnimatingOut) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setIsAnimatingOut(true);
      const timer = setTimeout(() => {
        setIsRendered(false);
        setIsAnimatingOut(false);
      }, 300); // match animation duration
      return () => clearTimeout(timer);
    }
  }, [isOpen, isRendered, isAnimatingOut]);

  const modalRef = React.useRef<HTMLDivElement>(null);
  const previousFocusRef = React.useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (isOpen) {
      previousFocusRef.current = document.activeElement as HTMLElement;
      // Small timeout to ensure modal is rendered
      setTimeout(() => {
        if (modalRef.current) {
          // Focus the first focusable element, or the modal itself
          const focusable = modalRef.current.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])') as HTMLElement;
          if (focusable) {
            focusable.focus();
          } else {
            modalRef.current.focus();
          }
        }
      }, 10);
    } else if (!isOpen && previousFocusRef.current) {
      previousFocusRef.current.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && !isAnimatingOut) {
        onClose();
      }
      
      // Simple focus trap
      if (e.key === 'Tab' && isOpen && modalRef.current) {
        const focusableElements = modalRef.current.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        
        if (focusableElements.length > 0) {
          const firstElement = focusableElements[0] as HTMLElement;
          const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

          if (e.shiftKey) {
            if (document.activeElement === firstElement) {
              lastElement.focus();
              e.preventDefault();
            }
          } else {
            if (document.activeElement === lastElement) {
              firstElement.focus();
              e.preventDefault();
            }
          }
        }
      }
    };
    
    if (isOpen) {
      document.body.style.overflow = 'hidden';
      window.addEventListener('keydown', handleKeyDown);
    } else {
      document.body.style.overflow = '';
    }
    
    return () => {
      document.body.style.overflow = '';
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose, isAnimatingOut]);

  if (!isRendered) return null;

  return (
    <div 
      className={`fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 md:p-12 ${isAnimatingOut ? 'animate-out fade-out duration-300' : 'animate-in fade-in duration-300'}`}
      role="dialog"
      aria-modal="true"
      aria-label="Interactive Avatar Call"
    >
      {/* Background Overlay */}
      <div 
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        aria-hidden="true"
        onClick={onClose}
      />

      {/* Main Container */}
      <div 
        ref={modalRef}
        tabIndex={-1}
        className={`relative w-full max-w-5xl aspect-[4/5] sm:aspect-video rounded-[22px] overflow-hidden bg-black border border-white/10 shadow-2xl shadow-black/50 ${isAnimatingOut ? 'animate-out zoom-out-95 duration-300' : 'animate-in zoom-in-95 duration-300'}`}
      >
        <AvatarVideo src={videoSrc} callState={callState} />

        {/* UI Overlay */}
        <div className="absolute inset-0 flex flex-col z-10 pointer-events-none">
          {/* Top Section */}
          <div className="flex justify-between items-start p-4 md:p-6 pointer-events-auto">
            {/* Left side could have a title or status, empty for now */}
            <div></div>
            
            <div className="flex items-center gap-3">
              <CallTimer isActive={callState === 'active'} />
              
              <button 
                className="h-9 px-4 bg-white/10 hover:bg-white/20 transition-colors text-white rounded-full text-xs font-semibold shadow-sm focus:outline-none focus:ring-2 focus:ring-white/50 focus:ring-offset-2 focus:ring-offset-black"
                aria-label="Hide transcript"
              >
                Hide transcript
              </button>
              
              {/* Optional Close Button (X) on top right, if the reference has one, otherwise Escape/End Call works */}
              <button 
                onClick={onClose}
                className="w-9 h-9 flex items-center justify-center bg-white/10 hover:bg-white/20 transition-colors text-white rounded-full shadow-sm focus:outline-none focus:ring-2 focus:ring-white/50 focus:ring-offset-2 focus:ring-offset-black"
                aria-label="Close call window"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          {/* Bottom Section */}
          <div className="flex flex-col flex-1 items-center justify-end pb-6 sm:pb-8 pointer-events-auto bg-gradient-to-t from-black/60 to-transparent">
            <AvatarControls 
              isMuted={isMuted} 
              onToggleMute={onToggleMute} 
              onEndCall={onClose} 
            />
          </div>
        </div>
      </div>
    </div>
  );
}
