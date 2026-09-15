import React from 'react';
import { useCallTimer } from '@/hooks/useCallTimer';

interface CallTimerProps {
  isActive: boolean;
}

export function CallTimer({ isActive }: CallTimerProps) {
  const { formattedTime } = useCallTimer(isActive);

  if (!isActive) return null;

  return (
    <div 
      className="h-9 px-3 flex items-center bg-[#363036] text-white rounded-full text-sm font-semibold shadow-sm"
      aria-label={`Call duration: ${formattedTime}`}
    >
      {formattedTime}
    </div>
  );
}
