import { useState, useEffect, useCallback } from 'react';

export function useCallTimer(isActive: boolean) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    if (isActive) {
      intervalId = setInterval(() => {
        setElapsedSeconds((prev) => prev + 1);
      }, 1000);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isActive]);

  // Reset when it becomes inactive
  if (!isActive && elapsedSeconds !== 0) {
    setElapsedSeconds(0);
  }

  const formattedTime = useCallback(() => {
    const minutes = Math.floor(elapsedSeconds / 60);
    const seconds = elapsedSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }, [elapsedSeconds]);

  return { formattedTime: formattedTime(), elapsedSeconds };
}
