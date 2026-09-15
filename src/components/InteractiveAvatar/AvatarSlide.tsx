'use client';

import React, { useEffect, useRef, useState } from 'react';
import { AvatarItem } from './types';

const DEFAULT_PREVIEW_START = 0.8;
// Hover previews only run with the desktop layout, same as the reference site.
const HOVER_PREVIEW_QUERY = '(min-width: 992px)';

interface AvatarSlideProps {
  item: AvatarItem;
  /** Duplicate rendered only to make the carousel loop; hidden from assistive tech. */
  isClone: boolean;
  isPlaying: boolean;
  onPlaybackStart: () => void;
  onPlaybackStop: () => void;
  onSelect: () => void;
}

const controlButtonClass =
  'pointer-events-auto flex items-center rounded-full border border-[rgba(233,236,246,0.4)] bg-white/40 p-1 whitespace-nowrap shadow-[0_2.5px_8px_0_rgba(16,24,40,0.08)] backdrop-blur-[4.4px] transition-colors duration-600 group-hover:bg-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white';

const controlIconClass =
  'flex size-8 items-center justify-center rounded-full bg-white text-[#333b52] shadow-[0_2px_10px_0_rgba(16,24,40,0.08),0_0_0_1px_#e6eaf4]';

export function AvatarSlide({
  item,
  isClone,
  isPlaying,
  onPlaybackStart,
  onPlaybackStop,
  onSelect,
}: AvatarSlideProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isVideoVisible, setIsVideoVisible] = useState(false);
  const [isSoundOn, setIsSoundOn] = useState(false);
  const hasSound = isPlaying && isSoundOn;

  // Another slide took over playback (or it was stopped): rewind to the poster state.
  useEffect(() => {
    const video = videoRef.current;
    if (isPlaying || !video) return;
    video.muted = true;
    video.pause();
    video.currentTime = 0;
  }, [isPlaying]);

  const startPlayback = (withSound: boolean) => {
    const video = videoRef.current;
    if (!video) return;
    onPlaybackStart();
    setIsVideoVisible(false);
    setIsSoundOn(withSound);
    video.currentTime = item.previewStartTime ?? DEFAULT_PREVIEW_START;
    video.muted = !withSound;
    video.play().catch((error: DOMException) => {
      // Sound was blocked by the autoplay policy; fall back to a muted preview.
      if (error.name !== 'NotAllowedError') return;
      video.muted = true;
      setIsSoundOn(false);
      video.play().catch(() => {});
    });
  };

  const handleMouseEnter = () => {
    if (window.matchMedia(HOVER_PREVIEW_QUERY).matches) startPlayback(false);
  };

  const handleMouseLeave = () => {
    if (isPlaying && window.matchMedia(HOVER_PREVIEW_QUERY).matches) onPlaybackStop();
  };

  const handleSoundToggle = (event: React.MouseEvent) => {
    event.stopPropagation();
    if (hasSound) {
      onPlaybackStop();
    } else {
      startPlayback(true);
    }
  };

  const handleClick = () => {
    if (isPlaying) onPlaybackStop();
    onSelect();
  };

  return (
    <div
      className="ia-slide group relative h-(--ia-slide-h) w-(--ia-slide-w) shrink-0 cursor-pointer select-none min-[992px]:transition-opacity min-[992px]:duration-500"
      aria-hidden={isClone || undefined}
      onClick={handleClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div className="relative h-full overflow-hidden rounded-2xl min-[992px]:transition-transform min-[992px]:duration-400 min-[992px]:group-hover:scale-[1.15]">
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url("${item.posterSrc}")` }}
        >
          <video
            ref={videoRef}
            className={`relative z-2 size-full object-cover transition-opacity duration-400 ${
              isPlaying && isVideoVisible ? 'opacity-100' : 'opacity-0'
            }`}
            preload="none"
            playsInline
            muted
            aria-hidden="true"
            onPlaying={() => setIsVideoVisible(true)}
            onEnded={onPlaybackStop}
          >
            <source src={item.previewVideoSrc} type="video/mp4" />
          </video>
        </div>

        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-0 top-1/2 -bottom-1 z-1 bg-linear-to-b from-transparent to-[#0d0f2c]"
        />

        <div className="absolute inset-x-0 bottom-0 z-2 p-6 text-center text-white">
          <p className="text-base leading-6 font-bold tracking-[-0.004em]">{item.title}</p>
          <p className="mt-1 text-sm leading-[1.375rem] font-medium text-balance opacity-60">
            {item.description}
          </p>
        </div>

        <div className="pointer-events-none absolute inset-0 z-3 flex items-start justify-end gap-2 p-4 min-[768px]:max-[991px]:p-6">
          <button
            type="button"
            className={controlButtonClass}
            tabIndex={isClone ? -1 : undefined}
            aria-pressed={hasSound}
            aria-label={hasSound ? `Mute ${item.title} preview` : `Play ${item.title} preview with sound`}
            onClick={handleSoundToggle}
          >
            <span className={controlIconClass}>
              {hasSound ? <SpeakerHighIcon /> : <SpeakerSlashIcon />}
            </span>
          </button>

          {/* Clicks bubble to the slide, which opens the call. */}
          <button
            type="button"
            className={controlButtonClass}
            tabIndex={isClone ? -1 : undefined}
            aria-label={`Talk to ${item.title}`}
          >
            <span className={controlIconClass}>
              <PhoneIcon />
            </span>
            <span className="grid grid-cols-[0fr] text-sm leading-[1.375rem] font-medium text-[#475070] opacity-0 transition-[grid-template-columns,margin,opacity] duration-600 group-hover:mx-4 group-hover:grid-cols-[1fr] group-hover:opacity-100">
              <span className="overflow-hidden">Talk</span>
            </span>
          </button>
        </div>
      </div>
    </div>
  );
}

const iconProps = {
  className: 'size-4',
  viewBox: '0 0 16 16',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.2,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  'aria-hidden': true,
} as const;

function SpeakerSlashIcon() {
  return (
    <svg {...iconProps}>
      <path d="M2.5 2.5l11 11" />
      <path d="M10.5 6.2V2L6.8 4.9M5 6H2.5a.5.5 0 0 0-.5.5v3a.5.5 0 0 0 .5.5H5l5.5 4V11" />
    </svg>
  );
}

function SpeakerHighIcon() {
  return (
    <svg {...iconProps}>
      <path d="M5 10H2.5a.5.5 0 0 1-.5-.5v-3a.5.5 0 0 1 .5-.5H5l4.5-4v12L5 10Z" />
      <path d="M12 6.2a2.5 2.5 0 0 1 0 3.6M13.8 4.5a5 5 0 0 1 0 7" />
    </svg>
  );
}

function PhoneIcon() {
  return (
    <svg {...iconProps}>
      <path d="M5.8 2.5 7 5.3a.6.6 0 0 1-.15.68l-1.1 1a6.6 6.6 0 0 0 3.27 3.27l1-1.1a.6.6 0 0 1 .68-.15l2.8 1.2a.6.6 0 0 1 .35.64l-.3 1.66a1.2 1.2 0 0 1-1.18.99A9.4 9.4 0 0 1 2.5 4.03a1.2 1.2 0 0 1 .99-1.18l1.66-.3a.6.6 0 0 1 .64.35Z" />
    </svg>
  );
}
