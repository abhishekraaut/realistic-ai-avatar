'use client';

import React, { useEffect, useRef, useState } from 'react';
import { AvatarItem } from './types';
import { AvatarSlide } from './AvatarSlide';

// Items render three times so the track can wrap seamlessly in both directions.
const COPIES = 3;
const DRAG_THRESHOLD_PX = 6;

interface AvatarCarouselProps {
  items: AvatarItem[];
  onSelect: (item: AvatarItem) => void;
}

export function AvatarCarousel({ items, onSelect }: AvatarCarouselProps) {
  const count = items.length;
  const total = count * COPIES;
  const trackRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef({ active: false, moved: false, startX: 0 });
  // Start centered on the middle card of the middle copy.
  const [index, setIndex] = useState(count + Math.floor(count / 2));
  const [isAnimated, setIsAnimated] = useState(true);
  const [dragX, setDragX] = useState(0);
  const [playingIndex, setPlayingIndex] = useState<number | null>(null);

  // After a silent jump back into the middle copy, re-enable the transition once it has painted.
  useEffect(() => {
    if (isAnimated) return;
    let frame = requestAnimationFrame(() => {
      frame = requestAnimationFrame(() => setIsAnimated(true));
    });
    return () => cancelAnimationFrame(frame);
  }, [isAnimated]);

  const scrollTo = (next: number) => setIndex(Math.min(total - 1, Math.max(0, next)));

  const handleTransitionEnd = (event: React.TransitionEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget || event.propertyName !== 'transform') return;
    if (index >= count && index < count * 2) return;
    setIsAnimated(false);
    setIndex((index % count) + count);
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    dragRef.current = { active: true, moved: false, startX: event.clientX };
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag.active) return;
    const delta = event.clientX - drag.startX;
    if (!drag.moved) {
      if (Math.abs(delta) < DRAG_THRESHOLD_PX) return;
      drag.moved = true;
      event.currentTarget.setPointerCapture(event.pointerId);
    }
    setDragX(delta);
  };

  const handlePointerUp = () => {
    const drag = dragRef.current;
    const track = trackRef.current;
    if (!drag.active) return;
    drag.active = false;
    if (!drag.moved || !track) return;

    const slide = track.firstElementChild as HTMLElement | null;
    const step = (slide?.offsetWidth ?? 0) + parseFloat(getComputedStyle(track).columnGap || '0');
    const shift = step ? Math.round(-dragX / step) : 0;
    setDragX(0);
    scrollTo(index + Math.max(1 - count, Math.min(count - 1, shift)));
  };

  // A drag ends with a click on whatever is under the pointer; swallow it so no call opens.
  const handleClickCapture = (event: React.MouseEvent) => {
    if (!dragRef.current.moved) return;
    dragRef.current.moved = false;
    event.stopPropagation();
    event.preventDefault();
  };

  const isDragging = dragX !== 0;
  const trackStyle = {
    '--ia-index': index,
    transform: `translate3d(calc(-1 * (var(--ia-index) * (var(--ia-slide-w) + var(--ia-gap)) + var(--ia-slide-w) / 2) + ${dragX}px), 0, 0)`,
  } as React.CSSProperties;

  return (
    <div
      className="ia-carousel relative"
      role="region"
      aria-roledescription="carousel"
      aria-label="Interactive avatars"
    >
      <div
        className="touch-pan-y overflow-hidden py-12"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onClickCapture={handleClickCapture}
      >
        <div
          ref={trackRef}
          className={`ia-track relative left-1/2 flex w-max gap-(--ia-gap) ${
            isAnimated && !isDragging ? 'transition-transform duration-500 ease-[cubic-bezier(0.25,1,0.5,1)]' : ''
          }`}
          style={trackStyle}
          onTransitionEnd={handleTransitionEnd}
        >
          {Array.from({ length: total }, (_, i) => {
            const item = items[i % count];
            return (
              <AvatarSlide
                key={`${item.id}-${i}`}
                item={item}
                isClone={i < count || i >= count * 2}
                isPlaying={playingIndex === i}
                onPlaybackStart={() => setPlayingIndex(i)}
                onPlaybackStop={() => setPlayingIndex((current) => (current === i ? null : current))}
                onSelect={() => {
                  scrollTo(i);
                  onSelect(item);
                }}
              />
            );
          })}
        </div>
      </div>

      <div className="pointer-events-none absolute inset-x-0 top-1/2 mx-auto flex w-[20rem] max-w-[calc(100%-5rem)] -translate-y-1/2 justify-between min-[992px]:w-[61rem]">
        <ArrowButton direction="prev" onClick={() => scrollTo(index - 1)} />
        <ArrowButton direction="next" onClick={() => scrollTo(index + 1)} />
      </div>
    </div>
  );
}

interface ArrowButtonProps {
  direction: 'prev' | 'next';
  onClick: () => void;
}

function ArrowButton({ direction, onClick }: ArrowButtonProps) {
  const isPrev = direction === 'prev';
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={isPrev ? 'Previous avatar' : 'Next avatar'}
      className="pointer-events-auto flex size-[2.375rem] items-center justify-center rounded-full border border-[#e6eaf4] bg-white text-[#333b52] transition-colors duration-300 hover:bg-[#f5f7fb] md:size-12"
    >
      <svg
        className="size-4"
        viewBox="0 0 16 16"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.2}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {isPrev ? (
          <path d="M14.667 8H1.333M6.788 2.545 1.333 8l5.455 5.455" />
        ) : (
          <path d="M1.333 8h13.334M9.212 2.545 14.667 8l-5.455 5.455" />
        )}
      </svg>
    </button>
  );
}
