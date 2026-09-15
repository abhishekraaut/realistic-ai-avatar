'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AvatarCarousel } from './AvatarCarousel';
import { AvatarIframeModal } from './AvatarIframeModal';
import { AvatarItem } from './types';

// Matches the modal fade, so the iframe is removed only once it is hidden.
const CLOSE_ANIMATION_MS = 300;

interface AvatarShowcaseProps {
  items: AvatarItem[];
}

export function AvatarShowcase({ items }: AvatarShowcaseProps) {
  const [activeItem, setActiveItem] = useState<AvatarItem | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const closeTimerRef = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(closeTimerRef.current), []);

  const handleSelect = useCallback((item: AvatarItem) => {
    window.clearTimeout(closeTimerRef.current);
    setActiveItem(item);
    setIsOpen(true);
  }, []);

  const handleClose = useCallback(() => {
    setIsOpen(false);
    window.clearTimeout(closeTimerRef.current);
    closeTimerRef.current = window.setTimeout(() => setActiveItem(null), CLOSE_ANIMATION_MS);
  }, []);

  return (
    <>
      <AvatarCarousel items={items} onSelect={handleSelect} />
      <AvatarIframeModal
        isOpen={isOpen}
        src={activeItem?.iframeSrc ?? null}
        title={activeItem?.title}
        onClose={handleClose}
      />
    </>
  );
}
