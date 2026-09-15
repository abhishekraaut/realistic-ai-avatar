'use client';

import React, { useEffect, useRef } from 'react';

// The embedded avatar posts { ns: 'ia-pg', type } messages; these types mean the call is over.
const MESSAGE_NAMESPACE = 'ia-pg';
const ENDED_MESSAGE_TYPES = new Set(['ended', 'disconnected']);
const TRUSTED_HOST_SUFFIX = '.synthesia.io';

interface AvatarIframeModalProps {
  isOpen: boolean;
  /** Kept set while the modal fades out, then cleared so the call is torn down. */
  src: string | null;
  title?: string;
  onClose: () => void;
}

export function AvatarIframeModal({ isOpen, src, title, onClose }: AvatarIframeModalProps) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!isOpen) return;

    const previousFocus = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    closeButtonRef.current?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };

    const handleMessage = (event: MessageEvent) => {
      let host: string;
      try {
        host = new URL(event.origin).hostname;
      } catch {
        return;
      }
      if (!host.endsWith(TRUSTED_HOST_SUFFIX)) return;
      const data = event.data as { ns?: string; type?: string } | null;
      if (data?.ns === MESSAGE_NAMESPACE && data.type && ENDED_MESSAGE_TYPES.has(data.type)) onClose();
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('message', handleMessage);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('message', handleMessage);
      previousFocus?.focus();
    };
  }, [isOpen, onClose]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={title ? `Talk to ${title}` : 'Interactive avatar'}
      inert={!isOpen}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
      // Visibility only transitions on close, so the dialog is focusable as soon as it opens.
      className={`fixed inset-0 z-50 flex items-center justify-center bg-[rgba(51,59,82,0.6)] px-10 duration-300 ${
        isOpen
          ? 'visible cursor-pointer opacity-100 transition-opacity'
          : 'pointer-events-none invisible opacity-0 transition-[opacity,visibility]'
      }`}
    >
      <div
        className={`relative aspect-square w-full max-w-[min(32rem,calc(100dvh-2rem))] cursor-auto overflow-hidden rounded-2xl bg-black transition-transform duration-300 ${
          isOpen ? 'scale-100' : 'scale-96'
        }`}
      >
        {src && (
          <iframe
            src={src}
            title={title ? `${title} interactive avatar` : 'Interactive avatar'}
            allow="microphone; autoplay"
            className="block size-full border-0"
          />
        )}
        <button
          ref={closeButtonRef}
          type="button"
          onClick={onClose}
          aria-label="Close avatar"
          className="absolute top-0 right-0 m-3 flex size-8 items-center justify-center rounded-full border border-[#e6eaf4] bg-white text-[#333b52] transition-colors duration-200 hover:bg-[#f5f7fb]"
        >
          <svg
            className="size-4"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth={1.2}
            strokeLinecap="round"
            aria-hidden="true"
          >
            <path d="M3.285 12.715 12.713 3.287M3.285 3.285l9.428 9.428" />
          </svg>
        </button>
      </div>
    </div>
  );
}
