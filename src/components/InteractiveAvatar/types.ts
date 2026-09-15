export type CallState = "idle" | "connecting" | "active" | "ended";

export interface InteractiveAvatarProps {
  videoSrc?: string;
}

export interface AvatarItem {
  id: string;
  title: string;
  description: string;
  /** Still image shown on the card until the preview video is playing. */
  posterSrc: string;
  /** Short greeting clip played on hover (desktop) or via the sound button. */
  previewVideoSrc: string;
  /** Embeddable URL of the live interactive avatar, opened in the modal. */
  iframeSrc: string;
  /** Seconds into the preview clip where playback starts. Defaults to 0.8. */
  previewStartTime?: number;
}
