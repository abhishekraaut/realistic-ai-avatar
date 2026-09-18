import { AvatarItem } from './types';

// The iframe SRCs still point to the external URL
// const IFRAME_BASE = 'https://ia-iframe-public.prd.synthesia.io';

export const INTERACTIVE_AVATARS: AvatarItem[] = [
  {
    id: 'ela',
    title: 'Emotional Rollercoaster',
    description: 'Ask me to act angry, sad, or thrilled',
    posterSrc: `/assets/emotional_rollercoaster.png`,
    previewVideoSrc: `/assets/emotional-rollercoaster.mp4`,
    iframeSrc: `/player`,
  },
  {
    id: 'tym',
    title: 'AI Advisor',
    description: 'Ask me how to use AI in your daily workflow',
    posterSrc: `/assets/how_I_AI.png`,
    previewVideoSrc: `/assets/how-i-ai.mp4`,
    iframeSrc: `/player`,
  },
  {
    id: 'jane',
    title: 'Practice giving feedback',
    description: 'Role-play a tough performance review with me',
    posterSrc: `/assets/practice-feedback.png`,
    previewVideoSrc: `/assets/practice-feedback.mp4`,
    iframeSrc: `/player`,
    previewStartTime: 0,
  },
  {
    id: 'alexa',
    title: 'Sales Concierge',
    description: 'Ask me anything about building videos in Synthesia',
    posterSrc: `/assets/Synthesia_assistant.png`,
    previewVideoSrc: `/assets/synthesia-assistant.mp4`,
    iframeSrc: `/player`,
  },
  {
    id: 'daisy',
    title: 'Play a Game',
    description: "Guess which animal I'm thinking of",
    posterSrc: `/assets/guess-the-animal.png`,
    previewVideoSrc: `/assets/guess-the-animal.mp4`,
    iframeSrc: `/player`,
  },
  {
    id: 'lyra',
    title: 'Space Alien',
    description: 'Discover the Solar System',
    posterSrc: `/assets/guess-where.png`,
    previewVideoSrc: `/assets/guess-where.mp4`,
    iframeSrc: `/player`,
    previewStartTime: 0,
  },
];
