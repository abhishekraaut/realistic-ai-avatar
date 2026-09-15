import { AvatarItem } from './types';

// Media and iframe URLs point at Synthesia's public demo assets. Swap them for your own before shipping.
const POSTER_BASE = 'https://cdn.prod.website-files.com/65e89895c5a4b8d764c0d710';
const VIDEO_BASE = 'https://webcdn.synthesia.io/interactive-avatars';
const IFRAME_BASE = 'https://ia-iframe-public.prd.synthesia.io';

export const INTERACTIVE_AVATARS: AvatarItem[] = [
  {
    id: 'ela',
    title: 'Emotional Rollercoaster',
    description: 'Ask me to act angry, sad, or thrilled',
    posterSrc: `${POSTER_BASE}/6a1ffa0571abbaae69094995_emotional_rollercoaster.png`,
    previewVideoSrc: `${VIDEO_BASE}/emotional-rollercoaster-personalized-greeting.mp4`,
    iframeSrc: `${IFRAME_BASE}/5`,
  },
  {
    id: 'tym',
    title: 'AI Advisor',
    description: 'Ask me how to use AI in your daily workflow',
    posterSrc: `${POSTER_BASE}/6a1ffa06a49ebaf1178b116f_ff04fdfc2b801681130c0a202ae4587c_how_I_AI.png`,
    previewVideoSrc: `${VIDEO_BASE}/how-i-ai-personalized-greeting.mp4`,
    iframeSrc: `${IFRAME_BASE}/3`,
  },
  {
    id: 'jane',
    title: 'Practice giving feedback',
    description: 'Role-play a tough performance review with me',
    posterSrc: `${POSTER_BASE}/6a3be44e956da561ff740966_interactive-avatar-practice-feedback.png`,
    previewVideoSrc: `${VIDEO_BASE}/practice-feedback-personalized-greeting-v2.mp4`,
    iframeSrc: `${IFRAME_BASE}/2`,
    previewStartTime: 0,
  },
  {
    id: 'alexa',
    title: 'Sales Concierge',
    description: 'Ask me anything about building videos in Synthesia',
    posterSrc: `${POSTER_BASE}/6a1ffa06d69a92014fceca23_924cbd81836bb241e40baba88768c9c2_Synthesia_assistant.png`,
    previewVideoSrc: `${VIDEO_BASE}/synthesia-assistant-personalized-greeting.mp4`,
    iframeSrc: `${IFRAME_BASE}/1`,
  },
  {
    id: 'daisy',
    title: 'Play a Game',
    description: "Guess which animal I'm thinking of",
    posterSrc: `${POSTER_BASE}/6a35615e3c27d7c39325c768_interactive-avatar-guess-the-animal.png`,
    previewVideoSrc: `${VIDEO_BASE}/guess-the-animal-personalized-greeting.mp4`,
    iframeSrc: `${IFRAME_BASE}/7`,
  },
  {
    id: 'lyra',
    title: 'Space Alien',
    description: 'Discover the Solar System',
    posterSrc: `${POSTER_BASE}/6a35615e3302889532076108_interactive-avatar-guess-where.png`,
    previewVideoSrc: `${VIDEO_BASE}/guess-where-personalized-greeting.mp4`,
    iframeSrc: `${IFRAME_BASE}/8`,
    previewStartTime: 0,
  },
];
