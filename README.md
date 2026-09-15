# Modern Avatar

A Next.js front-end clone of the **Interactive Avatars** section on [synthesia.io](https://www.synthesia.io/features/avatars/interactive-avatars): a hero plus a looping carousel of avatar cards. Hovering a card (desktop) plays a short greeting clip. Clicking a card, or its **Talk** button, opens a modal with the live avatar embedded in an iframe.

The live conversation runs inside Synthesia's hosted iframe. This repo contains no AI, audio, or streaming backend of its own.

## Tech stack

| | |
| --- | --- |
| Framework | [Next.js](https://nextjs.org) 16 (App Router) |
| UI | React 19 |
| Styling | Tailwind CSS v4 (via `@tailwindcss/postcss`) |
| Language | TypeScript (strict) |
| Linting | ESLint 9 with `eslint-config-next` |
| Fonts | Geist / Geist Mono via `next/font/google` |

## Getting started

### Prerequisites

- Node.js **20.9 or newer** (required by Next.js 16)
- npm (the repo ships a `package-lock.json`)

### Install and run

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Start the development server with hot reload |
| `npm run build` | Create a production build |
| `npm run start` | Serve the production build (run `build` first) |
| `npm run lint` | Run ESLint |

There is no test suite yet.

## Project structure

```
src/
├── app/
│   ├── layout.tsx                 # Root layout, fonts, metadata
│   ├── page.tsx                   # Hero + interactive avatar showcase (server component)
│   └── globals.css                # Tailwind import, theme tokens, carousel size variables
├── components/
│   └── InteractiveAvatar/
│       ├── index.ts               # Public exports
│       ├── avatars.ts             # The avatar cards: titles, posters, preview clips, iframe URLs
│       ├── AvatarShowcase.tsx     # Client entry: carousel + modal open/close state
│       ├── AvatarCarousel.tsx     # Looping, centered track with arrows and drag/swipe
│       ├── AvatarSlide.tsx        # One card: poster, hover preview, sound and Talk buttons
│       ├── AvatarIframeModal.tsx  # Dialog hosting the live avatar iframe
│       ├── types.ts               # AvatarItem, CallState, InteractiveAvatarProps
│       │
│       │   # Earlier local-video call prototype (not used by the page)
│       ├── InteractiveAvatar.tsx  # Trigger button + call state
│       ├── AvatarModal.tsx        # Dialog shell with timer and controls
│       ├── AvatarVideo.tsx        # <video> playback and error fallback
│       ├── AvatarControls.tsx     # Mute, end session, mic selector
│       └── CallTimer.tsx          # Elapsed-time pill
└── hooks/
    └── useCallTimer.ts            # Ticks once per second while active, resets on stop
```

The `@/*` import alias maps to `src/*`.

## Using the showcase

```tsx
import { AvatarShowcase } from "@/components/InteractiveAvatar/AvatarShowcase";
import { INTERACTIVE_AVATARS } from "@/components/InteractiveAvatar/avatars";

export default function Page() {
  return <AvatarShowcase items={INTERACTIVE_AVATARS} />;
}
```

Import from the component files directly in server components. The `index.ts` barrel also re-exports the older prototype, which uses client hooks without a `'use client'` directive.

### `AvatarItem`

| Field | Type | Description |
| --- | --- | --- |
| `id` | `string` | Stable key |
| `title` | `string` | Card heading |
| `description` | `string` | Card subheading |
| `posterSrc` | `string` | Image shown until the preview clip is playing |
| `previewVideoSrc` | `string` | MP4 greeting clip for the hover preview |
| `iframeSrc` | `string` | Live avatar URL opened in the modal |
| `previewStartTime` | `number?` | Seconds into the clip where the preview starts. Defaults to `0.8` |

## Behavior

### Carousel

- Cards render three times so the track loops in both directions. When a transition ends outside the middle copy, the track jumps back into it without animating.
- Arrow buttons move one card. Dragging with a mouse or swiping moves by the nearest number of cards, and the click that ends a drag doesn't open the modal.
- Clicking a card centers it and opens the modal.
- Card size is set by `--ia-slide-w` and `--ia-slide-h` in `globals.css`: 286 × 344 px at 992 px and wider, 270 × 328 px below that.

### Previews

- **992 px and wider:** hovering a card scales it to 115%, dims the other cards, expands the **Talk** label, and plays the clip muted. Moving the pointer away rewinds it.
- **Sound button:** plays the clip with sound, or stops it. If the browser blocks audio, the clip falls back to muted.
- Only one clip plays at a time.

### Modal

- The modal closes on the ✕ button, a backdrop click, or <kbd>Esc</kbd>. It also closes when the iframe posts `{ ns: "ia-pg", type: "ended" | "disconnected" }` from a `*.synthesia.io` origin.
- The iframe is removed 300 ms after closing, once the fade finishes, which ends the call.
- Focus moves to the close button on open and returns to the previous element on close. Body scroll is locked while the modal is open.

## Known limitations

- **Third-party assets.** Posters, preview clips, and iframe URLs in `avatars.ts` point at Synthesia's public demo. They are Synthesia's content, and the iframe may refuse to run outside synthesia.io. Replace them with your own before shipping.
- **No focus trap.** The modal moves focus to the close button but doesn't trap <kbd>Tab</kbd>.
- **No wheel or trackpad scrolling** on the carousel. Only the arrows, drag, and swipe move it.

## A note for contributors (and AI agents)

This project uses Next.js 16, which has breaking changes compared with earlier versions. Before changing framework-level code, read the bundled docs in `node_modules/next/dist/docs/`. See [AGENTS.md](AGENTS.md).

## Deployment

Any host that supports Next.js will work. [Vercel](https://vercel.com/new) is the simplest option. See the [Next.js deployment docs](https://nextjs.org/docs/app/getting-started/deploying) for other options.
