import { AvatarShowcase } from '@/components/InteractiveAvatar/AvatarShowcase';
import { INTERACTIVE_AVATARS } from '@/components/InteractiveAvatar/avatars';

export default function Home() {
  return (
    <main className="min-h-screen overflow-x-hidden bg-white font-sans text-[#0d0f2c]">
      <section className="bg-[linear-gradient(#fff_4.53%,#e3ebff_45%)] pt-20 pb-16 md:pt-28 md:pb-24">
        {/* <div className="mx-auto grid max-w-[80rem] gap-6 px-4 sm:px-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,27rem)] lg:gap-16">
          <div>
            <p className="mb-4 text-xs leading-4 font-medium tracking-[0.08em] text-[#3e57da] uppercase">
              Early LiveKit SDK access
            </p>
            <h1 className="text-[2.75rem] leading-[1.05] font-medium tracking-[-0.05em] md:text-6xl lg:text-[4.5rem] lg:leading-[4.625rem]">
              Get started with
              <br />
              Interactive
              <br />
              Avatar experiences
            </h1>
          </div>

          <div className="flex flex-col items-start gap-6 lg:pt-12">
            <p className="text-base leading-6 text-[#333b52] md:text-lg md:leading-7">
              Build Interactive Avatars that can listen, talk, and respond directly inside your product,
              website, or internal tools.
            </p>
            <a
              href="#"
              className="inline-flex h-10 items-center gap-2 rounded-lg border border-[#e6eaf4] bg-white px-4 text-sm font-medium text-[#0d0f2c] shadow-[0_1px_2px_rgba(16,24,40,0.05)] transition-colors hover:bg-[#f5f7fb]"
            >
              Request access
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
                <path d="M1.333 8h13.334M9.212 2.545 14.667 8l-5.455 5.455" />
              </svg>
            </a>
          </div>
        </div> */}

        <div className="mt-6 md:mt-10">
          <AvatarShowcase items={INTERACTIVE_AVATARS} />
        </div>
      </section>
    </main>
  );
}
