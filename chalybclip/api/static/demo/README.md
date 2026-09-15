# Landing hero demo thumbnails

The landing hero animation ("1 stream → 5 platforms") renders real images
from this folder: one big source VOD card plus five vertical clip cards.
Drop your files here:

| File         | Card / badge               |
|--------------|----------------------------|
| `source.jpg` | Big source VOD card (16:9) |
| `clip1.jpg`  | TikTok                     |
| `clip2.jpg`  | Reels                      |
| `clip3.jpg`  | Shorts                     |
| `clip4.jpg`  | Kick                       |
| `clip5.jpg`  | Twitch                     |

## Sizing

- **Clip cards (`clip1..5.jpg`):** 9:16 (vertical). The card crops to this
  with `object-fit: cover`, focal point ~28% from the top (faces sit well).
  Recommended 360×640 px, optimized JPG/WebP, < ~80 KB each.
- **Source card (`source.jpg`):** 16:9 (horizontal), e.g. 1024×576. It's
  treated as a finished card — its own LIVE tag, play button, filename and
  duration can be baked in; when it loads the template hides its duplicate
  CSS chrome and keeps only the animated progress sweep.
- `.jpg` is what the template references; rename or change the `src` in
  `landing.html` if you use `.webp`.

## Missing files are safe

If a file isn't present, the `<img>` removes itself (`onerror`) and the
original gradient placeholder shows. The page never breaks on a missing
asset — so you can add these one at a time.

## ⚠️ Rights / licensing — read before adding faces

These appear on the public, commercial landing page. Using a real,
identifiable person's image here implies they endorse / use ChalybClip and
carries real legal exposure:

- **Right of publicity** — commercial use of someone's likeness needs
  consent.
- **False endorsement** (Lanham Act §43(a)) — implying a sponsorship
  that doesn't exist.
- **Copyright** — the photo itself is owned by whoever shot it.

Only drop in images you actually have the rights to use commercially:
your own creators/clients (with written permission), licensed stock of
content creators, or AI-generated faces (no real person). Do **not** use
scraped photos of famous streamers.
