# site

The project site for tiny-llm-serve, published at
<https://lxynov.github.io/tiny-llm-serve/>. Markdown in, static site out — no
framework, eight dependencies, ~1,200 lines of build and dev-server code.

```
npm install
npm run dev      # http://localhost:3000/tiny-llm-serve/ — drafts visible, live reload
npm run preview  # http://localhost:4000/tiny-llm-serve/ — exactly what deploys, drafts hidden
npm run build    # writes dist/
npm run new "A post title"
```

`dev` and `preview` use different ports and different output directories, so you
can leave both running and flip between tabs.

## What's here

| Path | What it does |
| --- | --- |
| `site.config.js` | Title, base path, nav, links, syntax theme. Start here. |
| `src/posts/*.md` | One file per post. The index at `/` lists them. |
| `src/pages/*.md` | Standalone pages (`about.md` → `/about/`). |
| `src/styles/main.css` | All the styling. Design tokens are at the top. |
| `src/scripts/site.js` | Theme toggle, TOC scroll-spy, copy buttons, giscus. |
| `src/public/` | Copied verbatim to the site root (images, …). |
| `build.js` | Orchestrates the build. |
| `lib/markdown.js` | Markdown → HTML: anchors, TOC, Shiki, KaTeX, footnotes. |
| `lib/templates.js` | Page HTML. Plain template literals. |
| `lib/assets.js` | Self-hosted fonts, KaTeX CSS, stylesheet bundling. |
| `lib/feed.js` | Atom feed, sitemap, robots.txt. |

`dist/` is generated — never edit it, never commit it.

## The base path

GitHub Pages serves a project site under `/<repo>/`, not at the domain root, so
`base` in `site.config.js` is prefixed onto every internal link and asset URL.
Two consequences worth knowing:

- Write internal links root-relative (`/posts/`, `/diagram.png`). The build adds
  the base; `lib/markdown.js` rewrites them on the markdown token rather than
  over the rendered HTML, so an `href="/…"` inside a code fence is left alone.
- `npm run dev` serves under the same base path and redirects `/` to it, so
  local URLs match the deployed ones. Changing `base` needs a dev restart —
  only the build runs in a child process.

Setting `base: ''` makes it a root-hosted site again.

## Writing a post

`npm run new "Paged Attention"` creates `src/posts/2026-08-22-paged-attention.md`
with frontmatter. **`title`** and **`date`** are required — the date can come
from the filename (`YYYY-MM-DD-slug.md`) instead. **`slug`** overrides the URL,
**`description`** falls back to the first ~200 characters, and **`draft: true`**
hides an item from `npm run build` while keeping it visible in `npm run dev`.

Code fences are highlighted at build time by Shiki in both themes at once, so
the light/dark switch is a pure CSS variable swap with no client-side
highlighter shipped to readers. Math is rendered at build time by KaTeX
(`$inline$`, `$$display$$`), and its stylesheet is only linked on pages that
contain math. Also supported: GFM tables, footnotes, smart quotes, and a lone
`![alt](img.png)` in a paragraph becoming a `<figure>` with the alt as caption.

Headings `##` and `###` get anchor links and populate the table of contents.

## Deploying

`.github/workflows/deploy-site.yml` builds and publishes on every push to `main`
that touches `site/`. One-time setup: repo **Settings → Pages → Source → GitHub
Actions**.

## Provenance

Forked from the static site generator in
[lxynov/lxynov.github.io](https://github.com/lxynov/lxynov.github.io) and
diverged — there is no shared dependency, and fixes do not propagate in either
direction. What changed:

- **A base path** (`config.base`), threaded through the templates, the feed, the
  sitemap, the markdown link rewriting, and the dev server.
- **The duplicate-URL check covers generated pages.** Upstream, a
  `src/pages/index.md` silently raced the generated index at `/` — both were
  written inside one `Promise.all`, and the check only compared posts against
  pages. It now fails the build instead.
- **Comments are off**, and the project's own repo and identity are in
  `site.config.js`.
