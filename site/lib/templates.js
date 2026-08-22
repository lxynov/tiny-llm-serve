// HTML templates. Plain tagged-template functions -- no template language to
// learn, and the whole page structure is readable top to bottom.

export const esc = (s) =>
  String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

/**
 * Root-relative path -> href. GitHub Pages serves a project site under
 * /<repo>/, so every internal link carries `config.base`. Anything that isn't
 * root-relative (an external URL, a bare `#anchor`) is passed through.
 */
export const withBase = (config, p) =>
  p.startsWith('/') ? `${config.base}${p}` : p

/** Root-relative path -> absolute URL, for canonicals, the feed and sitemap. */
export const absolute = (config, p) => `${config.url}${withBase(config, p)}`

const MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
]

/** '2026-08-04' -> a UTC Date, so formatting never drifts by timezone. */
export function parseDate(value) {
  if (value instanceof Date) return value
  const [y, m, d] = String(value).slice(0, 10).split('-').map(Number)
  return new Date(Date.UTC(y, (m || 1) - 1, d || 1))
}

export const isoDate = (d) => parseDate(d).toISOString().slice(0, 10)
export const longDate = (d) => {
  const dt = parseDate(d)
  return `${MONTHS[dt.getUTCMonth()]} ${dt.getUTCDate()}, ${dt.getUTCFullYear()}`
}

// ---------------------------------------------------------------------------
// Base layout
// ---------------------------------------------------------------------------

export function layout({
  config,
  title,
  description,
  canonical,
  content,
  hasMath = false,
  bodyClass = '',
  dev = false,
}) {
  const fullTitle = title === config.title ? title : `${title} · ${config.title}`
  const desc = description || config.description

  return `<!doctype html>
<html lang="${esc(config.lang)}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(fullTitle)}</title>
<meta name="description" content="${esc(desc)}">
<meta name="author" content="${esc(config.author.name)}">
<link rel="canonical" href="${esc(canonical)}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="${esc(config.title)}">
<meta property="og:title" content="${esc(title)}">
<meta property="og:description" content="${esc(desc)}">
<meta property="og:url" content="${esc(canonical)}">
<meta name="twitter:card" content="summary">
<link rel="alternate" type="application/atom+xml" title="${esc(config.title)}" href="${withBase(config, '/feed.xml')}">
<link rel="icon" href="${withBase(config, '/favicon.svg')}" type="image/svg+xml">
<script>
// Applied before first paint so there is no flash of the wrong theme.
try{var t=localStorage.getItem('theme');if(t==='light'||t==='dark')document.documentElement.dataset.theme=t}catch(e){}
</script>
<link rel="stylesheet" href="${withBase(config, '/assets/main.css')}">
${hasMath ? `<link rel="stylesheet" href="${withBase(config, '/assets/katex.min.css')}">` : ''}
</head>
<body class="${esc(bodyClass)}">
<a class="skip-link" href="#main">Skip to content</a>

<header class="site-header">
  <a class="site-title" href="${withBase(config, '/')}">${esc(config.title)}</a>
  <nav class="site-nav">
    ${config.nav.map((l) => `<a href="${esc(withBase(config, l.href))}">${esc(l.label)}</a>`).join('')}
    ${themeToggle()}
  </nav>
</header>

<main id="main">
${content}
</main>

<footer class="site-footer">
  <p>© ${new Date().getUTCFullYear()} ${esc(config.author.name)}</p>
  <p class="footer-links">
    ${config.links.map((l) => `<a href="${esc(withBase(config, l.href))}">${esc(l.label)}</a>`).join('')}
  </p>
</footer>

<script src="${withBase(config, '/assets/site.js')}" defer></script>
${dev ? `<script src="${withBase(config, '/assets/livereload.js')}" defer></script>` : ''}
</body>
</html>
`
}

// Drafts only reach a rendered page under `npm run dev` (or an explicit
// `--drafts` build), so this marker never appears on the published site.
const draftBadge = (item) =>
  item.draft ? '<span class="draft-badge">draft</span>' : ''

function themeToggle() {
  return `<button class="theme-toggle" type="button" aria-label="Switch between light and dark theme" title="Switch theme">
  <svg class="icon-sun" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true">
    <circle cx="12" cy="12" r="4.2"/><path d="M12 2.4v2.2M12 19.4v2.2M4.2 4.2l1.6 1.6M18.2 18.2l1.6 1.6M2.4 12h2.2M19.4 12h2.2M4.2 19.8l1.6-1.6M18.2 5.8l1.6-1.6"/>
  </svg>
  <svg class="icon-moon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M20.5 14.6A8.6 8.6 0 1 1 9.4 3.5a6.9 6.9 0 0 0 11.1 11.1Z"/>
  </svg>
</button>`
}

// ---------------------------------------------------------------------------
// Home page -- date column + title, nothing else
// ---------------------------------------------------------------------------

export function renderIndex({ config, posts, dev }) {
  const rows = posts
    .map(
      (p) => `<li class="post-row">
      <time datetime="${isoDate(p.date)}">${isoDate(p.date)}</time>
      <a href="${esc(withBase(config, p.url))}">${p.titleHtml}</a>${draftBadge(p)}
    </li>`,
    )
    .join('\n')

  const content = `<div class="wrap">
  ${config.tagline ? `<p class="tagline">${esc(config.tagline)}</p>` : ''}
  ${
    posts.length
      ? `<ul class="post-list">\n${rows}\n</ul>`
      : `<p class="empty">No posts yet.</p>`
  }
</div>`

  return layout({
    config,
    title: config.title,
    description: config.description,
    canonical: absolute(config, '/'),
    content,
    bodyClass: 'page-home',
    dev,
  })
}

// ---------------------------------------------------------------------------
// Post page
// ---------------------------------------------------------------------------

export function renderPost({ config, post, prev, next, dev }) {
  const content = `<div class="post-wrap">
  ${renderToc(post.toc)}
  <article class="post">
    <header class="post-header">
      <h1>${post.titleHtml}${draftBadge(post)}</h1>
      <p class="post-meta">
        <time datetime="${isoDate(post.date)}">${longDate(post.date)}</time>
        <span class="dot">·</span>
        <span>${post.readingTime} min read</span>
        ${post.tags.length ? `<span class="dot">·</span><span class="post-tags">${post.tags.map((t) => esc(t)).join(', ')}</span>` : ''}
      </p>
    </header>

    <div class="prose">
${post.html}
    </div>

    ${renderPostNav(config, prev, next)}
    ${renderComments(config)}
  </article>
</div>`

  return layout({
    config,
    title: post.title,
    description: post.description,
    canonical: absolute(config, post.url),
    content,
    hasMath: post.hasMath,
    bodyClass: 'page-post',
    dev,
  })
}

function renderToc(toc) {
  if (!toc.length) return '<div class="toc-spacer" aria-hidden="true"></div>'

  const items = toc
    .map(
      (h) => `<li>
      <a href="#${esc(h.slug)}">${esc(h.text)}</a>
      ${
        h.children.length
          ? `<ul>${h.children
              .map((c) => `<li><a href="#${esc(c.slug)}">${esc(c.text)}</a></li>`)
              .join('')}</ul>`
          : ''
      }
    </li>`,
    )
    .join('\n')

  return `<details class="toc" open>
  <summary>Contents</summary>
  <nav aria-label="Table of contents">
    <ul>
${items}
    </ul>
  </nav>
</details>`
}

function renderPostNav(config, prev, next) {
  if (!prev && !next) return ''
  return `<nav class="post-nav">
    ${next ? `<a class="nav-next" href="${esc(withBase(config, next.url))}"><span>Next</span>${next.titleHtml}</a>` : '<span></span>'}
    ${prev ? `<a class="nav-prev" href="${esc(withBase(config, prev.url))}"><span>Previous</span>${prev.titleHtml}</a>` : '<span></span>'}
  </nav>`
}

function renderComments(config) {
  const c = config.comments
  if (!c.enabled || !c.repoId || !c.categoryId) return ''

  const settings = {
    repo: c.repo,
    repoId: c.repoId,
    category: c.category,
    categoryId: c.categoryId,
    mapping: c.mapping,
    reactionsEnabled: c.reactionsEnabled ? '1' : '0',
    themeLight: c.themeLight,
    themeDark: c.themeDark,
    lang: config.lang,
  }

  // The script tag is injected by site.js once this scrolls into view, so the
  // correct theme is known before giscus boots.
  return `<section class="comments" id="comments" data-giscus="${esc(JSON.stringify(settings))}">
    <h2>Comments</h2>
    <noscript><p class="comments-note">Comments require JavaScript. You can also reply in the <a href="https://github.com/${esc(c.repo)}/discussions">GitHub Discussions</a>.</p></noscript>
  </section>`
}

// ---------------------------------------------------------------------------
// Standalone pages (About, etc.) and 404
// ---------------------------------------------------------------------------

export function renderPage({ config, page, dev }) {
  const content = `<div class="post-wrap">
  ${renderToc(page.toc)}
  <article class="post">
    <header class="post-header"><h1>${page.titleHtml}${draftBadge(page)}</h1></header>
    <div class="prose">
${page.html}
    </div>
  </article>
</div>`

  return layout({
    config,
    title: page.title,
    description: page.description,
    canonical: absolute(config, page.url),
    content,
    hasMath: page.hasMath,
    bodyClass: 'page-static',
    dev,
  })
}

export function render404({ config, dev }) {
  return layout({
    config,
    title: 'Not found',
    description: 'Page not found',
    canonical: absolute(config, '/404.html'),
    bodyClass: 'page-404',
    dev,
    content: `<div class="wrap notfound">
  <h1>404</h1>
  <p>That page doesn't exist.</p>
  <p><a href="${withBase(config, '/')}">Back to all posts</a></p>
</div>`,
  })
}
