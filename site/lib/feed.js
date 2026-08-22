// Atom feed, sitemap and robots.txt.

import { esc, parseDate, absolute } from './templates.js'

/** Rewrite root-relative links so they still resolve inside a feed reader. */
function absolutize(html, base) {
  return html.replace(/\b(href|src)="\/(?!\/)/g, `$1="${base}/`)
}

export function renderFeed({ config, posts }) {
  const updated = posts.length
    ? parseDate(posts[0].date).toISOString()
    : new Date().toISOString()

  const entries = posts
    .slice(0, 20)
    .map((post) => {
      const url = absolute(config, post.url)
      return `  <entry>
    <title>${esc(post.title)}</title>
    <link href="${esc(url)}"/>
    <id>${esc(url)}</id>
    <updated>${parseDate(post.updated || post.date).toISOString()}</updated>
    <published>${parseDate(post.date).toISOString()}</published>
    <summary>${esc(post.description)}</summary>
    <content type="html">${esc(absolutize(post.html, config.url))}</content>
  </entry>`
    })
    .join('\n')

  return `<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>${esc(config.title)}</title>
  <subtitle>${esc(config.description)}</subtitle>
  <link href="${esc(absolute(config, '/feed.xml'))}" rel="self"/>
  <link href="${esc(absolute(config, '/'))}"/>
  <id>${esc(absolute(config, '/'))}</id>
  <updated>${updated}</updated>
  <author><name>${esc(config.author.name)}</name></author>
${entries}
</feed>
`
}

export function renderSitemap({ config, urls }) {
  const entries = urls
    .map(
      ({ loc, lastmod }) => `  <url>
    <loc>${esc(absolute(config, loc))}</loc>${lastmod ? `\n    <lastmod>${parseDate(lastmod).toISOString().slice(0, 10)}</lastmod>` : ''}
  </url>`,
    )
    .join('\n')

  return `<?xml version="1.0" encoding="utf-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${entries}
</urlset>
`
}

export function renderRobots({ config }) {
  return `User-agent: *
Allow: /

Sitemap: ${absolute(config, '/sitemap.xml')}
`
}
