#!/usr/bin/env node
// Static site build. Reads Markdown from src/, writes a complete site to dist/.

import fs from 'node:fs/promises'
import path from 'node:path'
import matter from 'gray-matter'

import config from './site.config.js'
import { renderMarkdown, renderInline, toPlainText } from './lib/markdown.js'
import { renderIndex, renderPost, renderPage, render404, isoDate } from './lib/templates.js'
import { renderFeed, renderSitemap, renderRobots } from './lib/feed.js'
import { buildAssets } from './lib/assets.js'

const ROOT = import.meta.dirname
const SRC = path.join(ROOT, 'src')

// '' for a root-hosted site, '/repo' for a GitHub Pages project site.
// Normalised here so site.config.js can be written either way.
const BASE = (config.base ?? '').replace(/\/+$/, '')

const FILENAME_RE = /^(\d{4}-\d{2}-\d{2})[-_]?(.*)\.md$/


// ---------------------------------------------------------------------------

async function readMarkdownDir(dir) {
  let names
  try {
    names = await fs.readdir(dir)
  } catch {
    return []
  }

  const files = names.filter((n) => n.endsWith('.md') && !n.startsWith('_'))
  return Promise.all(
    files.map(async (name) => ({
      name,
      raw: await fs.readFile(path.join(dir, name), 'utf8'),
    })),
  )
}

async function loadPost({ name, raw }, { drafts }) {
  const { data, content } = matter(raw)
  if (data.draft && !drafts) return null

  const match = name.match(FILENAME_RE)
  const date = data.date ?? match?.[1]
  if (!date) {
    throw new Error(
      `${name}: no date. Add \`date: YYYY-MM-DD\` to the frontmatter, or name the file YYYY-MM-DD-slug.md`,
    )
  }
  if (!data.title) throw new Error(`${name}: missing \`title\` in frontmatter`)

  const slug = data.slug ?? match?.[2] ?? name.replace(/\.md$/, '')
  const { html, toc, hasMath, words } = await renderMarkdown(content, config.shiki, BASE)

  return {
    slug,
    url: `/posts/${slug}/`,
    title: data.title,
    titleHtml: renderInline(data.title),
    date: isoDate(date),
    updated: data.updated ? isoDate(data.updated) : null,
    description: data.description || toPlainText(content),
    tags: Array.isArray(data.tags) ? data.tags : data.tags ? [data.tags] : [],
    draft: Boolean(data.draft),
    html,
    toc,
    hasMath,
    readingTime: Math.max(1, Math.round(words / config.wordsPerMinute)),
  }
}

async function loadPage({ name, raw }) {
  const { data, content } = matter(raw)
  const slug = data.slug ?? name.replace(/\.md$/, '')
  const { html, toc, hasMath } = await renderMarkdown(content, config.shiki, BASE)

  return {
    slug,
    url: slug === 'index' ? '/' : `/${slug}/`,
    title: data.title || slug,
    titleHtml: renderInline(data.title || slug),
    description: data.description || toPlainText(content),
    draft: Boolean(data.draft),
    html,
    toc,
    hasMath,
  }
}

// ---------------------------------------------------------------------------

async function write(outDir, relPath, contents) {
  const target = path.join(outDir, relPath)
  await fs.mkdir(path.dirname(target), { recursive: true })
  await fs.writeFile(target, contents)
}

export async function build({ dev = false, drafts = false, outDir } = {}) {
  const started = performance.now()
  const out = outDir ?? path.join(ROOT, 'dist')

  await fs.rm(out, { recursive: true, force: true })
  await fs.mkdir(out, { recursive: true })

  const [postFiles, pageFiles] = await Promise.all([
    readMarkdownDir(path.join(SRC, 'posts')),
    readMarkdownDir(path.join(SRC, 'pages')),
  ])

  const posts = (await Promise.all(postFiles.map((f) => loadPost(f, { drafts }))))
    .filter(Boolean)
    .sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : a.slug < b.slug ? 1 : -1))

  const allPages = await Promise.all(pageFiles.map(loadPage))
  const pages = allPages.filter((p) => drafts || !p.draft)

  // A drafted page must not be reachable, so drop any header or footer link
  // pointing at one -- otherwise hiding a page leaves a link to a 404.
  const hidden = new Set(
    allPages.filter((p) => !pages.includes(p)).map((p) => p.url),
  )
  const site = {
    ...config,
    // '' for a root-hosted site, '/repo' for a GitHub Pages project site.
    // Normalised here so site.config.js can be written either way.
    base: BASE,
    nav: config.nav.filter((l) => !hidden.has(l.href)),
    links: config.links.filter((l) => !hidden.has(l.href)),
  }

  // Fail loudly on duplicate URLs rather than silently overwriting.
  const seen = new Set(['/'])
  for (const item of [...posts, ...pages]) {
    if (seen.has(item.url)) throw new Error(`Duplicate URL: ${item.url}`)
    seen.add(item.url)
  }

  await Promise.all([
    write(out, 'index.html', renderIndex({ config: site, posts, dev })),
    write(out, '404.html', render404({ config: site, dev })),
    write(out, 'feed.xml', renderFeed({ config: site, posts })),
    write(out, 'robots.txt', renderRobots({ config: site })),
    write(
      out,
      'sitemap.xml',
      renderSitemap({
        config: site,
        urls: [
          { loc: '/', lastmod: posts[0]?.date },
          ...posts.map((p) => ({ loc: p.url, lastmod: p.updated || p.date })),
          ...pages.map((p) => ({ loc: p.url })),
        ],
      }),
    ),

    ...posts.map((post, i) =>
      write(
        out,
        `${post.url}index.html`,
        renderPost({ config: site, post, prev: posts[i + 1], next: posts[i - 1], dev }),
      ),
    ),

    ...pages.map((page) =>
      write(out, `${page.url}index.html`, renderPage({ config: site, page, dev })),
    ),

    // Tells GitHub Pages not to run the output through Jekyll.
    write(out, '.nojekyll', ''),

    buildAssets(out, { dev }),
  ])

  const ms = Math.round(performance.now() - started)
  const count = (items, noun) => {
    if (!items.length) return ''
    const drafted = items.filter((i) => i.draft).length
    return `${items.length} ${noun}${items.length === 1 ? '' : 's'}${drafted ? ` (${drafted} draft)` : ''}`
  }

  const summary = [count(posts, 'post'), count(pages, 'page')].filter(Boolean).join(', ')
  console.log(`${summary || 'nothing'} → ${path.relative(ROOT, out)}/ in ${ms}ms`)

  // Easy to hit while testing drafts, and baffling without an explanation.
  if (!drafts && !posts.length) {
    const hiddenCount = postFiles.length + pageFiles.length - posts.length - pages.length
    if (hiddenCount) {
      console.log(
        `note: ${hiddenCount} file${hiddenCount === 1 ? '' : 's'} skipped as \`draft: true\`. ` +
          'Run `npm run dev` to see them, or remove the frontmatter line to publish.',
      )
    }
  }

  if (config.comments.enabled && !config.comments.repoId) {
    console.log(
      'note: comments are off until you set comments.repoId / categoryId in site.config.js (see README)',
    )
  }

  return { posts, pages }
}

// Run directly (not when imported by dev.js).
if (process.argv[1] === import.meta.filename) {
  const argv = process.argv.slice(2)
  const outFlag = argv.indexOf('--out')

  build({
    drafts: argv.includes('--drafts'),
    dev: argv.includes('--dev'),
    outDir: outFlag === -1 ? undefined : path.resolve(argv[outFlag + 1]),
  }).catch((err) => {
    console.error(`\nbuild failed: ${err.message}\n`)
    process.exit(1)
  })
}
