#!/usr/bin/env node
// Local server with file watching.
//
//   node dev.js            authoring mode -- drafts visible, live reload
//   node dev.js --prod     preview mode   -- byte-for-byte what deploys
//
// Preview mode passes no flags to the build, so drafts are omitted and no
// live-reload script is injected; what you see is exactly the artifact the
// GitHub Actions workflow uploads. It still watches and rebuilds, so a manual
// refresh picks up changes.
//
// Rebuilds run in a child process so that edits to site.config.js and lib/*
// take effect too -- ES module caching would otherwise keep the old copies.

import http from 'node:http'
import fs from 'node:fs/promises'
import path from 'node:path'
import { spawn } from 'node:child_process'
import chokidar from 'chokidar'

import config from './site.config.js'

const ROOT = import.meta.dirname

// The site is served under a base path in production, so serve it under one
// here too and local URLs match the deployed ones. Changing `base` needs a
// dev-server restart -- only the build runs in a child process.
const BASE = (config.base ?? '').replace(/\/+$/, '')

const PROD = process.argv.includes('--prod')
const PORT = Number(process.env.PORT) || (PROD ? 4000 : 3000)

// Preview builds to its own directory so it can run alongside `npm run dev`
// without the two wiping each other's output.
const DIST = path.join(ROOT, PROD ? 'dist-preview' : 'dist')
const BUILD_ARGS = PROD ? ['--out', DIST] : ['--dev', '--drafts']

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.xml': 'application/xml; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.avif': 'image/avif',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.pdf': 'application/pdf',
}

// --- Build -----------------------------------------------------------------

let building = false
let queued = false

function rebuild() {
  if (building) {
    queued = true
    return
  }
  building = true

  const child = spawn(process.execPath, [path.join(ROOT, 'build.js'), ...BUILD_ARGS], {
    stdio: 'inherit',
  })

  child.on('exit', (code) => {
    building = false
    if (code === 0) notifyClients()
    if (queued) {
      queued = false
      rebuild()
    }
  })
}

// --- Live reload -----------------------------------------------------------

const clients = new Set()

function notifyClients() {
  for (const res of clients) res.write('data: reload\n\n')
}

// --- Server ----------------------------------------------------------------

async function resolveFile(urlPath) {
  // Keep requests inside dist/.
  const decoded = decodeURIComponent(urlPath.split('?')[0]).slice(BASE.length)
  const target = path.join(DIST, path.normalize(decoded).replace(/^(\.\.[/\\])+/, ''))
  if (!target.startsWith(DIST)) return null

  const candidates = target.endsWith('/')
    ? [path.join(target, 'index.html')]
    : [target, path.join(target, 'index.html'), `${target}.html`]

  for (const candidate of candidates) {
    try {
      const stat = await fs.stat(candidate)
      if (stat.isFile()) return candidate
    } catch {
      /* try the next candidate */
    }
  }
  return null
}

const server = http.createServer(async (req, res) => {
  if (req.url === '/__reload') {
    if (PROD) {
      res.writeHead(404).end()
      return
    }
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    })
    res.write(': connected\n\n')
    clients.add(res)
    req.on('close', () => clients.delete(res))
    return
  }

  // Anything outside the base path is a different site in production. Locally
  // it's almost always someone opening http://localhost:PORT/, so send them in.
  const pathname = req.url.split('?')[0]
  if (BASE && pathname !== BASE && !pathname.startsWith(`${BASE}/`)) {
    res.writeHead(302, { Location: `${BASE}/` }).end()
    return
  }

  const file = await resolveFile(req.url)

  if (!file) {
    const notFound = path.join(DIST, '404.html')
    const body = await fs.readFile(notFound).catch(() => 'Not found')
    res.writeHead(404, { 'Content-Type': 'text/html; charset=utf-8' })
    res.end(body)
    return
  }

  res.writeHead(200, {
    'Content-Type': MIME[path.extname(file)] || 'application/octet-stream',
    'Cache-Control': 'no-store',
  })
  res.end(await fs.readFile(file))
})

// --- Start -----------------------------------------------------------------

rebuild()

chokidar
  .watch(['src', 'lib', 'build.js', 'site.config.js'], {
    cwd: ROOT,
    ignoreInitial: true,
  })
  .on('all', (_event, file) => {
    console.log(`changed: ${file}`)
    rebuild()
  })

server.listen(PORT, () => {
  console.log(
    PROD
      ? `\n  http://localhost:${PORT}${BASE}/  — production preview: drafts hidden, no live reload (refresh manually)\n`
      : `\n  http://localhost:${PORT}${BASE}/  — authoring: drafts visible, live reload on\n`,
  )
})
