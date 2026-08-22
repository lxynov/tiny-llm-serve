#!/usr/bin/env node
// Scaffolds a new post:  npm run new "Paged Attention and the KV Cache"

import fs from 'node:fs/promises'
import path from 'node:path'
import { slugify } from '../lib/markdown.js'

const title = process.argv.slice(2).join(' ').trim()
if (!title) {
  console.error('usage: npm run new "Post title"')
  process.exit(1)
}

const date = new Date().toISOString().slice(0, 10)
const slug = slugify(title)
const file = path.join(import.meta.dirname, '..', 'src', 'posts', `${date}-${slug}.md`)

try {
  await fs.writeFile(
    file,
    `---
title: ${title}
date: ${date}
description:
tags: []
draft: true
---

`,
    { flag: 'wx' },
  )
  console.log(`created src/posts/${path.basename(file)}`)
} catch (err) {
  if (err.code === 'EEXIST') console.error(`already exists: ${path.basename(file)}`)
  else throw err
  process.exit(1)
}
