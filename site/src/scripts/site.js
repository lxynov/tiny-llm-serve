// Progressive enhancement only -- every page is fully readable without this.

const root = document.documentElement
const darkQuery = matchMedia('(prefers-color-scheme: dark)')

const activeTheme = () => root.dataset.theme || (darkQuery.matches ? 'dark' : 'light')

// --- Theme toggle ----------------------------------------------------------
// Clicking back to whatever the OS says drops the override entirely, so the
// site resumes following the system preference.

function initTheme() {
  const button = document.querySelector('.theme-toggle')
  if (!button) return

  button.addEventListener('click', () => {
    const next = activeTheme() === 'dark' ? 'light' : 'dark'

    if ((next === 'dark') === darkQuery.matches) {
      delete root.dataset.theme
      localStorage.removeItem('theme')
    } else {
      root.dataset.theme = next
      localStorage.setItem('theme', next)
    }
    syncThemeImages()
    syncGiscusTheme()
  })

  darkQuery.addEventListener('change', () => {
    syncThemeImages()
    syncGiscusTheme()
  })
}

// --- Theme-paired figures --------------------------------------------------
// `<picture>` already follows the OS preference on its own; this only has to
// take over when the toggle overrides it. Setting an unconditional media query
// either way is simpler than restoring the original one, since `activeTheme()`
// has already folded the OS preference in.

function syncThemeImages() {
  const dark = activeTheme() === 'dark'
  for (const source of document.querySelectorAll('picture source[data-theme="dark"]')) {
    source.media = dark ? 'all' : 'not all'
  }
}

// --- Table of contents -----------------------------------------------------

function initToc() {
  const toc = document.querySelector('.toc')
  if (!toc) return

  // Sidebar on wide screens, collapsed disclosure on narrow ones.
  const wide = matchMedia('(min-width: 74.0625rem)')
  const setOpen = () => {
    toc.open = wide.matches
  }
  setOpen()
  wide.addEventListener('change', setOpen)

  const links = [...toc.querySelectorAll('a[href^="#"]')]
  const targets = links
    .map((link) => ({
      link,
      heading: document.getElementById(decodeURIComponent(link.hash.slice(1))),
    }))
    .filter((t) => t.heading)

  if (!targets.length) return

  // Heading positions are measured once and cached, so scrolling costs a little
  // arithmetic instead of a forced layout per heading per frame. That also
  // keeps the highlight correct without depending on requestAnimationFrame.
  let offsets = []
  const measure = () => {
    offsets = targets.map((t) => t.heading.getBoundingClientRect().top + window.scrollY)
  }

  let current = null

  function update() {
    const line = window.scrollY + 100
    const atBottom =
      window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2

    let index = 0
    if (atBottom) {
      index = targets.length - 1
    } else {
      for (let i = 0; i < offsets.length; i++) {
        if (offsets[i] > line) break
        index = i
      }
    }

    const found = targets[index]
    if (found === current) return

    current?.link.classList.remove('active')
    found.link.classList.add('active')
    current = found
    keepVisible(toc, found.link)
  }

  const remeasure = () => {
    measure()
    update()
  }

  measure()
  update()

  addEventListener('scroll', update, { passive: true })
  addEventListener('resize', remeasure, { passive: true })
  // Webfonts land after first paint and shift every heading down.
  document.fonts?.ready.then(remeasure)
}

// Scroll the TOC itself (never the page) so the active entry stays in view.
function keepVisible(container, element) {
  if (container.scrollHeight <= container.clientHeight) return
  const top = element.offsetTop - container.offsetTop
  const bottom = top + element.offsetHeight

  if (top < container.scrollTop) {
    container.scrollTop = top - 8
  } else if (bottom > container.scrollTop + container.clientHeight) {
    container.scrollTop = bottom - container.clientHeight + 8
  }
}

// --- Copy buttons ----------------------------------------------------------

function initCopyButtons() {
  if (!navigator.clipboard) return

  for (const block of document.querySelectorAll('.code-block')) {
    const code = block.querySelector('pre code')
    if (!code) continue

    const button = document.createElement('button')
    button.className = 'copy-btn'
    button.type = 'button'
    button.textContent = 'copy'
    button.setAttribute('aria-label', 'Copy code to clipboard')

    let timer
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.textContent)
        button.textContent = 'copied'
      } catch {
        button.textContent = 'failed'
      }
      clearTimeout(timer)
      timer = setTimeout(() => {
        button.textContent = 'copy'
      }, 1600)
    })

    block.append(button)
  }
}

// --- giscus ----------------------------------------------------------------
// Injected from here rather than inlined in the HTML so the right theme is
// known before the widget boots, and so it only loads if you scroll to it.

const GISCUS_ORIGIN = 'https://giscus.app'
let giscusSettings = null

function initComments() {
  const container = document.getElementById('comments')
  if (!container?.dataset.giscus) return

  giscusSettings = JSON.parse(container.dataset.giscus)

  const observer = new IntersectionObserver(
    (entries) => {
      if (!entries.some((e) => e.isIntersecting)) return
      observer.disconnect()
      loadGiscus(container)
    },
    { rootMargin: '400px' },
  )
  observer.observe(container)
}

function giscusTheme() {
  return activeTheme() === 'dark' ? giscusSettings.themeDark : giscusSettings.themeLight
}

function loadGiscus(container) {
  const script = document.createElement('script')
  script.src = `${GISCUS_ORIGIN}/client.js`
  script.async = true
  script.crossOrigin = 'anonymous'

  Object.assign(script.dataset, {
    repo: giscusSettings.repo,
    repoId: giscusSettings.repoId,
    category: giscusSettings.category,
    categoryId: giscusSettings.categoryId,
    mapping: giscusSettings.mapping,
    strict: '1',
    reactionsEnabled: giscusSettings.reactionsEnabled,
    emitMetadata: '0',
    inputPosition: 'top',
    theme: giscusTheme(),
    lang: giscusSettings.lang,
    loading: 'lazy',
  })

  container.append(script)
}

function syncGiscusTheme() {
  if (!giscusSettings) return
  const frame = document.querySelector('iframe.giscus-frame')
  frame?.contentWindow?.postMessage(
    { giscus: { setConfig: { theme: giscusTheme() } } },
    GISCUS_ORIGIN,
  )
}

// --- Boot ------------------------------------------------------------------

initTheme()
syncThemeImages() // a stored override is applied before this script runs
initToc()
initCopyButtons()
initComments()
