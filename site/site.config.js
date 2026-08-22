// Site-wide configuration. Everything you'll routinely want to change lives here.

export default {
  // ---- Identity -----------------------------------------------------------
  title: 'journey of building tiny-llm-serve',
  tagline: 'a fully human-written logbook in plain English',
  description:
    'A tiny LLM inference and serving engine built from scratch in Python and PyTorch: KV caching, paged attention, continuous batching, and the benchmarks behind each step.',

  // Absolute origin, no trailing slash. Used for canonical URLs, RSS, sitemap.
  url: 'https://lxynov.github.io',

  // Path this site is served under, no trailing slash. GitHub Pages puts a
  // project site at /<repo>/ rather than at the domain root, so every internal
  // link and asset URL is prefixed with this. Use '' for a root-hosted site.
  base: '/tiny-llm-serve',

  lang: 'en',

  author: {
    name: 'Xingyuan Lin',
    github: 'lxynov',
    email: 'alxynov@gmail.com',
  },

  // ---- Header nav ---------------------------------------------------------
  // Keep this short. Minimalism is the point.
  nav: [],

  // ---- Footer links -------------------------------------------------------
  links: [
    { label: 'GitHub', href: 'https://github.com/lxynov/tiny-llm-serve' },
    { label: 'RSS', href: '/feed.xml' },
  ],

  // ---- Comments (giscus) --------------------------------------------------
  // Off for now. To turn it on, enable Discussions on the repo, install the
  // giscus GitHub App, and paste the ids from https://giscus.app below.
  comments: {
    enabled: false,
    repo: 'lxynov/tiny-llm-serve',
    repoId: '',
    category: 'Comments',
    categoryId: '',
    mapping: 'pathname',
    reactionsEnabled: true,
    // giscus theme names, synced with the site's light/dark toggle.
    themeLight: 'noborder_light',
    themeDark: 'noborder_gray',
  },

  // ---- Syntax highlighting ------------------------------------------------
  // Any theme from https://shiki.style/themes. Both are compiled into the
  // page as CSS variables, so switching modes needs no re-highlighting.
  shiki: {
    light: 'vitesse-light',
    dark: 'vitesse-dark',
  },

  // ---- Reading time -------------------------------------------------------
  wordsPerMinute: 200,
}
