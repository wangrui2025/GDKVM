// v3: cache-first is now scoped to content-hashed /_astro/ assets; everything
// else moved to stale-while-revalidate. The version bump is required — the
// `activate` handler deletes caches under any other name, which is what
// evicts the permanently-pinned entries left behind by the v2 policy.
const CACHE_NAME = "gdkvm-shell-v3";
const STATIC_ASSETS = [
  "/GDKVM/",
  "/GDKVM/favicon.png",
  "/GDKVM/en/",
  "/GDKVM/zh/",
  "/GDKVM/en/tool/",
  "/GDKVM/zh/tool/",
  "/GDKVM/en/reprod/",
  "/GDKVM/zh/reprod/",
  // Per-locale 404 pages (created in [lang]/404.astro). Without these,
  // users hitting a stale URL while offline see the browser default
  // 404 page instead of the themed one.
  "/GDKVM/en/404/",
  "/GDKVM/zh/404/",
  "/GDKVM/manifest.json",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS)),
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== CACHE_NAME)
            .map((key) => caches.delete(key)),
        ),
      ),
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  if (request.mode === "navigate") {
    event.respondWith(
      caches.open(CACHE_NAME).then((cache) =>
        cache.match(request).then((cached) => {
          const fetched = fetch(request)
            .then((response) => {
              if (response.ok) cache.put(request, response.clone());
              return response;
            })
            .catch(() => cached);
          return cached || fetched;
        }),
      ),
    );
    return;
  }

  if (url.origin === location.origin) {
    // GitHub Pages serves every asset with a short, non-configurable
    // Cache-Control, so the SW is the only place we can express the
    // immutable-vs-revalidate distinction:
    //
    //   /_astro/*  → content-hashed by Astro; a changed byte means a changed
    //                URL, so cache-first is safe and permanent (`immutable`).
    //   everything → mutable at a stable URL (favicon.png, manifest.json,
    //   else         sw-precached images). Cache-first pinned these for the
    //                lifetime of the cache; serve the cached copy for speed
    //                but revalidate in the background (stale-while-revalidate)
    //                so the next load is fresh.
    const isImmutable = url.pathname.includes("/_astro/");

    event.respondWith(
      caches.open(CACHE_NAME).then((cache) =>
        cache.match(request).then((cached) => {
          if (cached && isImmutable) return cached;

          const fetched = fetch(request)
            .then((response) => {
              if (response.ok) cache.put(request, response.clone());
              return response;
            })
            .catch(() => cached);

          // stale-while-revalidate: cached copy now, refresh for next time.
          return cached || fetched;
        }),
      ),
    );
  }
});
