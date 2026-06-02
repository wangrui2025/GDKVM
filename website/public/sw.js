const CACHE_NAME = 'gdkvm-shell-v1';
const STATIC_ASSETS = [
  '/GDKVM/',
  '/GDKVM/favicon.png',
  '/GDKVM/en/',
  '/GDKVM/zh/',
  '/GDKVM/en/tool/',
  '/GDKVM/zh/tool/',
  '/GDKVM/en/reprod/',
  '/GDKVM/zh/reprod/',
  // Per-locale 404 pages (created in [lang]/404.astro). Without these,
  // users hitting a stale URL while offline see the browser default
  // 404 page instead of the themed one.
  '/GDKVM/en/404/',
  '/GDKVM/zh/404/',
  '/GDKVM/manifest.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  if (request.mode === 'navigate') {
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
        })
      )
    );
    return;
  }

  if (url.origin === location.origin) {
    event.respondWith(
      caches.match(request).then((cached) => {
        if (cached) return cached;
        return fetch(request).then((response) => {
          if (response.ok) {
            caches.open(CACHE_NAME).then((cache) => cache.put(request, response.clone()));
          }
          return response;
        });
      })
    );
  }
});
