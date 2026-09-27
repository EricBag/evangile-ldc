// sw.js — Service worker minimal pour rendre l'application installable (PWA)
// Stratégie : "network-first" sur la navigation, avec repli sur le cache hors-ligne.
// Les requêtes API (POST) ne sont jamais mises en cache.

const CACHE = "evangile-ldc-v12";
const SHELL = [
  "/",
  "/static/styles.css?v=10",
  "/static/luisa.jpg?v=6",
  "/static/faustine.jpg?v=6",
  "/static/icon-192.png",
  "/static/icon-512.png",
  "/static/manifest.json?v=2",
  "/static/diptyque.png?v=3",
  "/static/diptyque@2x.png?v=3",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(SHELL)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const req = event.request;

  // Ne gérer que les GET ; laisser passer les POST (API) au réseau directement.
  if (req.method !== "GET") return;
  // Fichiers d'essai (static/_essais/, hors dépôt) : jamais mis en cache.
  if (new URL(req.url).pathname.startsWith("/static/_essais/")) return;

  event.respondWith(
    fetch(req)
      .then((res) => {
        // Mettre en cache une copie des GET réussis (même origine).
        if (res && res.ok && req.url.startsWith(self.location.origin)) {
          const copy = res.clone();
          caches.open(CACHE).then((cache) => cache.put(req, copy)).catch(() => {});
        }
        return res;
      })
      .catch(() => caches.match(req).then((hit) => hit || caches.match("/")))
  );
});
