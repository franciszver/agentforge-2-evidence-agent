/**
 * Clinical Co-Pilot module - service worker (P2.15).
 *
 * Static assets ONLY. This is a deliberate security control (plan §4.7):
 * every chat/API/ajax request carries or can carry PHI, so none of it may
 * ever be written to Cache Storage, which persists outside the normal
 * browser-session/cookie lifecycle and is not subject to the same
 * same-origin request auditing as a live fetch. isCacheable() is the single
 * gate every request passes through before this worker will even consider
 * caching it -- it is an allowlist (known-safe static file shapes), not a
 * blocklist, so a URL this module's authors never anticipated is denied by
 * default rather than cached "just in case".
 *
 * Lives at public/copilot-sw.js (not under assets/js/) on purpose: a service
 * worker's default registration scope is its own directory and everything
 * below it, and cannot be widened past that without a
 * `Service-Worker-Allowed` response header. Placing it under assets/js/
 * would scope it to assets/js/ only and it would never see requests for
 * copilot.php, ajax.php, or chat-proxy.php at all. Living in public/
 * directly gives it the same scope as every other route this module
 * serves.
 */
(function (self) {
    'use strict';

    var CACHE_NAME = 'copilot-static-v1';

    // Extensions that are always static, cacheable assets when nothing else
    // below vetoes the request.
    var STATIC_EXTENSIONS = ['.js', '.css', '.png', '.ico', '.svg', '.woff', '.woff2'];

    // Exact filenames (matched on the final path segment) that are static
    // and cacheable but don't share one of the extensions above.
    var STATIC_FILENAMES = ['manifest.json'];

    // Path substrings that must never be cached even if a future change to
    // the checks above would otherwise match them -- defense in depth for
    // the routes that can carry PHI. '.php' anywhere in the path denies
    // every dynamic PHP endpoint (this module's only PHI-bearing routes),
    // including PATH_INFO tricks that append a static-looking suffix such as
    // `copilot.php/x.js` -- Apache executes copilot.php and returns the full
    // session page (window.CopilotContext, CSRF token) on a URL that would
    // otherwise pass the static-extension allowlist below. No static asset
    // this worker legitimately caches contains '.php' in its path. '/apis/'
    // (the OpenEMR REST/FHIR surface, no '.php' in its path) is denied
    // separately.
    var NEVER_CACHE_SUBSTRINGS = ['.php', '/apis/'];

    function isCacheable(request) {
        if (!request || typeof request.url !== 'string' || request.method !== 'GET') {
            return false;
        }

        var url;
        try {
            url = new URL(request.url, self.location ? self.location.href : undefined);
        } catch (e) {
            return false;
        }

        // A query string can carry patient/context data even on an
        // otherwise-static-looking path (e.g. a cache-busting pid param);
        // never cache it.
        if (url.search) {
            return false;
        }

        for (var i = 0; i < NEVER_CACHE_SUBSTRINGS.length; i++) {
            if (url.pathname.indexOf(NEVER_CACHE_SUBSTRINGS[i]) !== -1) {
                return false;
            }
        }

        var lastSegment = url.pathname.slice(url.pathname.lastIndexOf('/') + 1);
        if (STATIC_FILENAMES.indexOf(lastSegment) !== -1) {
            return true;
        }

        for (var j = 0; j < STATIC_EXTENSIONS.length; j++) {
            var ext = STATIC_EXTENSIONS[j];
            if (url.pathname.slice(-ext.length) === ext) {
                return true;
            }
        }

        // Deny-by-default: an unanticipated path shape is never cached.
        return false;
    }

    if (typeof self.addEventListener === 'function') {
        self.addEventListener('fetch', function (event) {
            if (!isCacheable(event.request)) {
                return; // network-only; let the browser's default fetch handle it
            }
            event.respondWith(
                caches.open(CACHE_NAME).then(function (cache) {
                    return cache.match(event.request).then(function (cached) {
                        if (cached) {
                            return cached;
                        }
                        return fetch(event.request).then(function (response) {
                            if (response && response.status === 200) {
                                cache.put(event.request, response.clone());
                            }
                            return response;
                        });
                    });
                })
            );
        });
    }

    self.CopilotServiceWorker = {
        isCacheable: isCacheable
    };
})(typeof self !== 'undefined' ? self : this);
