/**
 * @jest-environment jsdom
 */

/**
 * Tests for interface/modules/custom_modules/oe-module-clinical-copilot/public/copilot-sw.js
 *
 * Covers the P2.15 standalone route's service worker caching-decision
 * predicate: isCacheable(request). This is a deny-by-default allowlist, not
 * a blocklist -- the heart of the security control in plan §4.7 is that PHI
 * (which only ever travels through ajax.php / chat-proxy.php / /apis/
 * routes, or any request carrying a query string) must NEVER be written to
 * Cache Storage. An unanticipated URL shape must be denied, not cached
 * "just in case".
 *
 * Run with: npm test -- tests/js/clinical-copilot-sw.test.js
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

const fs = require('fs');
const path = require('path');

// The service worker file runs in a ServiceWorkerGlobalScope in production
// (no `window`/`document`), so it is loaded here the same way
// clinical-copilot-chat.test.js loads copilot-chat.js -- read the source and
// evaluate it against a fake `self` that has no fetch/caches listener wiring,
// so only the pure isCacheable() export is exercised.
const src = fs.readFileSync(
    path.resolve(
        __dirname,
        '../../interface/modules/custom_modules/oe-module-clinical-copilot/public/copilot-sw.js'
    ),
    'utf8'
);

const fakeSelf = { location: { href: 'https://example.test/interface/modules/custom_modules/oe-module-clinical-copilot/public/copilot.php' } };
new Function('self', src)(fakeSelf);

const { isCacheable } = fakeSelf.CopilotServiceWorker;

const BASE = 'https://example.test/interface/modules/custom_modules/oe-module-clinical-copilot/public/';

function req(pathAndQuery, method) {
    return { url: BASE + pathAndQuery, method: method || 'GET' };
}

describe('isCacheable', () => {
    test.each([
        ['assets/js/copilot-chat.js'],
        ['assets/js/copilot.js'],
        ['assets/css/copilot.css'],
        ['assets/icons/icon-192.png'],
        ['assets/icons/icon-512.png'],
        ['manifest.json'],
        ['copilot-sw.js']
    ])('static asset %s is cacheable', (p) => {
        expect(isCacheable(req(p))).toBe(true);
    });

    test.each([
        ['ajax.php'],
        ['chat-proxy.php'],
        ['../../../../apis/default/api/patient'],
        ['apis/default/api/patient']
    ])('%s is never cacheable', (p) => {
        expect(isCacheable(req(p))).toBe(false);
    });

    test.each([
        ['copilot.php/report.js'],
        ['copilot.php/x.png'],
        ['ajax.php/context.js'],
        ['chat-proxy.php/data.css']
    ])('PATH_INFO trick %s (dynamic .php endpoint with a static-looking suffix) is never cacheable', (p) => {
        expect(isCacheable(req(p))).toBe(false);
    });

    test('a static-looking path is never cacheable if it carries a query string (may carry context)', () => {
        expect(isCacheable(req('assets/js/copilot-chat.js?pid=1'))).toBe(false);
        expect(isCacheable(req('manifest.json?v=2'))).toBe(false);
    });

    test('non-GET methods are never cacheable, even for a static-looking path', () => {
        expect(isCacheable(req('assets/js/copilot.js', 'POST'))).toBe(false);
        expect(isCacheable(req('ajax.php', 'POST'))).toBe(false);
        expect(isCacheable(req('assets/css/copilot.css', 'HEAD'))).toBe(false);
    });

    test('deny-by-default: an unanticipated path shape is never cached (allowlist, not blocklist)', () => {
        expect(isCacheable(req('some/unknown/route'))).toBe(false);
        expect(isCacheable(req('report.pdf'))).toBe(false);
        expect(isCacheable(req(''))).toBe(false);
    });

    test('handles a malformed request gracefully (no throw, denies)', () => {
        expect(() => isCacheable(null)).not.toThrow();
        expect(isCacheable(null)).toBe(false);
        expect(isCacheable({})).toBe(false);
    });
});
