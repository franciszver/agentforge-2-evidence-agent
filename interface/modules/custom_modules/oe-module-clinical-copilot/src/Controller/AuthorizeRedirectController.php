<?php

/**
 * Clinical Co-Pilot OAuth Authorize-Redirect Controller (#124 Phase 2b).
 *
 * Initiates the browser consent flow: it builds the OpenEMR
 * /oauth2/default/authorize URL and 302s the user's browser to it. Every
 * security-sensitive parameter is server-derived, never taken from request
 * input:
 *   - `state` is CSRF-bound to the session via CsrfUtils (a dedicated subject),
 *     so the callback can reject a forged/mismatched value in constant time;
 *   - PKCE is S256: a fresh per-request `code_verifier` is generated and stored
 *     SERVER-SIDE in the session; only its SHA-256 `code_challenge` is put on
 *     the wire (the verifier itself never leaves the server);
 *   - the SMART `launch` token is built from the session `pid` (never a request
 *     value), via an injected factory;
 *   - `redirect_uri`, `client_id`, `scope` and `aud` come from OAuthConsentConfig.
 *
 * The whole flow is gated by OAuthConsentConfig::$enabled; when off this
 * endpoint 404s and the DevAgentToken path is untouched.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Controller;

use OpenEMR\BC\ServiceContainer;
use OpenEMR\Common\Csrf\CsrfUtils;
use OpenEMR\Common\Session\PatientSessionUtil;
use OpenEMR\Common\Session\SessionUtil;
use OpenEMR\Common\Session\SessionWrapperFactory;
use OpenEMR\Modules\ClinicalCopilot\Auth\LaunchTokenFactory;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthConsentConfig;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthConsentSession;
use OpenEMR\Modules\ClinicalCopilot\Auth\PkcePair;
use Symfony\Component\HttpFoundation\Session\SessionInterface;

final class AuthorizeRedirectController
{
    public function __construct(
        private readonly OAuthConsentConfig $config,
        private readonly LaunchTokenFactory $launchTokenFactory,
        private readonly string $requestMethod = 'GET',
    ) {
    }

    public function handleRequest(): void
    {
        if (!$this->config->enabled) {
            $this->fail(404, 'This feature is not enabled.');
            return;
        }

        if ($this->requestMethod !== 'GET') {
            $this->fail(405, 'Method not allowed.');
            return;
        }

        $session = SessionWrapperFactory::getInstance()->getActiveSession();

        $rawAuthUserId = $session->get('authUserID');
        $authUserId = is_numeric($rawAuthUserId) ? (int) $rawAuthUserId : 0;
        if ($authUserId <= 0) {
            $this->fail(401, 'You must be signed in to authorize the Co-Pilot.');
            return;
        }

        $pid = PatientSessionUtil::getPid();
        if ($pid <= 0) {
            $this->fail(400, 'Open a patient chart before authorizing the Co-Pilot.');
            return;
        }

        try {
            $url = $this->buildAuthorizeUrl($session, $pid);
        } catch (\RuntimeException $e) {
            ServiceContainer::getLogger()->error('Co-Pilot authorize redirect failed to build', ['exception' => $e]);
            $this->fail(500, 'Could not start authorization. Please try again.');
            return;
        }

        header('Location: ' . $url, true, 302);
    }

    /**
     * Build the authorize URL and, as a side effect, persist the per-request
     * PKCE verifier in the session. State is derived (not stored) from the
     * session CSRF key so it can be re-derived and compared on callback.
     */
    public function buildAuthorizeUrl(SessionInterface $session, int $pid): string
    {
        $pkce = PkcePair::generate();
        // Server-side only: SessionUtil::setSession routes through the writable
        // active session (the verifier never travels to the client).
        SessionUtil::setSession(OAuthConsentSession::CODE_VERIFIER_KEY, $pkce->verifier);

        $params = [
            'response_type' => 'code',
            'client_id' => $this->config->clientId,
            'redirect_uri' => $this->config->redirectUri,
            'scope' => $this->config->scope,
            // collectCsrfToken throws if the session has no CSRF key; unreachable
            // here (handleRequest's auth gate guarantees an authenticated session,
            // which always has one) and, if it ever were, the caller's catch fails safe.
            'state' => CsrfUtils::collectCsrfToken($session, OAuthConsentSession::STATE_SUBJECT),
            'code_challenge' => $pkce->challenge,
            'code_challenge_method' => 'S256',
            'aud' => $this->config->audience,
            'launch' => $this->launchTokenFactory->create($pid),
        ];

        return $this->config->authorizeUrl . '?' . http_build_query($params);
    }

    private function fail(int $status, string $message): void
    {
        http_response_code($status);
        header('Content-Type: text/html; charset=utf-8');
        echo '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            . '<title>Clinical Co-Pilot</title></head><body><p>'
            . htmlspecialchars($message, ENT_QUOTES, 'UTF-8')
            . '</p></body></html>';
    }
}
