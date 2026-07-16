<?php

/**
 * Clinical Co-Pilot OAuth Callback Controller (#124 Phase 2b).
 *
 * Browser-facing endpoint OpenEMR redirects back to after consent. It is the
 * security-critical half of the flow and enforces, in order:
 *   1. the feature flag (off => 404, DevAgentToken path untouched);
 *   2. GET-only, authenticated session;
 *   3. `state` validation against the session CSRF key (CsrfUtils, constant-time
 *      hash_equals) BEFORE any code is exchanged -- a missing or forged state is
 *      rejected outright;
 *   4. the PKCE `code_verifier` must be present SERVER-SIDE in the session (it is
 *      never accepted from the request);
 *   5. the `database_encryption` precondition: with it OFF we REFUSE to store,
 *      rather than silently persisting a plaintext long-lived refresh token;
 *   6. the token exchange (injected seam) with fail-safe error handling;
 *   7. an empty-refresh-token guard;
 *   8. encrypted storage via the Phase 2a repository (atomic replace = rotation),
 *      then the one-time verifier is burned.
 *
 * Every failure returns a generic, detail-free response -- exception messages
 * are logged server-side only, never rendered.
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
use OpenEMR\Common\Crypto\CryptoGenException;
use OpenEMR\Common\Csrf\CsrfUtils;
use OpenEMR\Common\Session\SessionUtil;
use OpenEMR\Common\Session\SessionWrapperFactory;
use OpenEMR\Modules\ClinicalCopilot\Auth\AuthorizationCodeExchanger;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthConsentConfig;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthConsentSession;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthExchangeException;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthTokenResponse;
use OpenEMR\Modules\ClinicalCopilot\Auth\UserOAuthTokenRepository;

final class OAuthCallbackController
{
    /** Stable marker rendered on the success page (asserted by the isolated test). */
    public const SUCCESS_MARKER = 'clinical-copilot-oauth-complete';

    public function __construct(
        private readonly OAuthConsentConfig $config,
        private readonly AuthorizationCodeExchanger $exchanger,
        private readonly UserOAuthTokenRepository $repository,
        private readonly bool $databaseEncryptionEnabled,
        private readonly string $requestMethod = 'GET',
        private readonly ?string $code = null,
        private readonly ?string $state = null,
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
            $this->fail(401, 'You must be signed in to complete authorization.');
            return;
        }

        // From here the one-time PKCE verifier is treated as consumed: it is burned
        // on EVERY terminal path (success or failure) via the finally below, so a
        // failed, forged, or replayed callback can never reuse it. The burn is a
        // no-op when no verifier is present.
        try {
            // (3) State CSRF: constant-time comparison against the session key, before
            // any code is exchanged. Rejects missing, empty, or forged state.
            if (
                !is_string($this->state)
                || $this->state === ''
                || !CsrfUtils::verifyCsrfToken($this->state, $session, OAuthConsentSession::STATE_SUBJECT)
            ) {
                $this->fail(403, 'Authorization could not be verified. Please try again.');
                return;
            }

            if (!is_string($this->code) || $this->code === '') {
                $this->fail(400, 'Authorization could not be completed. Please try again.');
                return;
            }

            // (4) PKCE verifier must exist server-side; it is never taken from the request.
            $verifier = $session->get(OAuthConsentSession::CODE_VERIFIER_KEY);
            if (!is_string($verifier) || $verifier === '') {
                $this->fail(400, 'Your authorization session expired. Please try again.');
                return;
            }

            // (5) Refuse to store long-lived refresh tokens if the site would persist
            // them in plaintext. This is checked BEFORE the exchange so we never even
            // obtain a token we cannot store safely.
            if (!$this->databaseEncryptionEnabled) {
                ServiceContainer::getLogger()->error(
                    'Co-Pilot OAuth: refusing to store tokens because database_encryption is disabled',
                    ['user' => $authUserId],
                );
                $this->fail(500, 'Secure token storage is not configured. Contact your administrator.');
                return;
            }

            // (6) Exchange, fail-safe.
            try {
                $token = $this->exchanger->exchange($this->code, $verifier);
            } catch (OAuthExchangeException $e) {
                ServiceContainer::getLogger()->error('Co-Pilot OAuth token exchange failed', ['exception' => $e]);
                $this->fail(400, 'Authorization could not be completed. Please try again.');
                return;
            }

            // (7) Empty-refresh-token guard: without a refresh token the record is useless.
            if ($token->refreshToken === '') {
                ServiceContainer::getLogger()->error('Co-Pilot OAuth: exchange returned an empty refresh token', ['user' => $authUserId]);
                $this->fail(400, 'Authorization could not be completed. Please try again.');
                return;
            }

            // (8) Store encrypted (atomic replace = refresh-token rotation).
            try {
                $this->store($authUserId, $token);
            } catch (\RuntimeException | CryptoGenException $e) {
                ServiceContainer::getLogger()->error('Co-Pilot OAuth token storage failed', ['exception' => $e]);
                $this->fail(500, 'Could not save authorization. Please try again.');
                return;
            }

            $this->success();
        } finally {
            // Burn the one-time verifier on all outcomes (writable-session helper).
            SessionUtil::unsetSession(OAuthConsentSession::CODE_VERIFIER_KEY);
        }
    }

    private function store(int $authUserId, OAuthTokenResponse $token): void
    {
        $this->repository->upsert(
            $authUserId,
            $token->refreshToken,
            $token->accessToken,
            $token->accessTokenExpiresAt,
        );
    }

    private function success(): void
    {
        http_response_code(200);
        header('Content-Type: text/html; charset=utf-8');
        echo '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            . '<title>Clinical Co-Pilot</title></head><body>'
            . '<!-- ' . self::SUCCESS_MARKER . ' -->'
            . '<p>Clinical Co-Pilot authorization complete. You may close this window.</p>'
            . '</body></html>';
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
