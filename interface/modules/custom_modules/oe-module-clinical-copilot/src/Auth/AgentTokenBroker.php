<?php

/**
 * Decides which bearer the Co-Pilot panel hands the agent (#124 Phase 3).
 *
 * When the consent flag is OFF, this mints the DevAgentToken stand-in exactly
 * as the broker did before Phase 3 (regression-preserving default). When the
 * flag is ON it uses the Phase 2b per-user token stored by the consent flow:
 *   - a valid, unexpired access token is returned as-is;
 *   - an expired/near-expiry access token is refreshed on demand
 *     (grant_type=refresh_token), the rotated refresh + new access token are
 *     atomically re-stored (Phase 2a), and the fresh access token is returned;
 *   - no stored token, or a refresh that fails (revoked/expired), yields a
 *     consent-required signal so the panel restarts the authorize flow --
 *     never a silent DevAgentToken fallback, which would mask the real path.
 *
 * The encryption precondition mirrors the callback controller: with
 * database_encryption OFF we refuse to obtain-and-store a rotated refresh
 * token (fail-safe) rather than persist a long-lived secret in plaintext.
 *
 * No token or secret value is ever logged here; only outcomes are returned.
 *
 * @package   OpenEMR\Modules\ClinicalCopilot
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Auth;

use OpenEMR\Common\Crypto\CryptoGenException;

final class AgentTokenBroker
{
    /**
     * Treat a token expiring within this margin as already stale, so the agent
     * never receives one that lapses mid-request.
     */
    private const EXPIRY_SKEW_SECONDS = 60;

    public function __construct(
        private readonly OAuthConsentConfig $config,
        private readonly AuthorizationCodeExchanger $exchanger,
        private readonly UserOAuthTokenRepository $repository,
        private readonly bool $databaseEncryptionEnabled,
    ) {
    }

    /**
     * Resolve the bearer for the logged-in user. The DevAgentToken arguments are
     * only used on the flag-off path; they are supplied by the controller from
     * the authenticated session.
     */
    public function broker(
        int $authUserId,
        string $username,
        int $pid,
        string $signingKey,
        int $issuedAt,
        int $ttlSeconds,
    ): BrokerResult {
        if (!$this->config->enabled) {
            return BrokerResult::token(
                DevAgentToken::mint($authUserId, $username, $pid, $signingKey, $issuedAt, $ttlSeconds),
            );
        }

        return $this->brokerRealToken($authUserId, $issuedAt);
    }

    private function brokerRealToken(int $authUserId, int $now): BrokerResult
    {
        $stored = $this->repository->findByUser($authUserId);
        if ($stored === null) {
            return BrokerResult::consentRequired();
        }

        if ($this->isUsable($stored, $now)) {
            return BrokerResult::token((string) $stored->accessToken);
        }

        // Expired/near-expiry: enforce the encryption precondition BEFORE we
        // obtain a rotated refresh token we could not store safely.
        if (!$this->databaseEncryptionEnabled) {
            return BrokerResult::error();
        }

        try {
            $refreshed = $this->exchanger->refresh($stored->refreshToken);
        } catch (OAuthExchangeException) {
            // Refresh token revoked/expired: re-consent, not a 500, not a fallback.
            return BrokerResult::consentRequired();
        }

        if (
            $refreshed->refreshToken === ''
            || $refreshed->accessToken === null
            || $refreshed->accessToken === ''
        ) {
            return BrokerResult::consentRequired();
        }

        try {
            // Atomic replace = refresh-token rotation: the NEW refresh token is
            // persisted, the old one discarded.
            $this->repository->upsert(
                $authUserId,
                $refreshed->refreshToken,
                $refreshed->accessToken,
                $refreshed->accessTokenExpiresAt,
            );
        } catch (\RuntimeException | CryptoGenException) {
            return BrokerResult::error();
        }

        return BrokerResult::token($refreshed->accessToken);
    }

    private function isUsable(UserOAuthToken $token, int $now): bool
    {
        if ($token->accessToken === null || $token->accessToken === '') {
            return false;
        }

        if ($token->accessTokenExpiresAt === null) {
            return false;
        }

        return $token->accessTokenExpiresAt->getTimestamp() > $now + self::EXPIRY_SKEW_SECONDS;
    }
}
