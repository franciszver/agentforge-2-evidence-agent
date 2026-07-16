<?php

/**
 * Seam for the server-side authorization_code -> token exchange.
 *
 * Kept behind an interface so the callback controller's security logic (state
 * validation, encryption precondition, storage) is unit-testable without a
 * running OAuth server. The production implementation
 * (GuzzleAuthorizationCodeExchanger) POSTs to the token endpoint with the
 * confidential client's credentials and the stored code_verifier.
 *
 * @package   OpenEMR\Modules\ClinicalCopilot
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Auth;

interface AuthorizationCodeExchanger
{
    /**
     * Exchange an authorization code (plus the server-stored PKCE verifier) for
     * tokens. Throws OAuthExchangeException on any failure; the message is
     * log-only and must never reach the browser.
     */
    public function exchange(string $code, string $codeVerifier): OAuthTokenResponse;

    /**
     * Redeem a stored refresh token (grant_type=refresh_token) for a fresh
     * access token and a ROTATED refresh token. Throws OAuthExchangeException on
     * any failure (e.g. a revoked/expired refresh token); the message is
     * log-only and must never reach the browser.
     */
    public function refresh(string $refreshToken): OAuthTokenResponse;
}
