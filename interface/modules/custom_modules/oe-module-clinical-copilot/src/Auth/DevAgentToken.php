<?php

/**
 * Clinical Co-Pilot dev-only agent bearer token.
 *
 * DEV-ONLY bridge (plan §4.2): the production auth flow is the OAuth2
 * authorization_code grant, which mints a per-user OpenEMR bearer token the
 * agent validates by introspection. That flow requires a browser
 * redirect/consent and is deferred to before Phase 5. Until then the broker
 * (see TokenBrokerController) issues this compact, HMAC-signed token as a
 * stand-in identity assertion for the already-authenticated OpenEMR session.
 *
 * The token is ``base64url(payloadJson) . "." . base64url(HMAC-SHA256)``, i.e.
 * a minimal JWS-shaped structure. It carries only the logged-in user's own id
 * and username plus the panel pid it was anchored to - no PHI, no secrets. The
 * signing key never leaves the server; the token itself is a bearer credential
 * the panel legitimately holds.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Auth;

final readonly class DevAgentToken
{
    /**
     * Mint a signed dev bearer token for the logged-in user.
     *
     * @param int    $authUserId  The logged-in OpenEMR user id (session authUserID).
     * @param string $username    The logged-in username (for agent-side audit).
     * @param int    $pid         The patient the panel was opened on (context anchor).
     * @param string $signingKey  Server-side secret; never sent to the browser.
     * @param int    $issuedAt    Unix timestamp the token was issued.
     * @param int    $ttlSeconds  Lifetime in seconds; sets the exp claim.
     */
    public static function mint(
        int $authUserId,
        string $username,
        int $pid,
        string $signingKey,
        int $issuedAt,
        int $ttlSeconds
    ): string {
        $payload = [
            'sub' => $authUserId,
            'username' => $username,
            'pid' => $pid,
            'iat' => $issuedAt,
            'exp' => $issuedAt + $ttlSeconds,
            'typ' => 'copilot-dev',
        ];

        $payloadSegment = self::base64UrlEncode(json_encode($payload, JSON_THROW_ON_ERROR));
        $signature = self::base64UrlEncode(hash_hmac('sha256', $payloadSegment, $signingKey, true));

        return $payloadSegment . '.' . $signature;
    }

    private static function base64UrlEncode(string $raw): string
    {
        return rtrim(strtr(base64_encode($raw), '+/', '-_'), '=');
    }
}
