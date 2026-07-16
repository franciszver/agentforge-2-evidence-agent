<?php

/**
 * PKCE (RFC 7636) verifier/challenge pair for the OAuth consent flow.
 *
 * The code_verifier is a high-entropy per-request secret that stays SERVER-SIDE
 * (stored in the OpenEMR session); only the S256 code_challenge -- the URL-safe,
 * unpadded base64 of SHA-256(verifier) -- travels on the authorize request.
 * OpenEMR's CustomAuthCodeGrant recomputes the challenge from the verifier at
 * token-exchange time, so the two must agree exactly.
 *
 * @package   OpenEMR\Modules\ClinicalCopilot
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Auth;

final readonly class PkcePair
{
    private function __construct(
        public string $verifier,
        public string $challenge,
    ) {
    }

    public static function generate(): self
    {
        // 32 random bytes -> 43-char url-safe verifier, within RFC 7636's 43-128 range.
        $verifier = self::base64UrlEncode(random_bytes(32));

        return new self($verifier, self::challengeFor($verifier));
    }

    public static function challengeFor(string $verifier): string
    {
        return self::base64UrlEncode(hash('sha256', $verifier, true));
    }

    private static function base64UrlEncode(string $binary): string
    {
        return rtrim(strtr(base64_encode($binary), '+/', '-_'), '=');
    }
}
