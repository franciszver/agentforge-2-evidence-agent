<?php

/**
 * Tokens returned by a successful authorization_code exchange.
 *
 * Plaintext value object handed from the exchanger to the callback controller,
 * which immediately hands it to the repository for encrypted-at-rest storage.
 *
 * @package   OpenEMR\Modules\ClinicalCopilot
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Auth;

final readonly class OAuthTokenResponse
{
    public function __construct(
        public string $refreshToken,
        public ?string $accessToken,
        public ?\DateTimeImmutable $accessTokenExpiresAt,
    ) {
    }
}
