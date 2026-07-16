<?php

/**
 * Decrypted per-user OAuth token record.
 *
 * Immutable value object returned by UserOAuthTokenRepository::findByUser().
 * Holds the plaintext tokens (decrypted on read); the encrypted-at-rest form
 * never leaves the repository.
 *
 * @package   OpenEMR\Modules\ClinicalCopilot
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Auth;

final readonly class UserOAuthToken
{
    public function __construct(
        public int $openEmrUserId,
        public string $refreshToken,
        public ?string $accessToken,
        public ?\DateTimeImmutable $accessTokenExpiresAt,
    ) {
    }
}
