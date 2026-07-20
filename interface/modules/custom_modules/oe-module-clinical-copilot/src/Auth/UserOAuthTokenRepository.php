<?php

/**
 * Per-user OAuth token storage for the Clinical Co-Pilot module.
 *
 * Persists a single OAuth token record per OpenEMR user, encrypting the token
 * columns at rest with CryptoGen (the same encryptForDatabase/decryptFromDatabase
 * pattern OpenEMR uses for oauth_clients.client_secret in
 * src/Common/Auth/OpenIDConnect/Repositories/ClientRepository.php). Refresh-token
 * rotation means each store REPLACES the row, done atomically via
 * INSERT ... ON DUPLICATE KEY UPDATE on the unique openemr_user_id key.
 *
 * @package   OpenEMR\Modules\ClinicalCopilot
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Auth;

use OpenEMR\Common\Crypto\CryptoInterface;

final class UserOAuthTokenRepository
{
    public function __construct(
        private readonly CryptoInterface $crypto,
        private readonly TokenStorageGateway $gateway,
    ) {
    }

    /**
     * Store (or replace) the token record for a user. The refresh token is always
     * encrypted; a null access token / expiry is stored as SQL NULL, not encrypted.
     */
    public function upsert(
        int $openEmrUserId,
        string $refreshToken,
        ?string $accessToken,
        ?\DateTimeImmutable $accessExpires,
    ): void {
        $refreshEncrypted = $this->crypto->encryptForDatabase($refreshToken);
        $accessEncrypted = $accessToken === null ? null : $this->crypto->encryptForDatabase($accessToken);
        $expires = $accessExpires?->format('Y-m-d H:i:s');

        $sql = "INSERT INTO `clinical_copilot_user_oauth_token` "
            . "(`openemr_user_id`, `refresh_token_encrypted`, `access_token_encrypted`, `access_token_expires_at`) "
            . "VALUES (?, ?, ?, ?) "
            . "ON DUPLICATE KEY UPDATE "
            . "`refresh_token_encrypted` = VALUES(`refresh_token_encrypted`), "
            . "`access_token_encrypted` = VALUES(`access_token_encrypted`), "
            . "`access_token_expires_at` = VALUES(`access_token_expires_at`), "
            . "`updated` = CURRENT_TIMESTAMP";

        $this->gateway->execute($sql, [$openEmrUserId, $refreshEncrypted, $accessEncrypted, $expires]);
    }

    /**
     * Fetch and decrypt the token record for a user, or null when none is stored.
     */
    public function findByUser(int $openEmrUserId): ?UserOAuthToken
    {
        $sql = "SELECT `refresh_token_encrypted`, `access_token_encrypted`, `access_token_expires_at` "
            . "FROM `clinical_copilot_user_oauth_token` "
            . "WHERE `openemr_user_id` = ?";

        $row = $this->gateway->fetchRow($sql, [$openEmrUserId]);
        if ($row === null) {
            return null;
        }

        $refreshEncrypted = $row['refresh_token_encrypted'] ?? null;
        $refreshToken = $this->crypto->decryptFromDatabase(is_string($refreshEncrypted) ? $refreshEncrypted : null);

        $accessEncrypted = $row['access_token_encrypted'] ?? null;
        $accessToken = is_string($accessEncrypted) && $accessEncrypted !== ''
            ? $this->crypto->decryptFromDatabase($accessEncrypted)
            : null;

        $expiresRaw = $row['access_token_expires_at'] ?? null;
        $accessExpires = is_string($expiresRaw) && $expiresRaw !== ''
            ? new \DateTimeImmutable($expiresRaw)
            : null;

        return new UserOAuthToken($openEmrUserId, $refreshToken, $accessToken, $accessExpires);
    }

    /**
     * Remove any stored token record for a user.
     */
    public function deleteByUser(int $openEmrUserId): void
    {
        $this->gateway->execute(
            "DELETE FROM `clinical_copilot_user_oauth_token` WHERE `openemr_user_id` = ?",
            [$openEmrUserId],
        );
    }
}
