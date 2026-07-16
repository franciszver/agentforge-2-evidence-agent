--
-- Clinical Co-Pilot module schema.
--
-- Run once by the module installer (OpenEMR\Services\Utils\SQLUpgradeService via
-- InstModuleTable::installSQLWithUpgradeService) when the module is installed.
-- Guarded with #IfNotTable so it is idempotent and safe to re-run.
--

#IfNotTable clinical_copilot_user_oauth_token
CREATE TABLE `clinical_copilot_user_oauth_token` (
    `id` INT(11) UNSIGNED NOT NULL AUTO_INCREMENT,
    `openemr_user_id` INT(11) NOT NULL COMMENT 'Conceptual FK to users.id (owning clinician)',
    `refresh_token_encrypted` MEDIUMTEXT NOT NULL COMMENT 'CryptoGen-encrypted OAuth refresh token',
    `access_token_encrypted` MEDIUMTEXT DEFAULT NULL COMMENT 'CryptoGen-encrypted OAuth access token',
    `access_token_expires_at` DATETIME DEFAULT NULL COMMENT 'Access token expiry (UTC)',
    `created` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `openemr_user_id` (`openemr_user_id`)
) ENGINE=InnoDB COMMENT='Clinical Co-Pilot per-user OAuth tokens (encrypted at rest)';
#EndIf
