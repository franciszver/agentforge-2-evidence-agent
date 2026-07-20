<?php

/**
 * Narrow persistence seam for the OAuth token repository.
 *
 * Wraps the two database operations the repository needs so the encryption
 * logic can be unit-tested in isolation (no DB) by substituting a double.
 * The production implementation is QueryUtilsTokenStorageGateway.
 *
 * @package   OpenEMR\Modules\ClinicalCopilot
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Auth;

interface TokenStorageGateway
{
    /**
     * Execute a write statement (INSERT/UPDATE/DELETE), throwing on error.
     *
     * @param list<string|int|null> $binds
     */
    public function execute(string $sql, array $binds): void;

    /**
     * Fetch the first matching row, or null when there is no match.
     *
     * @param list<string|int|null> $binds
     * @return array<mixed>|null
     */
    public function fetchRow(string $sql, array $binds): ?array;
}
