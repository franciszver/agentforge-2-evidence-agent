<?php

/**
 * Production TokenStorageGateway backed by OpenEMR's QueryUtils.
 *
 * @package   OpenEMR\Modules\ClinicalCopilot
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Auth;

use OpenEMR\Common\Database\QueryUtils;

final class QueryUtilsTokenStorageGateway implements TokenStorageGateway
{
    /**
     * @param list<string|int|null> $binds
     */
    public function execute(string $sql, array $binds): void
    {
        QueryUtils::sqlStatementThrowException($sql, $binds);
    }

    /**
     * @param list<string|int|null> $binds
     * @return array<mixed>|null
     */
    public function fetchRow(string $sql, array $binds): ?array
    {
        $row = QueryUtils::querySingleRow($sql, $binds);

        return is_array($row) && $row !== [] ? $row : null;
    }
}
