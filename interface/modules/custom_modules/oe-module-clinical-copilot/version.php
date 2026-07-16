<?php

/**
 * Clinical Co-Pilot module SQL versioning flat file.
 *
 * Read by the OpenEMR module installer (InstallerController::getModuleVersion-
 * FromFile / makeButtonForSqlAction) to decide whether a versioned upgrade in
 * sql/ needs to run for an already-installed module. Bumped to 1.0.1 to ship
 * the per-user OAuth token table to installs that predate table.sql (#124
 * Phase 2b); the matching upgrade file is sql/1_0_0-to-1_0_1_upgrade.sql.
 *
 * @package   OpenEMR\Modules\ClinicalCopilot
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

$v_major = '1';
$v_minor = '0';
$v_patch = '1';
