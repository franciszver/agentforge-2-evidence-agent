<?php

/**
 * Clinical Co-Pilot Module - Token Broker AJAX Handler
 *
 * Entry point for the panel's token broker. globals.php restores the OpenEMR
 * session; the controller verifies the CSRF token on every request before
 * issuing the panel a bearer token + agent URL (see TokenBrokerController).
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

require_once(__DIR__ . "/../../../../globals.php");

use OpenEMR\Modules\ClinicalCopilot\Controller\TokenBrokerController;

$controller = new TokenBrokerController();
$controller->handleRequest();
