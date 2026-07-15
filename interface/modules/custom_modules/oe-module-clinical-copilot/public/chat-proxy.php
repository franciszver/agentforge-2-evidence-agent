<?php

/**
 * Clinical Co-Pilot Module - Chat Proxy SSE Endpoint
 *
 * Entry point for the panel's same-origin chat bridge. globals.php restores
 * the OpenEMR session; the controller verifies auth + CSRF on every request
 * before streaming the agent's POST /chat SSE response through (see
 * ChatProxyController).
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

require_once(__DIR__ . "/../../../../globals.php");

use OpenEMR\Modules\ClinicalCopilot\Controller\ChatProxyController;

$controller = new ChatProxyController();
$controller->handleRequest();
