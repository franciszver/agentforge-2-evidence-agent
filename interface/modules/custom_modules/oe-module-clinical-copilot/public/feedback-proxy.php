<?php

/**
 * Clinical Co-Pilot Module - Feedback Proxy Endpoint
 *
 * Entry point for the P4.4 feedback buttons' same-origin bridge. globals.php
 * restores the OpenEMR session; the controller verifies auth + CSRF on every
 * request before forwarding the thumb/comment to the agent's POST /feedback
 * (see FeedbackProxyController).
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

require_once(__DIR__ . "/../../../../globals.php");

use OpenEMR\Modules\ClinicalCopilot\Controller\FeedbackProxyController;

$controller = new FeedbackProxyController();
$controller->handleRequest();
