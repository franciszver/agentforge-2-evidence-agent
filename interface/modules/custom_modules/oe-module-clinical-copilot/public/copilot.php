<?php

/**
 * Clinical Co-Pilot Module - Standalone PWA Page
 *
 * Entry point for the plan's "/copilot" route (§4.7). globals.php restores
 * the OpenEMR session (no $ignoreAuth -- same gating discipline as
 * public/ajax.php and public/chat-proxy.php); the controller renders the
 * standalone chat page, reusing the P2.14 chat assets rather than forking
 * them (see CopilotStandaloneController).
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

require_once(__DIR__ . "/../../../../globals.php");

use OpenEMR\Modules\ClinicalCopilot\Controller\CopilotStandaloneController;

$controller = new CopilotStandaloneController();
echo $controller->renderPage();
