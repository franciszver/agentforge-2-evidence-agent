<?php

/**
 * Clinical Co-Pilot Module - OAuth Authorize-Redirect Entry (#124 Phase 2b).
 *
 * Initiates the browser consent flow. globals.php restores the OpenEMR session;
 * the controller verifies the flag + auth, then 302s the browser to the
 * OpenEMR /oauth2/default/authorize endpoint with a PKCE S256 challenge, a
 * session-bound state, and a SMART launch token. Superglobals are read only
 * here at the outermost boundary and parsed into typed values immediately.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

require_once(__DIR__ . "/../../../../globals.php");

use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthConsentConfig;
use OpenEMR\Modules\ClinicalCopilot\Auth\SmartLaunchTokenFactory;
use OpenEMR\Modules\ClinicalCopilot\Controller\AuthorizeRedirectController;

$requestMethod = filter_input(INPUT_SERVER, 'REQUEST_METHOD', FILTER_UNSAFE_RAW, FILTER_REQUIRE_SCALAR);

$controller = new AuthorizeRedirectController(
    OAuthConsentConfig::fromEnvironment(),
    new SmartLaunchTokenFactory(),
    is_string($requestMethod) ? $requestMethod : 'GET',
);
$controller->handleRequest();
