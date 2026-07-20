<?php

/**
 * Clinical Co-Pilot Module - OAuth Callback Entry (#124 Phase 2b).
 *
 * The browser-facing redirect_uri OpenEMR sends the user back to after consent.
 * This path MUST match Phase 1's canonical redirect_uri byte-for-byte
 * (OAuthConsentConfig::CANONICAL_REDIRECT_URI). globals.php restores the OpenEMR
 * session; the controller validates state + PKCE and exchanges the code for
 * tokens server-side, storing them encrypted. Superglobals (`code`, `state`,
 * request method) are read only here and parsed into typed values immediately;
 * the confidential exchange + storage collaborators are wired from server
 * configuration, never request input.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

require_once(__DIR__ . "/../../../../globals.php");

use OpenEMR\BC\ServiceContainer;
use OpenEMR\Core\OEGlobalsBag;
use OpenEMR\Modules\ClinicalCopilot\Auth\GuzzleAuthorizationCodeExchanger;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthConsentConfig;
use OpenEMR\Modules\ClinicalCopilot\Auth\QueryUtilsTokenStorageGateway;
use OpenEMR\Modules\ClinicalCopilot\Auth\UserOAuthTokenRepository;
use OpenEMR\Modules\ClinicalCopilot\Controller\OAuthCallbackController;

$config = OAuthConsentConfig::fromEnvironment();
$globals = OEGlobalsBag::getInstance();

// TLS verification for the server-side token exchange. Secure by default; a dev
// stack with a self-signed cert opts out via clinical_copilot_oauth_verify_ssl=0.
$verifySslRaw = $globals->get('clinical_copilot_oauth_verify_ssl');
$verifySsl = ($verifySslRaw === null || $verifySslRaw === '')
    ? true
    : $globals->getBoolean('clinical_copilot_oauth_verify_ssl');

$repository = new UserOAuthTokenRepository(
    ServiceContainer::getCrypto(),
    new QueryUtilsTokenStorageGateway(),
);

$requestMethod = filter_input(INPUT_SERVER, 'REQUEST_METHOD', FILTER_UNSAFE_RAW, FILTER_REQUIRE_SCALAR);
$code = filter_input(INPUT_GET, 'code', FILTER_UNSAFE_RAW, FILTER_REQUIRE_SCALAR);
$state = filter_input(INPUT_GET, 'state', FILTER_UNSAFE_RAW, FILTER_REQUIRE_SCALAR);

$controller = new OAuthCallbackController(
    $config,
    new GuzzleAuthorizationCodeExchanger($config, $verifySsl),
    $repository,
    $globals->getBoolean('database_encryption'),
    is_string($requestMethod) ? $requestMethod : 'GET',
    is_string($code) ? $code : null,
    is_string($state) ? $state : null,
);
$controller->handleRequest();
