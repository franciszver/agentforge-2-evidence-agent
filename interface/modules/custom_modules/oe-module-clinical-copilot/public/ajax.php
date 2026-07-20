<?php

/**
 * Clinical Co-Pilot Module - Token Broker AJAX Handler
 *
 * Entry point for the panel's token broker. globals.php restores the OpenEMR
 * session; the controller verifies the CSRF token on every request before
 * issuing the panel a bearer token + agent URL (see TokenBrokerController).
 *
 * The which-bearer decision is made by AgentTokenBroker (#124 Phase 3): its
 * confidential collaborators -- the Phase 2b refresh exchanger, the Phase 2a
 * token store, the consent flag, and the encryption precondition -- are wired
 * here from server configuration only, never from request input.
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
use OpenEMR\Modules\ClinicalCopilot\Auth\AgentTokenBroker;
use OpenEMR\Modules\ClinicalCopilot\Auth\GuzzleAuthorizationCodeExchanger;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthConsentConfig;
use OpenEMR\Modules\ClinicalCopilot\Auth\QueryUtilsTokenStorageGateway;
use OpenEMR\Modules\ClinicalCopilot\Auth\UserOAuthTokenRepository;
use OpenEMR\Modules\ClinicalCopilot\Controller\TokenBrokerController;

$config = OAuthConsentConfig::fromEnvironment();
$globals = OEGlobalsBag::getInstance();

// TLS verification for the server-side refresh call. Secure by default; a dev
// stack with a self-signed cert opts out via clinical_copilot_oauth_verify_ssl=0.
$verifySslRaw = $globals->get('clinical_copilot_oauth_verify_ssl');
$verifySsl = ($verifySslRaw === null || $verifySslRaw === '')
    ? true
    : $globals->getBoolean('clinical_copilot_oauth_verify_ssl');

$repository = new UserOAuthTokenRepository(
    ServiceContainer::getCrypto(),
    new QueryUtilsTokenStorageGateway(),
);

$broker = new AgentTokenBroker(
    $config,
    new GuzzleAuthorizationCodeExchanger($config, $verifySsl),
    $repository,
    $globals->getBoolean('database_encryption'),
);

$controller = new TokenBrokerController($broker);
$controller->handleRequest();
