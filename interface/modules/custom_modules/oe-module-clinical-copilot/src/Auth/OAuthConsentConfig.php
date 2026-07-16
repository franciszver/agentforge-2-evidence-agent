<?php

/**
 * Resolved, server-side configuration for the OAuth consent flow (#124 Phase 2b).
 *
 * Every value here comes from server configuration -- OpenEMR globals, module
 * constants, and the site's own OAuth server URLs -- never from request input,
 * so the browser cannot influence the client_id, redirect_uri, scope or
 * audience of the authorize request or the token exchange.
 *
 * The redirect_uri is a fixed constant that MUST stay byte-for-byte identical
 * to Phase 1's `copilot_prod_client_redirect_uri` (services/copilot-agent/
 * app/config.py): OpenEMR enforces exact redirect_uri matching at both authorize
 * and token time. The scope mirrors Phase 1's `copilot_prod_client_scopes`.
 *
 * @package   OpenEMR\Modules\ClinicalCopilot
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Auth;

use OpenEMR\Core\OEGlobalsBag;
use OpenEMR\FHIR\Config\ServerConfig;

final readonly class OAuthConsentConfig
{
    /**
     * Browser-facing callback URL. Single source of truth on the PHP side;
     * kept byte-for-byte identical to Phase 1's config.py constant.
     */
    public const CANONICAL_REDIRECT_URI =
        'https://localhost:9300/interface/modules/custom_modules/'
        . 'oe-module-clinical-copilot/public/oauth-callback.php';

    /**
     * SMART-on-FHIR scopes, mirroring Phase 1's copilot_prod_client_scopes.
     * ScopeRepository::finalizeScopes only grants scopes the client REGISTERED
     * with, so this must be a subset of the registered set.
     */
    public const SCOPE =
        'openid offline_access launch launch/patient api:oemr api:fhir fhirUser '
        . 'user/patient.read user/medication.read user/allergy.read '
        . 'user/medical_problem.read user/encounter.read user/appointment.read '
        . 'user/vital.read user/procedure.read user/Observation.read';

    /** OpenEMR global gating the whole consent flow (default off; DevAgentToken path stays live). */
    public const ENABLED_GLOBAL = 'clinical_copilot_oauth_consent_enabled';

    /** OpenEMR globals holding the Phase 1 confidential client credentials. */
    public const CLIENT_ID_GLOBAL = 'clinical_copilot_prod_client_id';
    public const CLIENT_SECRET_GLOBAL = 'clinical_copilot_prod_client_secret';

    /**
     * OpenEMR global overriding the SERVER-SIDE token-exchange URL. The token
     * exchange (and refresh) run inside the openemr container and must reach the
     * OAuth server over the internal docker network, NOT via the browser-facing
     * public origin ($tokenUrl / site_addr_oath) -- that host (e.g. localhost:9300)
     * is a host port map apache does not listen on inside the container, so a POST
     * there fails outright. Unset => DEFAULT_INTERNAL_TOKEN_URL.
     */
    public const INTERNAL_TOKEN_URL_GLOBAL = 'clinical_copilot_oauth_internal_token_url';

    /**
     * Default container-internal token endpoint: the `openemr` docker-network
     * alias the agent already uses for server-to-server calls. Site is the
     * "default" site, matching the fixed browser-facing redirect_uri path.
     */
    public const DEFAULT_INTERNAL_TOKEN_URL = 'https://openemr/oauth2/default/token';

    public function __construct(
        public bool $enabled,
        public string $clientId,
        public string $clientSecret,
        public string $redirectUri,
        public string $scope,
        public string $authorizeUrl,
        public string $tokenUrl,
        public string $internalTokenUrl,
        public string $audience,
    ) {
    }

    /**
     * Build the config from the live OpenEMR environment. Called only at the
     * public entry points (the parse-don't-validate boundary); the controllers
     * receive the resolved value object and never touch globals themselves.
     */
    public static function fromEnvironment(): self
    {
        $globals = OEGlobalsBag::getInstance();
        $server = new ServerConfig();

        return new self(
            enabled: $globals->getBoolean(self::ENABLED_GLOBAL),
            clientId: self::readString($globals, self::CLIENT_ID_GLOBAL),
            clientSecret: self::readString($globals, self::CLIENT_SECRET_GLOBAL),
            redirectUri: self::CANONICAL_REDIRECT_URI,
            scope: self::SCOPE,
            authorizeUrl: $server->getAuthorizeUrl(),
            tokenUrl: $server->getTokenUrl(),
            internalTokenUrl: self::resolveInternalTokenUrl($globals),
            // SMART `aud`: OpenEMR forces this to the FHIR/API resource base (not
            // the token endpoint) whenever a launch token is present. This is the
            // browser-facing FHIR base, so it stays on the public origin.
            audience: $server->getFhirUrl(),
        );
    }

    /**
     * Server-side token endpoint reachable from inside the openemr container.
     * Honours the override global; falls back to the internal docker alias
     * default. The browser-facing $tokenUrl (used for the authorize `aud`) is
     * deliberately left unchanged -- only the server-to-server POST target moves.
     */
    private static function resolveInternalTokenUrl(OEGlobalsBag $globals): string
    {
        $override = self::readString($globals, self::INTERNAL_TOKEN_URL_GLOBAL);

        return $override !== '' ? $override : self::DEFAULT_INTERNAL_TOKEN_URL;
    }

    private static function readString(OEGlobalsBag $globals, string $key): string
    {
        $value = $globals->get($key);

        return is_string($value) ? $value : '';
    }
}
