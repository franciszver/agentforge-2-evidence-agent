<?php

/**
 * Isolated tests for the Clinical Co-Pilot OAuth authorize-redirect controller
 * (#124 Phase 2b).
 *
 * These assert the security-critical shape of the /oauth2/default/authorize
 * URL without a browser or a running OpenEMR: every required parameter is
 * present, the redirect_uri matches Phase 1's canonical constant byte-for-byte,
 * PKCE is S256 with the challenge derived from a per-request code_verifier that
 * is stored SERVER-SIDE in the session (never placed in the URL), and the
 * state is CSRF-bound to the session via CsrfUtils (the same mechanism the rest
 * of this module uses). The SMART launch token is produced through an injected
 * factory seam so the controller stays unit-testable (the real token needs
 * CryptoGen + the patient UUID from the DB).
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Tests\Isolated\Modules\ClinicalCopilot;

use OpenEMR\Common\Csrf\CsrfUtils;
use OpenEMR\Common\Session\SessionWrapperFactory;
use OpenEMR\Core\ModulesClassLoader;
use OpenEMR\Modules\ClinicalCopilot\Auth\LaunchTokenFactory;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthConsentConfig;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthConsentSession;
use OpenEMR\Modules\ClinicalCopilot\Auth\PkcePair;
use OpenEMR\Modules\ClinicalCopilot\Controller\AuthorizeRedirectController;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Symfony\Component\HttpFoundation\Session\SessionInterface;

class AuthorizeRedirectControllerTest extends TestCase
{
    /** The single source of truth for the browser-facing redirect_uri (Phase 1 config.py). */
    private const CANONICAL_REDIRECT_URI =
        'https://localhost:9300/interface/modules/custom_modules/'
        . 'oe-module-clinical-copilot/public/oauth-callback.php';

    private const CLIENT_ID = 'test-client-id-0000';
    private const SCOPE = 'openid offline_access launch launch/patient api:oemr api:fhir';
    private const AUTHORIZE_URL = 'https://localhost:9300/oauth2/default/authorize';
    private const TOKEN_URL = 'https://localhost:9300/oauth2/default/token';

    protected function setUp(): void
    {
        $projectDir = dirname(__DIR__, 5);
        $classLoader = new ModulesClassLoader($projectDir);
        $classLoader->registerNamespaceIfNotExists(
            'OpenEMR\\Modules\\ClinicalCopilot\\',
            $projectDir . '/interface/modules/custom_modules/oe-module-clinical-copilot/src'
        );

        $this->resetSessionWrapperFactorySingleton();
    }

    protected function tearDown(): void
    {
        $this->resetSessionWrapperFactorySingleton();
    }

    #[Test]
    public function redirectUriConstantMatchesPhase1Canonical(): void
    {
        // Lock the module constant against Phase 1's config.py value byte-for-byte
        // by reading config.py itself -- OpenEMR enforces exact redirect_uri
        // matching across the PHP and Python sides, and reading the real source of
        // truth is both meaningful and non-tautological (a self-comparison of two
        // known literals is a tautology PHPStan rejects).
        $projectDir = dirname(__DIR__, 5);
        $configPy = (string) file_get_contents($projectDir . '/services/copilot-agent/app/config.py');
        $this->assertNotSame('', $configPy, 'Phase 1 config.py must be readable');

        $matched = preg_match(
            '/copilot_prod_client_redirect_uri:\s*str\s*=\s*\(([^)]*)\)/s',
            $configPy,
            $block
        );
        $this->assertSame(1, $matched, 'config.py must define copilot_prod_client_redirect_uri');

        // The value spans adjacent "..." string literals; concatenate them.
        preg_match_all('/"([^"]*)"/', $block[1], $pieces);
        $expected = implode('', $pieces[1]);

        $this->assertSame($expected, OAuthConsentConfig::CANONICAL_REDIRECT_URI);
    }

    #[Test]
    public function authorizeUrlCarriesEveryRequiredParameter(): void
    {
        $session = $this->makeSession(authUserId: 5);
        $controller = new AuthorizeRedirectController($this->config(), $this->launchFactory('LAUNCH-STUB'));

        $params = $this->queryParamsOf($controller->buildAuthorizeUrl($session, 42));

        $this->assertSame('code', $params['response_type']);
        $this->assertSame(self::CLIENT_ID, $params['client_id']);
        $this->assertSame(self::CANONICAL_REDIRECT_URI, $params['redirect_uri']);
        $this->assertSame(self::SCOPE, $params['scope']);
        $this->assertSame('S256', $params['code_challenge_method']);
        $this->assertSame(self::TOKEN_URL, $params['aud']);
        $this->assertArrayHasKey('code_challenge', $params);
        $this->assertArrayHasKey('state', $params);
        $this->assertArrayHasKey('launch', $params);
    }

    #[Test]
    public function stateIsCsrfBoundToTheSession(): void
    {
        $session = $this->makeSession(authUserId: 5);
        $controller = new AuthorizeRedirectController($this->config(), $this->launchFactory('LAUNCH-STUB'));

        $params = $this->queryParamsOf($controller->buildAuthorizeUrl($session, 42));

        // The state must equal the CsrfUtils token for this session + subject,
        // so the callback can verify it in constant time and reject forgeries.
        $this->assertSame(
            CsrfUtils::collectCsrfToken($session, OAuthConsentSession::STATE_SUBJECT),
            $params['state']
        );
    }

    #[Test]
    public function codeVerifierIsStoredServerSideAndNeverLeavesInTheUrl(): void
    {
        $session = $this->makeSession(authUserId: 5);
        $controller = new AuthorizeRedirectController($this->config(), $this->launchFactory('LAUNCH-STUB'));

        $url = $controller->buildAuthorizeUrl($session, 42);
        $params = $this->queryParamsOf($url);

        $verifier = $session->get(OAuthConsentSession::CODE_VERIFIER_KEY);
        $this->assertIsString($verifier);
        $this->assertNotSame('', $verifier);

        // The challenge on the wire is S256(verifier); the verifier itself is
        // never transmitted (no 'code_verifier' param, and its literal value
        // appears nowhere in the URL).
        $this->assertSame(PkcePair::challengeFor($verifier), $params['code_challenge']);
        $this->assertArrayNotHasKey('code_verifier', $params);
        $this->assertStringNotContainsString($verifier, $url);
    }

    #[Test]
    public function launchTokenComesFromTheInjectedFactory(): void
    {
        $session = $this->makeSession(authUserId: 5);
        $controller = new AuthorizeRedirectController($this->config(), $this->launchFactory('LAUNCH-STUB'));

        $params = $this->queryParamsOf($controller->buildAuthorizeUrl($session, 42));

        $this->assertSame('LAUNCH-STUB', $params['launch']);
    }

    #[Test]
    public function disabledFlagShortCircuitsHandleRequestWithoutRedirect(): void
    {
        $session = $this->makeSession(authUserId: 5);
        $session->set('pid', 42);
        SessionWrapperFactory::getInstance()->setActiveSession($session);

        $controller = new AuthorizeRedirectController(
            $this->config(enabled: false),
            $this->launchFactory('LAUNCH-STUB')
        );

        ob_start();
        $controller->handleRequest();
        ob_get_clean();

        $this->assertSame(404, http_response_code());
        // No verifier is minted or stored when the flow is off.
        $this->assertNull($session->get(OAuthConsentSession::CODE_VERIFIER_KEY));
    }

    private function config(bool $enabled = true): OAuthConsentConfig
    {
        return new OAuthConsentConfig(
            enabled: $enabled,
            clientId: self::CLIENT_ID,
            clientSecret: 'unused-here',
            redirectUri: self::CANONICAL_REDIRECT_URI,
            scope: self::SCOPE,
            authorizeUrl: self::AUTHORIZE_URL,
            tokenUrl: self::TOKEN_URL,
        );
    }

    private function launchFactory(string $token): LaunchTokenFactory
    {
        $factory = $this->createStub(LaunchTokenFactory::class);
        $factory->method('create')->willReturn($token);

        return $factory;
    }

    /**
     * @return array<string, string>
     */
    private function queryParamsOf(string $url): array
    {
        $this->assertStringStartsWith(self::AUTHORIZE_URL . '?', $url);
        parse_str((string) parse_url($url, PHP_URL_QUERY), $params);

        /** @var array<string, string> $params */
        return $params;
    }

    private function makeSession(int $authUserId): SessionInterface
    {
        $store = ['authUserID' => $authUserId];
        $session = $this->createStub(SessionInterface::class);
        $session->method('set')
            ->willReturnCallback(function (string $key, mixed $value) use (&$store): void {
                $store[$key] = $value;
            });
        $session->method('get')
            ->willReturnCallback(function (string $key, mixed $default = null) use (&$store): mixed {
                return $store[$key] ?? $default;
            });
        $session->method('remove')
            ->willReturnCallback(function (string $key) use (&$store): mixed {
                $value = $store[$key] ?? null;
                unset($store[$key]);

                return $value;
            });
        CsrfUtils::setupCsrfKey($session);

        // The controller writes the PKCE verifier via SessionUtil, which routes
        // through the factory's active session -- so this stub must be active.
        SessionWrapperFactory::getInstance()->setActiveSession($session);

        return $session;
    }

    private function resetSessionWrapperFactorySingleton(): void
    {
        $reflection = new \ReflectionClass(SessionWrapperFactory::class);
        $instancesProperty = $reflection->getProperty('instances');
        $instancesProperty->setValue(null, []);
    }
}
