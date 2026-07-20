<?php

/**
 * Isolated tests for the Clinical Co-Pilot authorization_code -> token exchanger
 * (#124 Phase 2b / Phase 6 fix).
 *
 * The server-side token exchange runs INSIDE the openemr container and therefore
 * must POST to a container-INTERNAL token URL (e.g. https://openemr/...), NOT the
 * browser-facing public origin (https://localhost:9300/...) that the authorize
 * redirect + redirect_uri legitimately use. The public origin is not reachable
 * from inside the container (apache listens on 443, not the host's 9300 port
 * map), so a POST there fails outright. These tests lock the exchanger onto the
 * decoupled internal URL for both the initial exchange and the refresh, while
 * leaving the browser-facing config values untouched.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Tests\Isolated\Modules\ClinicalCopilot;

use GuzzleHttp\Client;
use GuzzleHttp\Handler\MockHandler;
use GuzzleHttp\HandlerStack;
use GuzzleHttp\Psr7\Response;
use OpenEMR\Core\ModulesClassLoader;
use OpenEMR\Modules\ClinicalCopilot\Auth\GuzzleAuthorizationCodeExchanger;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthConsentConfig;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Psr\Http\Message\RequestInterface;

class GuzzleAuthorizationCodeExchangerTest extends TestCase
{
    /** Browser-facing values -- the exchanger must NOT POST here. */
    private const PUBLIC_TOKEN_URL = 'https://localhost:9300/oauth2/default/token';
    /** Container-internal token URL -- the exchanger MUST POST here. */
    private const INTERNAL_TOKEN_URL = 'https://openemr/oauth2/default/token';

    protected function setUp(): void
    {
        $projectDir = dirname(__DIR__, 5);
        $classLoader = new ModulesClassLoader($projectDir);
        $classLoader->registerNamespaceIfNotExists(
            'OpenEMR\\Modules\\ClinicalCopilot\\',
            $projectDir . '/interface/modules/custom_modules/oe-module-clinical-copilot/src'
        );
    }

    #[Test]
    public function exchangePostsToInternalTokenUrlNotThePublicOrigin(): void
    {
        [$client, $mock] = $this->mockClient([
            new Response(200, [], (string) json_encode([
                'refresh_token' => 'rt-abc',
                'access_token' => 'at-xyz',
                'expires_in' => 3600,
            ])),
        ]);

        $exchanger = new GuzzleAuthorizationCodeExchanger($this->config(), false, $client);
        $token = $exchanger->exchange('the-code', 'the-verifier');

        $this->assertSame('rt-abc', $token->refreshToken);
        $this->assertSame(self::INTERNAL_TOKEN_URL, $this->postedUri($mock));
    }

    #[Test]
    public function refreshPostsToInternalTokenUrlNotThePublicOrigin(): void
    {
        [$client, $mock] = $this->mockClient([
            new Response(200, [], (string) json_encode([
                'refresh_token' => 'rt-rotated',
                'access_token' => 'at-new',
                'expires_in' => 3600,
            ])),
        ]);

        $exchanger = new GuzzleAuthorizationCodeExchanger($this->config(), false, $client);
        $exchanger->refresh('rt-old');

        $this->assertSame(self::INTERNAL_TOKEN_URL, $this->postedUri($mock));
    }

    /**
     * @param list<Response> $responses
     * @return array{Client, MockHandler}
     */
    private function mockClient(array $responses): array
    {
        $mock = new MockHandler($responses);
        $client = new Client(['handler' => HandlerStack::create($mock)]);

        return [$client, $mock];
    }

    private function postedUri(MockHandler $mock): string
    {
        // MockHandler records the request it served; no history middleware needed.
        $request = $mock->getLastRequest();
        $this->assertInstanceOf(RequestInterface::class, $request);

        return (string) $request->getUri();
    }

    private function config(): OAuthConsentConfig
    {
        return new OAuthConsentConfig(
            enabled: true,
            clientId: 'test-client-id',
            clientSecret: 'test-client-secret',
            redirectUri: OAuthConsentConfig::CANONICAL_REDIRECT_URI,
            scope: 'openid offline_access',
            authorizeUrl: 'https://localhost:9300/oauth2/default/authorize',
            tokenUrl: self::PUBLIC_TOKEN_URL,
            internalTokenUrl: self::INTERNAL_TOKEN_URL,
            audience: 'https://localhost:9300/apis/default/fhir',
        );
    }
}
