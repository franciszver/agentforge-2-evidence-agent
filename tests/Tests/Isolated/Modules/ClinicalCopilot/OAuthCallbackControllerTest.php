<?php

/**
 * Isolated tests for the Clinical Co-Pilot OAuth callback controller
 * (#124 Phase 2b) -- the security-critical half of the consent flow.
 *
 * Covered without a browser or DB (the token exchange and the token store are
 * injected seams, mocked here):
 *  - the feature flag: when off, the callback does nothing (DevAgentToken path
 *    is left entirely alone);
 *  - state CSRF: a missing or mismatched `state` is rejected in constant time
 *    (via CsrfUtils) before any code is exchanged;
 *  - the server-side `code_verifier` must be present in the session or the
 *    callback fails safe (it is never accepted from the request);
 *  - the `database_encryption` precondition: with it off the controller REFUSES
 *    to store, rather than silently persisting a plaintext refresh token;
 *  - the empty-refresh-token guard;
 *  - the happy path stores refresh+access ENCRYPTED via the Phase 2a repository
 *    with replace/rotation semantics, and burns the one-time verifier;
 *  - every failure is fail-safe: a generic response, never the exception detail.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Tests\Isolated\Modules\ClinicalCopilot;

use OpenEMR\Common\Crypto\CryptoInterface;
use OpenEMR\Common\Csrf\CsrfUtils;
use OpenEMR\Common\Session\SessionWrapperFactory;
use OpenEMR\Core\ModulesClassLoader;
use OpenEMR\Modules\ClinicalCopilot\Auth\AuthorizationCodeExchanger;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthConsentConfig;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthConsentSession;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthExchangeException;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthTokenResponse;
use OpenEMR\Modules\ClinicalCopilot\Auth\TokenStorageGateway;
use OpenEMR\Modules\ClinicalCopilot\Auth\UserOAuthTokenRepository;
use OpenEMR\Modules\ClinicalCopilot\Controller\OAuthCallbackController;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Symfony\Component\HttpFoundation\Session\SessionInterface;

class OAuthCallbackControllerTest extends TestCase
{
    private const AUTH_USER_ID = 5;
    private const SECRET_LEAK_MARKER = 'super-secret-internal-detail';

    /** @var list<string|int|null>|null */
    private ?array $capturedBinds = null;
    private ?string $capturedSql = null;

    protected function setUp(): void
    {
        $projectDir = dirname(__DIR__, 5);
        $classLoader = new ModulesClassLoader($projectDir);
        $classLoader->registerNamespaceIfNotExists(
            'OpenEMR\\Modules\\ClinicalCopilot\\',
            $projectDir . '/interface/modules/custom_modules/oe-module-clinical-copilot/src'
        );

        $this->capturedBinds = null;
        $this->capturedSql = null;
        $this->resetSessionWrapperFactorySingleton();
    }

    protected function tearDown(): void
    {
        $this->resetSessionWrapperFactorySingleton();
    }

    #[Test]
    public function disabledFlagDoesNothingAndNeverExchanges(): void
    {
        $session = $this->activeSession(withVerifier: true);
        $state = CsrfUtils::collectCsrfToken($session, OAuthConsentSession::STATE_SUBJECT);

        [$status, $body] = $this->invoke(
            exchanger: $this->exchangerNeverCalled(),
            gateway: $this->gatewayNeverCalled(),
            enabled: false,
            code: 'good-code',
            state: $state,
        );

        $this->assertSame(404, $status);
        $this->assertStringNotContainsString(OAuthCallbackController::SUCCESS_MARKER, $body);
    }

    #[Test]
    public function nonGetMethodIsRejected(): void
    {
        $this->activeSession(withVerifier: true);

        [$status] = $this->invoke(
            exchanger: $this->exchangerNeverCalled(),
            gateway: $this->gatewayNeverCalled(),
            method: 'POST',
            code: 'good-code',
            state: 'anything',
        );

        $this->assertSame(405, $status);
    }

    #[Test]
    public function unauthenticatedSessionIsRejected(): void
    {
        $this->activeSession(withVerifier: true, authUserId: 0);

        [$status] = $this->invoke(
            exchanger: $this->exchangerNeverCalled(),
            gateway: $this->gatewayNeverCalled(),
            code: 'good-code',
            state: 'anything',
        );

        $this->assertSame(401, $status);
    }

    #[Test]
    public function missingStateIsRejectedBeforeExchange(): void
    {
        $this->activeSession(withVerifier: true);

        [$status, $body] = $this->invoke(
            exchanger: $this->exchangerNeverCalled(),
            gateway: $this->gatewayNeverCalled(),
            code: 'good-code',
            state: null,
        );

        $this->assertSame(403, $status);
        $this->assertStringNotContainsString(OAuthCallbackController::SUCCESS_MARKER, $body);
    }

    #[Test]
    public function mismatchedStateIsRejectedBeforeExchange(): void
    {
        $session = $this->activeSession(withVerifier: true);
        $valid = CsrfUtils::collectCsrfToken($session, OAuthConsentSession::STATE_SUBJECT);

        [$status] = $this->invoke(
            exchanger: $this->exchangerNeverCalled(),
            gateway: $this->gatewayNeverCalled(),
            code: 'good-code',
            state: $valid . 'tampered',
        );

        $this->assertSame(403, $status);
    }

    #[Test]
    public function missingCodeIsRejected(): void
    {
        $session = $this->activeSession(withVerifier: true);
        $state = CsrfUtils::collectCsrfToken($session, OAuthConsentSession::STATE_SUBJECT);

        [$status] = $this->invoke(
            exchanger: $this->exchangerNeverCalled(),
            gateway: $this->gatewayNeverCalled(),
            code: '',
            state: $state,
        );

        $this->assertSame(400, $status);
    }

    #[Test]
    public function missingServerSideVerifierFailsSafe(): void
    {
        // Valid state but no code_verifier in the session (e.g. session rotated):
        // the verifier is never taken from the request, so the flow must abort.
        $session = $this->activeSession(withVerifier: false);
        $state = CsrfUtils::collectCsrfToken($session, OAuthConsentSession::STATE_SUBJECT);

        [$status] = $this->invoke(
            exchanger: $this->exchangerNeverCalled(),
            gateway: $this->gatewayNeverCalled(),
            code: 'good-code',
            state: $state,
        );

        $this->assertSame(400, $status);
    }

    #[Test]
    public function encryptionOffRefusesToStoreAndNeverExchanges(): void
    {
        $session = $this->activeSession(withVerifier: true);
        $state = CsrfUtils::collectCsrfToken($session, OAuthConsentSession::STATE_SUBJECT);

        [$status, $body] = $this->invoke(
            exchanger: $this->exchangerNeverCalled(),
            gateway: $this->gatewayNeverCalled(),
            databaseEncryptionEnabled: false,
            code: 'good-code',
            state: $state,
        );

        $this->assertSame(500, $status);
        $this->assertStringNotContainsString(OAuthCallbackController::SUCCESS_MARKER, $body);
    }

    #[Test]
    public function exchangeFailureFailsSafeWithoutLeakingDetail(): void
    {
        $session = $this->activeSession(withVerifier: true);
        $state = CsrfUtils::collectCsrfToken($session, OAuthConsentSession::STATE_SUBJECT);

        $exchanger = $this->createStub(AuthorizationCodeExchanger::class);
        $exchanger->method('exchange')
            ->willThrowException(new OAuthExchangeException(self::SECRET_LEAK_MARKER));

        [$status, $body] = $this->invoke(
            exchanger: $exchanger,
            gateway: $this->gatewayNeverCalled(),
            code: 'good-code',
            state: $state,
        );

        $this->assertSame(400, $status);
        $this->assertStringNotContainsString(self::SECRET_LEAK_MARKER, $body);
        $this->assertStringNotContainsString(OAuthCallbackController::SUCCESS_MARKER, $body);
    }

    #[Test]
    public function emptyRefreshTokenIsGuardedAndNotStored(): void
    {
        $session = $this->activeSession(withVerifier: true);
        $state = CsrfUtils::collectCsrfToken($session, OAuthConsentSession::STATE_SUBJECT);

        $exchanger = $this->createStub(AuthorizationCodeExchanger::class);
        $exchanger->method('exchange')
            ->willReturn(new OAuthTokenResponse('', 'access-raw', null));

        [$status] = $this->invoke(
            exchanger: $exchanger,
            gateway: $this->gatewayNeverCalled(),
            code: 'good-code',
            state: $state,
        );

        $this->assertSame(400, $status);
    }

    #[Test]
    public function happyPathStoresEncryptedTokensAndBurnsTheVerifier(): void
    {
        $session = $this->activeSession(withVerifier: true);
        $state = CsrfUtils::collectCsrfToken($session, OAuthConsentSession::STATE_SUBJECT);

        $exchanger = $this->createStub(AuthorizationCodeExchanger::class);
        $exchanger->method('exchange')
            ->willReturn(new OAuthTokenResponse(
                'refresh-raw',
                'access-raw',
                new \DateTimeImmutable('2026-02-03 04:05:06'),
            ));

        [$status, $body] = $this->invoke(
            exchanger: $exchanger,
            gateway: $this->capturingGateway(),
            code: 'good-code',
            state: $state,
        );

        $this->assertSame(200, $status);
        $this->assertStringContainsString(OAuthCallbackController::SUCCESS_MARKER, $body);

        // Stored via the Phase 2a repo with replace/rotation semantics and the
        // refresh token ENCRYPTED (never the raw value) before it hits the bind.
        $this->assertStringContainsStringIgnoringCase('ON DUPLICATE KEY UPDATE', (string) $this->capturedSql);
        $this->assertSame(
            [self::AUTH_USER_ID, 'ENC:refresh-raw', 'ENC:access-raw', '2026-02-03 04:05:06'],
            $this->capturedBinds,
        );
        $this->assertNotContains('refresh-raw', (array) $this->capturedBinds);

        // The one-time verifier is burned so a replayed callback cannot reuse it.
        $this->assertNull($session->get(OAuthConsentSession::CODE_VERIFIER_KEY));
    }

    // --- helpers -----------------------------------------------------------

    /**
     * @return array{0: int, 1: string}
     */
    private function invoke(
        AuthorizationCodeExchanger $exchanger,
        TokenStorageGateway $gateway,
        bool $enabled = true,
        bool $databaseEncryptionEnabled = true,
        string $method = 'GET',
        ?string $code = null,
        ?string $state = null,
    ): array {
        $repository = new UserOAuthTokenRepository($this->reversibleCrypto(), $gateway);
        $controller = new OAuthCallbackController(
            $this->config($enabled),
            $exchanger,
            $repository,
            $databaseEncryptionEnabled,
            $method,
            $code,
            $state,
        );

        ob_start();
        $controller->handleRequest();
        $output = ob_get_clean();

        $status = http_response_code();

        return [is_int($status) ? $status : 0, is_string($output) ? $output : ''];
    }

    private function config(bool $enabled): OAuthConsentConfig
    {
        return new OAuthConsentConfig(
            enabled: $enabled,
            clientId: 'test-client-id',
            clientSecret: 'unused-in-controller',
            redirectUri: OAuthConsentConfig::CANONICAL_REDIRECT_URI,
            scope: 'openid offline_access',
            authorizeUrl: 'https://localhost:9300/oauth2/default/authorize',
            tokenUrl: 'https://localhost:9300/oauth2/default/token',
            internalTokenUrl: 'https://openemr/oauth2/default/token',
            audience: 'https://localhost:9300/apis/default/fhir',
        );
    }

    private function activeSession(bool $withVerifier, int $authUserId = self::AUTH_USER_ID): SessionInterface
    {
        $store = ['authUserID' => $authUserId];
        if ($withVerifier) {
            $store[OAuthConsentSession::CODE_VERIFIER_KEY] = 'stored-verifier-value';
        }

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

        SessionWrapperFactory::getInstance()->setActiveSession($session);

        return $session;
    }

    private function reversibleCrypto(): CryptoInterface
    {
        $crypto = $this->createMock(CryptoInterface::class);
        $crypto->method('encryptForDatabase')
            ->willReturnCallback(static fn(?string $v): string => 'ENC:' . (string) $v);
        $crypto->method('decryptFromDatabase')
            ->willReturnCallback(static fn(?string $v): string => (string) $v);

        return $crypto;
    }

    private function capturingGateway(): TokenStorageGateway
    {
        $gateway = $this->createMock(TokenStorageGateway::class);
        $gateway->method('execute')
            ->willReturnCallback(function (string $sql, array $binds): void {
                $this->capturedSql = $sql;
                /** @var list<string|int|null> $binds */
                $this->capturedBinds = $binds;
            });

        return $gateway;
    }

    private function gatewayNeverCalled(): TokenStorageGateway
    {
        $gateway = $this->createMock(TokenStorageGateway::class);
        $gateway->expects($this->never())->method('execute');

        return $gateway;
    }

    private function exchangerNeverCalled(): AuthorizationCodeExchanger
    {
        $exchanger = $this->createMock(AuthorizationCodeExchanger::class);
        $exchanger->expects($this->never())->method('exchange');

        return $exchanger;
    }

    private function resetSessionWrapperFactorySingleton(): void
    {
        $reflection = new \ReflectionClass(SessionWrapperFactory::class);
        $instancesProperty = $reflection->getProperty('instances');
        $instancesProperty->setValue(null, []);
    }
}
