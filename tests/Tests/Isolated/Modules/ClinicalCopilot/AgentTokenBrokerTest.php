<?php

/**
 * Isolated tests for the Clinical Co-Pilot agent-token broker (#124 Phase 3).
 *
 * The broker owns the security-critical decision of WHICH bearer the panel
 * hands the agent, without a browser or DB (the refresh exchange and the token
 * store are injected seams, exercised here with a fake gateway + reversible
 * crypto):
 *  - flag OFF: mints the DevAgentToken, byte-identical to before (regression);
 *  - flag ON + a valid, unexpired stored token: returns that real access token;
 *  - flag ON + an expired token: refreshes on demand, ROTATES the stored row
 *    (the NEW refresh token is persisted, not the old), and returns the fresh
 *    access token;
 *  - flag ON + no stored token: a consent-required signal (never a silent
 *    DevAgentToken fallback);
 *  - flag ON + a failed refresh (revoked/expired): a consent-required signal
 *    (not an error, not a DevAgentToken);
 *  - flag ON + database_encryption off: refuses to refresh-and-store (fail-safe
 *    error), rather than persist a rotated refresh token in plaintext.
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
use OpenEMR\Core\ModulesClassLoader;
use OpenEMR\Modules\ClinicalCopilot\Auth\AgentTokenBroker;
use OpenEMR\Modules\ClinicalCopilot\Auth\AuthorizationCodeExchanger;
use OpenEMR\Modules\ClinicalCopilot\Auth\BrokerOutcome;
use OpenEMR\Modules\ClinicalCopilot\Auth\DevAgentToken;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthConsentConfig;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthExchangeException;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthTokenResponse;
use OpenEMR\Modules\ClinicalCopilot\Auth\TokenStorageGateway;
use OpenEMR\Modules\ClinicalCopilot\Auth\UserOAuthTokenRepository;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

class AgentTokenBrokerTest extends TestCase
{
    private const AUTH_USER_ID = 7;
    private const USERNAME = 'clinician';
    private const PID = 42;
    private const SIGNING_KEY = 'signing-key-value';
    private const NOW = 1_700_000_000;
    private const TTL = 3600;

    /** @var list<string|int|null>|null */
    private ?array $capturedBinds = null;

    protected function setUp(): void
    {
        $projectDir = dirname(__DIR__, 5);
        $classLoader = new ModulesClassLoader($projectDir);
        $classLoader->registerNamespaceIfNotExists(
            'OpenEMR\\Modules\\ClinicalCopilot\\',
            $projectDir . '/interface/modules/custom_modules/oe-module-clinical-copilot/src'
        );

        $this->capturedBinds = null;
    }

    #[Test]
    public function flagOffMintsTheDevAgentTokenUnchanged(): void
    {
        $broker = new AgentTokenBroker(
            $this->config(enabled: false),
            $this->exchangerNeverCalled(),
            $this->repository($this->gatewayNeverWrites(fetchRow: null)),
            databaseEncryptionEnabled: true,
        );

        $result = $this->broker($broker);

        $this->assertSame(BrokerOutcome::Token, $result->outcome);
        $this->assertSame(
            DevAgentToken::mint(self::AUTH_USER_ID, self::USERNAME, self::PID, self::SIGNING_KEY, self::NOW, self::TTL),
            $result->token,
        );
    }

    #[Test]
    public function flagOnWithValidUnexpiredTokenReturnsTheRealAccessToken(): void
    {
        $broker = new AgentTokenBroker(
            $this->config(enabled: true),
            $this->exchangerNeverCalled(),
            $this->repository($this->gatewayNeverWrites(fetchRow: $this->storedRow('refresh-in', 'access-in', '2099-01-01 00:00:00'))),
            databaseEncryptionEnabled: true,
        );

        $result = $this->broker($broker);

        $this->assertSame(BrokerOutcome::Token, $result->outcome);
        $this->assertSame('access-in', $result->token);
    }

    #[Test]
    public function flagOnWithExpiredTokenRefreshesAndRotates(): void
    {
        $exchanger = $this->createStub(AuthorizationCodeExchanger::class);
        $exchanger->method('refresh')->willReturn(new OAuthTokenResponse(
            'refresh-out',
            'access-out',
            new \DateTimeImmutable('2099-06-01 00:00:00'),
        ));

        $broker = new AgentTokenBroker(
            $this->config(enabled: true),
            $exchanger,
            $this->repository($this->capturingGateway($this->storedRow('refresh-in', 'access-in', '2000-01-01 00:00:00'))),
            databaseEncryptionEnabled: true,
        );

        $result = $this->broker($broker);

        // The refreshed access token is returned...
        $this->assertSame(BrokerOutcome::Token, $result->outcome);
        $this->assertSame('access-out', $result->token);

        // ...and the row was rotated: the NEW refresh token is persisted (encrypted),
        // never the old one.
        $this->assertSame(
            [self::AUTH_USER_ID, 'ENC:refresh-out', 'ENC:access-out', '2099-06-01 00:00:00'],
            $this->capturedBinds,
        );
        $this->assertNotContains('ENC:refresh-in', (array) $this->capturedBinds);
    }

    #[Test]
    public function flagOnWithNoStoredTokenSignalsConsentRequired(): void
    {
        $broker = new AgentTokenBroker(
            $this->config(enabled: true),
            $this->exchangerNeverCalled(),
            $this->repository($this->gatewayNeverWrites(fetchRow: null)),
            databaseEncryptionEnabled: true,
        );

        $result = $this->broker($broker);

        $this->assertSame(BrokerOutcome::ConsentRequired, $result->outcome);
        $this->assertNull($result->token);
    }

    #[Test]
    public function flagOnWithFailedRefreshSignalsConsentRequired(): void
    {
        $exchanger = $this->createStub(AuthorizationCodeExchanger::class);
        $exchanger->method('refresh')->willThrowException(new OAuthExchangeException('revoked'));

        $broker = new AgentTokenBroker(
            $this->config(enabled: true),
            $exchanger,
            $this->repository($this->gatewayNeverWrites(fetchRow: $this->storedRow('refresh-in', 'access-in', '2000-01-01 00:00:00'))),
            databaseEncryptionEnabled: true,
        );

        $result = $this->broker($broker);

        $this->assertSame(BrokerOutcome::ConsentRequired, $result->outcome);
        $this->assertNull($result->token);
    }

    #[Test]
    public function flagOnWithEncryptionOffRefusesToRefresh(): void
    {
        $broker = new AgentTokenBroker(
            $this->config(enabled: true),
            $this->exchangerNeverCalled(),
            $this->repository($this->gatewayNeverWrites(fetchRow: $this->storedRow('refresh-in', 'access-in', '2000-01-01 00:00:00'))),
            databaseEncryptionEnabled: false,
        );

        $result = $this->broker($broker);

        $this->assertSame(BrokerOutcome::Error, $result->outcome);
        $this->assertNull($result->token);
    }

    // --- helpers -----------------------------------------------------------

    private function broker(AgentTokenBroker $broker): \OpenEMR\Modules\ClinicalCopilot\Auth\BrokerResult
    {
        return $broker->broker(
            self::AUTH_USER_ID,
            self::USERNAME,
            self::PID,
            self::SIGNING_KEY,
            self::NOW,
            self::TTL,
        );
    }

    private function config(bool $enabled): OAuthConsentConfig
    {
        return new OAuthConsentConfig(
            enabled: $enabled,
            clientId: 'test-client-id',
            clientSecret: 'unused-in-broker',
            redirectUri: OAuthConsentConfig::CANONICAL_REDIRECT_URI,
            scope: 'openid offline_access',
            authorizeUrl: 'https://localhost:9300/oauth2/default/authorize',
            tokenUrl: 'https://localhost:9300/oauth2/default/token',
        );
    }

    private function repository(TokenStorageGateway $gateway): UserOAuthTokenRepository
    {
        return new UserOAuthTokenRepository($this->reversibleCrypto(), $gateway);
    }

    /**
     * @return array<string, string>
     */
    private function storedRow(string $refresh, string $access, string $expiresAt): array
    {
        return [
            'refresh_token_encrypted' => $refresh,
            'access_token_encrypted' => $access,
            'access_token_expires_at' => $expiresAt,
        ];
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

    /**
     * @param array<string, string>|null $fetchRow
     */
    private function capturingGateway(?array $fetchRow): TokenStorageGateway
    {
        $gateway = $this->createMock(TokenStorageGateway::class);
        $gateway->method('fetchRow')->willReturn($fetchRow);
        $gateway->method('execute')
            ->willReturnCallback(function (string $sql, array $binds): void {
                /** @var list<string|int|null> $binds */
                $this->capturedBinds = $binds;
            });

        return $gateway;
    }

    /**
     * @param array<string, string>|null $fetchRow
     */
    private function gatewayNeverWrites(?array $fetchRow): TokenStorageGateway
    {
        $gateway = $this->createMock(TokenStorageGateway::class);
        $gateway->method('fetchRow')->willReturn($fetchRow);
        $gateway->expects($this->never())->method('execute');

        return $gateway;
    }

    private function exchangerNeverCalled(): AuthorizationCodeExchanger
    {
        $exchanger = $this->createMock(AuthorizationCodeExchanger::class);
        $exchanger->expects($this->never())->method('refresh');

        return $exchanger;
    }
}
