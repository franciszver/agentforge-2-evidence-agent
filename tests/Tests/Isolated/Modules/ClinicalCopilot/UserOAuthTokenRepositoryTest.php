<?php

/**
 * Isolated tests for the Clinical Co-Pilot per-user OAuth token repository.
 *
 * Verifies encryption-at-rest behaviour without a database: the CryptoInterface
 * and the storage gateway are mocked so we can assert that raw tokens are passed
 * through encryption before reaching the SQL bind values, that reads decrypt, and
 * that upsert/find/delete build the expected replace-semantics queries. The live
 * DB round-trip (real CryptoGen + real table) is exercised separately against the
 * running stack, same discipline as the other ClinicalCopilot isolated tests.
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
use OpenEMR\Modules\ClinicalCopilot\Auth\TokenStorageGateway;
use OpenEMR\Modules\ClinicalCopilot\Auth\UserOAuthToken;
use OpenEMR\Modules\ClinicalCopilot\Auth\UserOAuthTokenRepository;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

class UserOAuthTokenRepositoryTest extends TestCase
{
    protected function setUp(): void
    {
        $projectDir = dirname(__DIR__, 5);
        $classLoader = new ModulesClassLoader($projectDir);
        $classLoader->registerNamespaceIfNotExists(
            'OpenEMR\\Modules\\ClinicalCopilot\\',
            $projectDir . '/interface/modules/custom_modules/oe-module-clinical-copilot/src'
        );
    }

    /**
     * A CryptoInterface double that reversibly wraps values, so tests can assert
     * that a bound column value is the encrypted form and never the raw token.
     */
    private function reversibleCrypto(): CryptoInterface
    {
        $crypto = $this->createMock(CryptoInterface::class);
        $crypto->method('encryptForDatabase')
            ->willReturnCallback(static fn(?string $v): string => 'ENC:' . (string) $v);
        $crypto->method('decryptFromDatabase')
            ->willReturnCallback(static fn(?string $v): string => str_starts_with((string) $v, 'ENC:')
                ? substr((string) $v, 4)
                : (string) $v);

        return $crypto;
    }

    #[Test]
    public function upsertEncryptsTokensBeforeBindingAndUsesReplaceSemantics(): void
    {
        $capturedSql = null;
        $capturedBinds = null;

        $gateway = $this->createMock(TokenStorageGateway::class);
        $gateway->expects($this->once())
            ->method('execute')
            ->willReturnCallback(function (string $sql, array $binds) use (&$capturedSql, &$capturedBinds): void {
                $capturedSql = $sql;
                $capturedBinds = $binds;
            });

        $repo = new UserOAuthTokenRepository($this->reversibleCrypto(), $gateway);
        $repo->upsert(7, 'refresh-raw', 'access-raw', new \DateTimeImmutable('2026-01-02 03:04:05'));

        // Replace-semantics: a single atomic statement that overwrites on the unique user key.
        $this->assertStringContainsStringIgnoringCase('INSERT INTO', (string) $capturedSql);
        $this->assertStringContainsStringIgnoringCase('ON DUPLICATE KEY UPDATE', (string) $capturedSql);

        // Encryption applied before the bind: encrypted forms present, raw tokens absent.
        $this->assertSame(
            [7, 'ENC:refresh-raw', 'ENC:access-raw', '2026-01-02 03:04:05'],
            $capturedBinds
        );
        $this->assertNotContains('refresh-raw', (array) $capturedBinds);
        $this->assertNotContains('access-raw', (array) $capturedBinds);
    }

    #[Test]
    public function upsertBindsNullAccessTokenWithoutEncryptingNull(): void
    {
        $capturedBinds = null;

        $crypto = $this->createMock(CryptoInterface::class);
        // Only the refresh token should be encrypted; a null access token stays null.
        $crypto->expects($this->once())
            ->method('encryptForDatabase')
            ->with('refresh-raw')
            ->willReturn('ENC:refresh-raw');

        $gateway = $this->createMock(TokenStorageGateway::class);
        $gateway->method('execute')
            ->willReturnCallback(function (string $sql, array $binds) use (&$capturedBinds): void {
                $capturedBinds = $binds;
            });

        $repo = new UserOAuthTokenRepository($crypto, $gateway);
        $repo->upsert(7, 'refresh-raw', null, null);

        $this->assertSame([7, 'ENC:refresh-raw', null, null], $capturedBinds);
    }

    #[Test]
    public function findByUserDecryptsTokens(): void
    {
        $capturedSql = null;
        $capturedBinds = null;

        $gateway = $this->createMock(TokenStorageGateway::class);
        $gateway->method('fetchRow')
            ->willReturnCallback(function (string $sql, array $binds) use (&$capturedSql, &$capturedBinds): array {
                $capturedSql = $sql;
                $capturedBinds = $binds;

                return [
                    'refresh_token_encrypted' => 'ENC:refresh-raw',
                    'access_token_encrypted' => 'ENC:access-raw',
                    'access_token_expires_at' => '2026-01-02 03:04:05',
                ];
            });

        $repo = new UserOAuthTokenRepository($this->reversibleCrypto(), $gateway);
        $token = $repo->findByUser(7);

        $this->assertStringContainsStringIgnoringCase('SELECT', (string) $capturedSql);
        $this->assertStringContainsStringIgnoringCase('WHERE `openemr_user_id` = ?', (string) $capturedSql);
        $this->assertSame([7], $capturedBinds);

        $this->assertInstanceOf(UserOAuthToken::class, $token);
        $this->assertSame(7, $token->openEmrUserId);
        $this->assertSame('refresh-raw', $token->refreshToken);
        $this->assertSame('access-raw', $token->accessToken);
        $this->assertInstanceOf(\DateTimeImmutable::class, $token->accessTokenExpiresAt);
        $this->assertSame('2026-01-02 03:04:05', $token->accessTokenExpiresAt->format('Y-m-d H:i:s'));
    }

    #[Test]
    public function findByUserReturnsNullWhenAbsent(): void
    {
        $gateway = $this->createMock(TokenStorageGateway::class);
        $gateway->method('fetchRow')->willReturn(null);

        $repo = new UserOAuthTokenRepository($this->reversibleCrypto(), $gateway);

        $this->assertNull($repo->findByUser(7));
    }

    #[Test]
    public function findByUserHandlesNullAccessToken(): void
    {
        $gateway = $this->createMock(TokenStorageGateway::class);
        $gateway->method('fetchRow')->willReturn([
            'refresh_token_encrypted' => 'ENC:refresh-raw',
            'access_token_encrypted' => null,
            'access_token_expires_at' => null,
        ]);

        $repo = new UserOAuthTokenRepository($this->reversibleCrypto(), $gateway);
        $token = $repo->findByUser(7);

        $this->assertInstanceOf(UserOAuthToken::class, $token);
        $this->assertSame('refresh-raw', $token->refreshToken);
        $this->assertNull($token->accessToken);
        $this->assertNull($token->accessTokenExpiresAt);
    }

    #[Test]
    public function deleteByUserBuildsDeleteQuery(): void
    {
        $capturedSql = null;
        $capturedBinds = null;

        $gateway = $this->createMock(TokenStorageGateway::class);
        $gateway->expects($this->once())
            ->method('execute')
            ->willReturnCallback(function (string $sql, array $binds) use (&$capturedSql, &$capturedBinds): void {
                $capturedSql = $sql;
                $capturedBinds = $binds;
            });

        $repo = new UserOAuthTokenRepository($this->reversibleCrypto(), $gateway);
        $repo->deleteByUser(7);

        $this->assertStringContainsStringIgnoringCase('DELETE FROM', (string) $capturedSql);
        $this->assertStringContainsStringIgnoringCase('WHERE `openemr_user_id` = ?', (string) $capturedSql);
        $this->assertSame([7], $capturedBinds);
    }
}
