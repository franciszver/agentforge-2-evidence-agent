<?php

/**
 * Dev Agent Token Test for Clinical Co-Pilot Module
 *
 * Exercises the pure token-minting logic used by the P2.13 token broker: a
 * compact, HMAC-signed, dev-only bearer token that represents the logged-in
 * OpenEMR user. No database or session is needed, so this runs isolated.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Tests\Isolated\Modules\ClinicalCopilot;

use OpenEMR\Core\ModulesClassLoader;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

class DevAgentTokenTest extends TestCase
{
    private const CLASS_NAME = 'OpenEMR\\Modules\\ClinicalCopilot\\Auth\\DevAgentToken';

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
    public function testMintEncodesTheLoggedInUserAndExpiry(): void
    {
        $mint = self::CLASS_NAME . '::mint';
        $token = $mint(42, 'admin', 7, 'signing-key', 1_000_000, 3600);

        $this->assertNotSame('', $token);
        $segments = explode('.', $token);
        $this->assertCount(2, $segments, 'token must be payload.signature');
        $this->assertNotSame('', $segments[0]);
        $this->assertNotSame('', $segments[1]);

        $payload = json_decode((string) $this->base64UrlDecode($segments[0]), true);
        $this->assertIsArray($payload);
        $this->assertSame(42, $payload['sub'] ?? null, 'token binds the logged-in user id');
        $this->assertSame('admin', $payload['username'] ?? null);
        $this->assertSame(7, $payload['pid'] ?? null, 'token is anchored to the panel pid');
        $this->assertSame('copilot-dev', $payload['typ'] ?? null, 'token is marked dev-only');
        $this->assertSame(1_000_000, $payload['iat'] ?? null);
        $this->assertSame(1_003_600, $payload['exp'] ?? null, 'exp is iat + ttl');
    }

    #[Test]
    public function testSignatureBindsToPayloadWithTheSigningKey(): void
    {
        $mint = self::CLASS_NAME . '::mint';
        $token = $mint(42, 'admin', 7, 'signing-key', 1_000_000, 3600);
        [$payloadSegment, $signatureSegment] = explode('.', $token);

        $expected = $this->base64UrlEncode(hash_hmac('sha256', $payloadSegment, 'signing-key', true));
        $this->assertSame($expected, $signatureSegment, 'signature is HMAC-SHA256 over the payload segment');

        $wrongKey = $this->base64UrlEncode(hash_hmac('sha256', $payloadSegment, 'other-key', true));
        $this->assertNotSame($wrongKey, $signatureSegment, 'signature must depend on the secret signing key');
    }

    #[Test]
    public function testDistinctUsersProduceDistinctSignatures(): void
    {
        $mint = self::CLASS_NAME . '::mint';
        $tokenA = $mint(42, 'admin', 7, 'signing-key', 1_000_000, 3600);
        $tokenB = $mint(99, 'nurse', 7, 'signing-key', 1_000_000, 3600);

        $this->assertNotSame($tokenA, $tokenB, 'a different user must yield a different token');
    }

    private function base64UrlEncode(string $raw): string
    {
        return rtrim(strtr(base64_encode($raw), '+/', '-_'), '=');
    }

    private function base64UrlDecode(string $encoded): string
    {
        $padded = str_pad(strtr($encoded, '-_', '+/'), (int) (ceil(strlen($encoded) / 4) * 4), '=');
        $decoded = base64_decode($padded, true);
        return $decoded === false ? '' : $decoded;
    }
}
