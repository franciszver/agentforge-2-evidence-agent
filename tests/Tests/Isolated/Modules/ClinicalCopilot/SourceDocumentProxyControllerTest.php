<?php

/**
 * Source Document Proxy Controller Test for Clinical Co-Pilot Module (P3.7)
 *
 * Exercises SourceDocumentProxyController's method/auth/source_id-format
 * gating in isolation -- no database, no network. Every case here returns
 * BEFORE the controller would broker a token or call the agent, mirroring
 * FeedbackProxyControllerTest's scoping: the upstream relay itself needs a
 * real (or HTTP-mocked) agent and is left to the paired Panther/manual
 * verification, not this unit test.
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
use OpenEMR\Modules\ClinicalCopilot\Controller\SourceDocumentProxyController;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Symfony\Component\HttpFoundation\Session\SessionInterface;

class SourceDocumentProxyControllerTest extends TestCase
{
    protected function setUp(): void
    {
        $projectDir = dirname(__DIR__, 5);
        $classLoader = new ModulesClassLoader($projectDir);
        $classLoader->registerNamespaceIfNotExists(
            'OpenEMR\\Modules\\ClinicalCopilot\\',
            $projectDir . '/interface/modules/custom_modules/oe-module-clinical-copilot/src'
        );

        $this->resetSessionWrapperFactorySingleton();
        $_GET = [];
    }

    protected function tearDown(): void
    {
        $this->resetSessionWrapperFactorySingleton();
        $_GET = [];
    }

    #[Test]
    public function testNonGetMethodIsRejectedWith405(): void
    {
        [$status] = $this->invokeController(method: 'POST', sourceId: 'a' . str_repeat('0', 31));

        $this->assertSame(405, $status);
    }

    #[Test]
    public function testUnauthenticatedSessionIsRejectedWith401(): void
    {
        $session = $this->makeSession(authUserId: 0);
        SessionWrapperFactory::getInstance()->setActiveSession($session);

        [$status] = $this->invokeController(method: 'GET', sourceId: str_repeat('a', 32));

        $this->assertSame(401, $status);
    }

    #[Test]
    public function testAuthenticatedButMissingSourceIdIsRejectedWith400(): void
    {
        $session = $this->makeSession(authUserId: 5);
        SessionWrapperFactory::getInstance()->setActiveSession($session);

        [$status, $body] = $this->invokeController(method: 'GET', sourceId: null);

        $this->assertSame(400, $status);
        $this->assertArrayHasKey('error', $body);
    }

    /**
     * @return array<string, array{string}>
     *
     * @codeCoverageIgnore Data providers run before coverage instrumentation starts.
     */
    public static function malformedSourceIdProvider(): array
    {
        return [
            'path traversal' => ['../../../../etc/passwd'],
            'encoded path traversal' => ['..%2f..%2fetc%2fpasswd'],
            'non-hex characters' => ['not-hex-chars-!!!!!!!!!!!!!!!!!!'],
            'too short' => ['abc123'],
            'hex plus extra suffix' => [str_repeat('a', 32) . '-extra'],
            'uppercase hex (store never emits this)' => [str_repeat('A', 32)],
        ];
    }

    #[Test]
    #[\PHPUnit\Framework\Attributes\DataProvider('malformedSourceIdProvider')]
    public function testMalformedOrPathTraversalSourceIdIsRejectedWith400(string $sourceId): void
    {
        $session = $this->makeSession(authUserId: 5);
        SessionWrapperFactory::getInstance()->setActiveSession($session);

        [$status, $body] = $this->invokeController(method: 'GET', sourceId: $sourceId);

        $this->assertSame(400, $status);
        $this->assertArrayHasKey('error', $body);
    }

    /**
     * @return array{0: int, 1: array<array-key, mixed>}
     */
    private function invokeController(string $method, ?string $sourceId): array
    {
        $_GET = $sourceId !== null ? ['source_id' => $sourceId] : [];

        $controller = new SourceDocumentProxyController(requestMethod: $method);

        ob_start();
        $controller->handleRequest();
        $output = ob_get_clean();

        $status = http_response_code();
        $decoded = json_decode(is_string($output) ? $output : '', true);

        return [is_int($status) ? $status : 0, is_array($decoded) ? $decoded : []];
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
        CsrfUtils::setupCsrfKey($session);
        return $session;
    }

    private function resetSessionWrapperFactorySingleton(): void
    {
        $reflection = new \ReflectionClass(SessionWrapperFactory::class);
        $instancesProperty = $reflection->getProperty('instances');
        $instancesProperty->setValue(null, []);
    }
}
