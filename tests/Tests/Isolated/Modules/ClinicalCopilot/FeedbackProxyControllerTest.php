<?php

/**
 * Feedback Proxy Controller Test for Clinical Co-Pilot Module (P4.4)
 *
 * Exercises FeedbackProxyController's session/CSRF/method gating in
 * isolation -- no database needed, since a Symfony Session can be
 * constructed in-memory (a stub backed by an array, same pattern as
 * CsrfUtilsTest) and injected via SessionWrapperFactory::setActiveSession
 * (same seam SiteSetupListenerTest uses). Every case tested here returns
 * BEFORE the controller ever calls out to the agent, so no network access
 * is needed either.
 *
 * The request method is passed to the controller's ``$requestMethod``
 * constructor seam rather than assigned to ``$_SERVER['REQUEST_METHOD']``:
 * ``filter_input(INPUT_SERVER, ...)`` reads PHP's original input buffer, not
 * a test-mutated superglobal, so mutating ``$_SERVER`` has no effect on it.
 *
 * The controller's actual upstream relay (forwardToAgent) is NOT exercised
 * here -- that needs a real (or mocked-at-the-HTTP-layer) agent, which is
 * out of scope for an isolated unit test; it is covered by the paired
 * Panther scenario (ClinicalCopilotFeedbackTest) against the real dev stack,
 * same discipline ChatProxyController's Panther coverage uses.
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
use OpenEMR\Common\Http\RawRequestBodyReader;
use OpenEMR\Common\Session\SessionWrapperFactory;
use OpenEMR\Core\ModulesClassLoader;
use OpenEMR\Modules\ClinicalCopilot\Controller\FeedbackProxyController;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Symfony\Component\HttpFoundation\Session\SessionInterface;

class FeedbackProxyControllerTest extends TestCase
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
    }

    protected function tearDown(): void
    {
        $this->resetSessionWrapperFactorySingleton();
    }

    #[Test]
    public function testNonPostMethodIsRejectedWith405(): void
    {
        [$status, $body] = $this->invokeController(method: 'GET', bodyJson: '{}');

        $this->assertSame(405, $status);
        $this->assertArrayNotHasKey('token', $body);
    }

    #[Test]
    public function testUnauthenticatedSessionIsRejectedWith401(): void
    {
        $session = $this->makeSession(authUserId: 0);
        SessionWrapperFactory::getInstance()->setActiveSession($session);

        [$status, $body] = $this->invokeController(method: 'POST', bodyJson: '{}');

        $this->assertSame(401, $status);
        $this->assertArrayNotHasKey('token', $body);
    }

    #[Test]
    public function testMissingCsrfTokenIsRejectedWith403(): void
    {
        $session = $this->makeSession(authUserId: 5);
        SessionWrapperFactory::getInstance()->setActiveSession($session);

        $payload = json_encode([
            'correlation_id' => 'corr-1',
            'thumb' => 'up',
            'token' => 'dev-token',
        ], JSON_THROW_ON_ERROR);

        [$status, $body] = $this->invokeController(method: 'POST', bodyJson: $payload);

        $this->assertSame(403, $status);
        $this->assertArrayNotHasKey('token', $body);
    }

    #[Test]
    public function testTamperedCsrfTokenIsRejectedWith403(): void
    {
        $session = $this->makeSession(authUserId: 5);
        SessionWrapperFactory::getInstance()->setActiveSession($session);
        $validToken = CsrfUtils::collectCsrfToken($session);

        $payload = json_encode([
            'csrf_token_form' => $validToken . 'tampered',
            'correlation_id' => 'corr-1',
            'thumb' => 'up',
            'token' => 'dev-token',
        ], JSON_THROW_ON_ERROR);

        [$status, $body] = $this->invokeController(method: 'POST', bodyJson: $payload);

        $this->assertSame(403, $status);
        $this->assertArrayNotHasKey('token', $body);
    }

    #[Test]
    public function testMalformedBodyMissingThumbIsRejectedWith400(): void
    {
        $session = $this->makeSession(authUserId: 5);
        SessionWrapperFactory::getInstance()->setActiveSession($session);
        $validToken = CsrfUtils::collectCsrfToken($session);

        $payload = json_encode([
            'csrf_token_form' => $validToken,
            'correlation_id' => 'corr-1',
            'token' => 'dev-token',
            // 'thumb' deliberately omitted
        ], JSON_THROW_ON_ERROR);

        [$status, $body] = $this->invokeController(method: 'POST', bodyJson: $payload);

        $this->assertSame(400, $status);
        $this->assertArrayNotHasKey('token', $body);
    }

    #[Test]
    public function testMalformedBodyInvalidThumbValueIsRejectedWith400(): void
    {
        $session = $this->makeSession(authUserId: 5);
        SessionWrapperFactory::getInstance()->setActiveSession($session);
        $validToken = CsrfUtils::collectCsrfToken($session);

        $payload = json_encode([
            'csrf_token_form' => $validToken,
            'correlation_id' => 'corr-1',
            'thumb' => 'sideways',
            'token' => 'dev-token',
        ], JSON_THROW_ON_ERROR);

        [$status, $body] = $this->invokeController(method: 'POST', bodyJson: $payload);

        $this->assertSame(400, $status);
    }

    #[Test]
    public function testNonJsonBodyNeverReachesParsingUnhandled(): void
    {
        $session = $this->makeSession(authUserId: 5);
        SessionWrapperFactory::getInstance()->setActiveSession($session);

        // Not valid JSON at all -- json_decode() yields null, decoded to [],
        // so this exercises the same "no csrf_token_form present" 403 path
        // as a missing field would (the CSRF gate runs before body shape
        // validation) -- confirms malformed input never reaches
        // FeedbackProxyRequest::fromArray() with an unhandled error.
        [$status] = $this->invokeController(method: 'POST', bodyJson: 'not json at all');

        $this->assertSame(403, $status);
    }

    /**
     * Runs the controller against the given method/session/body, capturing
     * its HTTP status + decoded JSON response body without ever touching the
     * network (every case here returns before forwardToAgent() would be
     * reached).
     *
     * @return array{0: int, 1: array<array-key, mixed>}
     */
    private function invokeController(string $method, string $bodyJson): array
    {
        $bodyReader = new RawRequestBodyReader('data://text/plain;base64,' . base64_encode($bodyJson));
        $controller = new FeedbackProxyController($bodyReader, $method);

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
