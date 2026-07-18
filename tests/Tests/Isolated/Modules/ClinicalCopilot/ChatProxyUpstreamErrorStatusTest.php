<?php

/**
 * Chat Proxy upstream-error-status decision test (Clinical Co-Pilot).
 *
 * Exercises ChatProxyController::upstreamErrorStatus() in isolation: the pure
 * mapping from a caught GuzzleException to the status code emitted in the SSE
 * `error` frame. This is the decision point where a regression would actually
 * hide -- e.g. a Guzzle exception-hierarchy change or a reordered catch block
 * silently turning a real 500 into the transfer-error sentinel, or (the bug
 * this locks) a mid-stream drop emitting a self-contradictory `status: 200`
 * inside an `error` frame.
 *
 * No output capture and no live agent are needed -- the four inputs are
 * Guzzle-constructed exceptions -- so this dodges the output-buffer-drain
 * problem that blocks a full controller-path test. The end-to-end error-path
 * wiring (the `on_headers` non-200 throw under a real stream) is covered by
 * the live acceptance measurement per TEST_PLAN's mock-pairing rule; this
 * unit test covers only the decision logic.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <ciscodg@gmail.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Tests\Isolated\Modules\ClinicalCopilot;

use GuzzleHttp\Exception\ConnectException;
use GuzzleHttp\Exception\GuzzleException;
use GuzzleHttp\Exception\RequestException;
use GuzzleHttp\Psr7\Request;
use GuzzleHttp\Psr7\Response;
use OpenEMR\Core\ModulesClassLoader;
use OpenEMR\Modules\ClinicalCopilot\Controller\ChatProxyController;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

class ChatProxyUpstreamErrorStatusTest extends TestCase
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

    #[Test]
    public function nonSuccessResponseYieldsItsRealStatus404(): void
    {
        $exception = new RequestException(
            'not found',
            new Request('POST', 'http://agent:8000/chat'),
            new Response(404)
        );

        $this->assertSame(404, $this->decide($exception));
    }

    #[Test]
    public function nonSuccessResponseYieldsItsRealStatus500(): void
    {
        $exception = new RequestException(
            'server error',
            new Request('POST', 'http://agent:8000/chat'),
            new Response(500)
        );

        $this->assertSame(500, $this->decide($exception));
    }

    #[Test]
    public function midStreamDropWith200ResponseYieldsTheTransferSentinelNotTwoHundred(): void
    {
        // on_headers only throws for a non-200 status, so a RequestException
        // carrying a 200 response is necessarily a mid-stream transfer drop
        // AFTER headers already parsed 200 -- never an on_headers rejection.
        // Emitting `status: 200` in an `error` frame would be self-contradictory.
        $exception = new RequestException(
            'transfer dropped',
            new Request('POST', 'http://agent:8000/chat'),
            new Response(200)
        );

        $this->assertSame(0, $this->decide($exception));
    }

    #[Test]
    public function connectExceptionWithNoResponseYieldsTheTransferSentinel(): void
    {
        $exception = new ConnectException(
            'connection refused',
            new Request('POST', 'http://agent:8000/chat')
        );

        $this->assertSame(0, $this->decide($exception));
    }

    /**
     * Reach the private decision method directly. The controller is `final`
     * (a subclass wrapper is impossible), so reflection is used to exercise
     * the pure helper without going through the streaming handleRequest()
     * path -- no session, output buffer, or live agent involved.
     */
    private function decide(GuzzleException $exception): int
    {
        $method = new \ReflectionMethod(ChatProxyController::class, 'upstreamErrorStatus');
        $status = $method->invoke(new ChatProxyController(), $exception);

        self::assertIsInt($status);

        return $status;
    }
}
