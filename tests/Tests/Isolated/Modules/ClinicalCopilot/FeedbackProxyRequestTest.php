<?php

/**
 * Feedback Proxy Request Parsing Test for Clinical Co-Pilot Module
 *
 * Exercises the pure request-validation logic used by the P4.4 feedback
 * proxy (FeedbackProxyController): a decoded JSON body is parsed into a
 * typed FeedbackProxyRequest, or rejected. No database or session is
 * needed, so this runs isolated -- mirrors ChatProxyRequestTest for P2.14.
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

class FeedbackProxyRequestTest extends TestCase
{
    private const CLASS_NAME = 'OpenEMR\\Modules\\ClinicalCopilot\\Chat\\FeedbackProxyRequest';
    private const EXCEPTION_CLASS_NAME = 'OpenEMR\\Modules\\ClinicalCopilot\\Chat\\FeedbackProxyRequestException';

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
    public function testValidPayloadParsesCorrelationIdThumbCommentAndToken(): void
    {
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $request = $fromArray([
            'correlation_id' => 'corr-123',
            'thumb' => 'down',
            'comment' => 'Missed the recent A1C.',
            'token' => 'dev-token',
        ]);

        $this->assertSame('corr-123', $request->correlationId);
        $this->assertSame('down', $request->thumb);
        $this->assertSame('Missed the recent A1C.', $request->comment);
        $this->assertSame('dev-token', $request->token);
    }

    #[Test]
    public function testCommentIsOptionalAndDefaultsToNull(): void
    {
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $request = $fromArray([
            'correlation_id' => 'corr-123',
            'thumb' => 'up',
            'token' => 'dev-token',
        ]);

        $this->assertNull($request->comment);
    }

    #[Test]
    public function testMissingCorrelationIdIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray(['thumb' => 'up', 'token' => 'dev-token']);
    }

    #[Test]
    public function testBlankCorrelationIdIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray(['correlation_id' => '', 'thumb' => 'up', 'token' => 'dev-token']);
    }

    #[Test]
    public function testNonStringCorrelationIdIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray(['correlation_id' => 42, 'thumb' => 'up', 'token' => 'dev-token']);
    }

    #[Test]
    public function testMissingThumbIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray(['correlation_id' => 'corr-123', 'token' => 'dev-token']);
    }

    #[Test]
    public function testInvalidThumbValueIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray(['correlation_id' => 'corr-123', 'thumb' => 'sideways', 'token' => 'dev-token']);
    }

    #[Test]
    public function testOverlongCommentIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray([
            'correlation_id' => 'corr-123',
            'thumb' => 'up',
            'token' => 'dev-token',
            'comment' => str_repeat('a', 2001),
        ]);
    }

    #[Test]
    public function testCommentAtMaxLengthIsAccepted(): void
    {
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $request = $fromArray([
            'correlation_id' => 'corr-123',
            'thumb' => 'up',
            'token' => 'dev-token',
            'comment' => str_repeat('a', 2000),
        ]);

        $this->assertSame(2000, strlen((string) $request->comment));
    }

    #[Test]
    public function testNonStringCommentIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray(['correlation_id' => 'corr-123', 'thumb' => 'up', 'token' => 'dev-token', 'comment' => 42]);
    }

    #[Test]
    public function testMissingTokenIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray(['correlation_id' => 'corr-123', 'thumb' => 'up']);
    }

    #[Test]
    public function testBlankTokenIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray(['correlation_id' => 'corr-123', 'thumb' => 'up', 'token' => '']);
    }
}
