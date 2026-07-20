<?php

/**
 * Chat Proxy Request Parsing Test for Clinical Co-Pilot Module
 *
 * Exercises the pure request-validation logic used by the P2.14 chat proxy
 * (ChatProxyController): a decoded JSON body is parsed into a typed
 * ChatProxyRequest, or rejected. No database or session is needed, so this
 * runs isolated -- the proxy's CSRF/method/session gating (which does need a
 * live session) is exercised by the paired Panther scenario instead, same
 * discipline as the P2.13 token broker (see ClinicalCopilotTokenBrokerTest).
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

class ChatProxyRequestTest extends TestCase
{
    private const CLASS_NAME = 'OpenEMR\\Modules\\ClinicalCopilot\\Chat\\ChatProxyRequest';
    private const EXCEPTION_CLASS_NAME = 'OpenEMR\\Modules\\ClinicalCopilot\\Chat\\ChatProxyRequestException';

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
    public function testValidPayloadParsesMessageAndConversationIdAndToken(): void
    {
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $request = $fromArray([
            'message' => 'What medications is she on?',
            'conversation_id' => 'abc-123',
            'token' => 'dev-token',
        ]);

        $this->assertSame('What medications is she on?', $request->message);
        $this->assertSame('abc-123', $request->conversationId);
        $this->assertSame('dev-token', $request->token);
    }

    #[Test]
    public function testConversationIdIsOptionalAndDefaultsToNull(): void
    {
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $request = $fromArray([
            'message' => 'hello',
            'token' => 'dev-token',
        ]);

        $this->assertNull($request->conversationId);
    }

    #[Test]
    public function testMessageIsTrimmed(): void
    {
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $request = $fromArray([
            'message' => '  hello there  ',
            'token' => 'dev-token',
        ]);

        $this->assertSame('hello there', $request->message);
    }

    #[Test]
    public function testMissingMessageIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray(['token' => 'dev-token']);
    }

    #[Test]
    public function testBlankMessageIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray(['message' => '   ', 'token' => 'dev-token']);
    }

    #[Test]
    public function testNonStringMessageIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray(['message' => 42, 'token' => 'dev-token']);
    }

    #[Test]
    public function testOverlongMessageIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray(['message' => str_repeat('a', 4001), 'token' => 'dev-token']);
    }

    #[Test]
    public function testMissingTokenIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray(['message' => 'hello']);
    }

    #[Test]
    public function testBlankTokenIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray(['message' => 'hello', 'token' => '']);
    }

    #[Test]
    public function testNonStringConversationIdIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray(['message' => 'hello', 'token' => 'dev-token', 'conversation_id' => 42]);
    }

    #[Test]
    public function testBlankConversationIdIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray(['message' => 'hello', 'token' => 'dev-token', 'conversation_id' => '']);
    }
}
