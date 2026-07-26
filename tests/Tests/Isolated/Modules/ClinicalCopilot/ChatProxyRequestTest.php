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

    /**
     * Issue #167 Gate 3 re-review finding: the ASCII-only overlong-message
     * test above passes identically whether the length check is byte-based
     * (strlen) or character-based (mb_strlen) -- it gives zero coverage for
     * the strlen -> mb_strlen fix (ChatProxyRequest::parseMessage), so
     * reverting that fix would leave this suite green. A multi-byte
     * (3-byte-per-character UTF-8) message pins the actual unit: exactly
     * 4000 CHARACTERS (12,000 BYTES) must be ACCEPTED -- a byte-based
     * strlen() check would wrongly reject this at 4000 bytes (barely 1333
     * characters in), well under the documented 4000-character limit.
     */
    #[Test]
    public function testMultiByteMessageAtExactly4000CharactersIsAccepted(): void
    {
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $message = str_repeat('病', 4000);
        $this->assertSame(12000, strlen($message), 'sanity: each 病 char is 3 UTF-8 bytes');
        $this->assertSame(4000, mb_strlen($message, 'UTF-8'));

        $request = $fromArray(['message' => $message, 'token' => 'dev-token']);

        $this->assertSame($message, $request->message);
    }

    /** Mirrors the boundary shape of the Python ChatRequest tests: one
     * character over the limit is rejected, multi-byte or not. */
    #[Test]
    public function testMultiByteMessageAt4001CharactersIsRejected(): void
    {
        $this->expectException(self::EXCEPTION_CLASS_NAME);
        $fromArray = [self::CLASS_NAME, 'fromArray'];
        $fromArray(['message' => str_repeat('病', 4001), 'token' => 'dev-token']);
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
