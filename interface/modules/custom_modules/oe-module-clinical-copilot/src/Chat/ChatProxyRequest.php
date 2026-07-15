<?php

/**
 * Chat Proxy Request
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Chat;

/**
 * A validated chat-panel request: parsed from the decoded JSON body posted
 * to the P2.14 chat proxy (public/chat-proxy.php), never trusted raw.
 *
 * Deliberately excludes patient_id and the CSRF token: patient_id is never
 * taken from client input anywhere in this module (see
 * CopilotPanelController) -- ChatProxyController reads it from the session
 * via PatientSessionUtil -- and the CSRF token is a transport-level
 * concern the controller checks before this class is ever reached.
 */
final readonly class ChatProxyRequest
{
    /** Matches the agent's ChatRequest.message practical limit for a chat turn. */
    private const MAX_MESSAGE_LENGTH = 4000;

    private function __construct(
        public string $message,
        public ?string $conversationId,
        public string $token,
    ) {
    }

    /**
     * @param array<array-key, mixed> $decoded The JSON-decoded request body.
     *
     * @throws ChatProxyRequestException if the shape is invalid.
     */
    public static function fromArray(array $decoded): self
    {
        $message = self::parseMessage($decoded['message'] ?? null);
        $token = self::parseToken($decoded['token'] ?? null);
        $conversationId = self::parseConversationId($decoded['conversation_id'] ?? null);

        return new self($message, $conversationId, $token);
    }

    private static function parseMessage(mixed $raw): string
    {
        if (!is_string($raw)) {
            throw new ChatProxyRequestException('message must be a string');
        }
        $trimmed = trim($raw);
        if ($trimmed === '') {
            throw new ChatProxyRequestException('message must not be blank');
        }
        if (strlen($trimmed) > self::MAX_MESSAGE_LENGTH) {
            throw new ChatProxyRequestException('message exceeds the maximum length');
        }
        return $trimmed;
    }

    private static function parseToken(mixed $raw): string
    {
        if (!is_string($raw) || $raw === '') {
            throw new ChatProxyRequestException('token must be a non-empty string');
        }
        return $raw;
    }

    private static function parseConversationId(mixed $raw): ?string
    {
        if ($raw === null) {
            return null;
        }
        if (!is_string($raw) || $raw === '') {
            throw new ChatProxyRequestException('conversation_id must be a non-empty string when present');
        }
        return $raw;
    }
}
