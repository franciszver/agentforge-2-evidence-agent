<?php

/**
 * Feedback Proxy Request
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
 * A validated feedback-widget request: parsed from the decoded JSON body
 * posted to the P4.4 feedback proxy (public/feedback-proxy.php), never
 * trusted raw. Mirrors ChatProxyRequest's shape/discipline (see that class):
 * the CSRF token is a transport-level concern the controller checks before
 * this class is ever reached, so it is deliberately excluded here too.
 */
final readonly class FeedbackProxyRequest
{
    /** Matches the agent's FeedbackRequest.comment bound (P4.3, app/feedback.py). */
    private const MAX_COMMENT_LENGTH = 2000;

    private const VALID_THUMBS = ['up', 'down'];

    private function __construct(
        public string $correlationId,
        public string $thumb,
        public ?string $comment,
        public string $token,
    ) {
    }

    /**
     * @param array<array-key, mixed> $decoded The JSON-decoded request body.
     *
     * @throws FeedbackProxyRequestException if the shape is invalid.
     */
    public static function fromArray(array $decoded): self
    {
        $correlationId = self::parseCorrelationId($decoded['correlation_id'] ?? null);
        $thumb = self::parseThumb($decoded['thumb'] ?? null);
        $comment = self::parseComment($decoded['comment'] ?? null);
        $token = self::parseToken($decoded['token'] ?? null);

        return new self($correlationId, $thumb, $comment, $token);
    }

    private static function parseCorrelationId(mixed $raw): string
    {
        if (!is_string($raw) || $raw === '') {
            throw new FeedbackProxyRequestException('correlation_id must be a non-empty string');
        }
        return $raw;
    }

    private static function parseThumb(mixed $raw): string
    {
        if (!is_string($raw) || !in_array($raw, self::VALID_THUMBS, true)) {
            throw new FeedbackProxyRequestException('thumb must be "up" or "down"');
        }
        return $raw;
    }

    private static function parseComment(mixed $raw): ?string
    {
        if ($raw === null) {
            return null;
        }
        if (!is_string($raw)) {
            throw new FeedbackProxyRequestException('comment must be a string when present');
        }
        if (strlen($raw) > self::MAX_COMMENT_LENGTH) {
            throw new FeedbackProxyRequestException('comment exceeds the maximum length');
        }
        return $raw;
    }

    private static function parseToken(mixed $raw): string
    {
        if (!is_string($raw) || $raw === '') {
            throw new FeedbackProxyRequestException('token must be a non-empty string');
        }
        return $raw;
    }
}
