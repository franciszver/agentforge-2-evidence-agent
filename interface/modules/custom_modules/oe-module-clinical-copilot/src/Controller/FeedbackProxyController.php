<?php

/**
 * Clinical Co-Pilot Feedback Proxy Controller
 *
 * Same-origin bridge for the P4.4 feedback buttons: the agent service sits
 * on the `copilot_internal` docker network only (no host port, no
 * browser-reachable URL -- see docker-compose.copilot.yml), so the browser
 * cannot call its `POST /feedback` (P4.3) directly. This controller forwards
 * the panel's thumbs up/down (+ optional comment) through the OpenEMR origin
 * instead -- the browser talks only to OpenEMR, exactly the same shape as
 * ChatProxyController's SSE bridge, just a single request/response instead
 * of a stream.
 *
 * Auth/CSRF/agent-URL resolution deliberately mirror ChatProxyController
 * (which passed a rigorous security review) rather than reimplementing a
 * second pattern: session-anchored auth check, CSRF-gated on the same
 * `csrf_token_form` field, and the same server-configured agent origin
 * (never client input, so the panel cannot be redirected to an
 * attacker-chosen host).
 *
 * No patient_id involved: feedback is linked to a `/chat` response purely by
 * the P4.1 correlation id the panel already has from that response's
 * `conversation` SSE frame (see app/chat.py) -- there is no patient-context
 * binding to enforce here, unlike the chat proxy.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Controller;

use GuzzleHttp\Client;
use GuzzleHttp\Exception\GuzzleException;
use OpenEMR\Common\Csrf\CsrfUtils;
use OpenEMR\Common\Http\RawRequestBodyReader;
use OpenEMR\Common\Session\SessionWrapperFactory;
use OpenEMR\Core\OEGlobalsBag;
use OpenEMR\Modules\ClinicalCopilot\Chat\FeedbackProxyRequest;
use OpenEMR\Modules\ClinicalCopilot\Chat\FeedbackProxyRequestException;

final class FeedbackProxyController
{
    /**
     * OpenEMR global (OEGlobalsBag) that overrides the agent base URL.
     * Matches ChatProxyController's global/default exactly -- both
     * controllers resolve the same server-configured agent origin.
     */
    private const AGENT_URL_GLOBAL = 'clinical_copilot_agent_url';

    private const DEFAULT_AGENT_URL = 'http://agent:8000';

    /**
     * A feedback write is a single, quick record_feedback_span() call (P4.3)
     * -- not a multi-turn planner loop -- so a short ceiling is appropriate,
     * unlike ChatProxyController's generous streaming timeout.
     */
    private const UPSTREAM_TIMEOUT_SECONDS = 15;

    public function __construct(
        private readonly RawRequestBodyReader $bodyReader = new RawRequestBodyReader(),
        /**
         * Request method override for tests. ``filter_input(INPUT_SERVER, ...)``
         * reads PHP's original input buffer, not the (test-mutated)
         * ``$_SERVER`` superglobal, so it cannot be driven from a PHPUnit
         * test by assigning ``$_SERVER['REQUEST_METHOD']`` -- this seam is
         * the workaround. ``null`` (the production default) falls back to
         * the real ``filter_input()`` read.
         */
        private readonly ?string $requestMethod = null,
    ) {
    }

    public function handleRequest(): void
    {
        $method = $this->requestMethod
            ?? filter_input(INPUT_SERVER, 'REQUEST_METHOD', FILTER_UNSAFE_RAW, FILTER_REQUIRE_SCALAR);
        if ($method !== 'POST') {
            $this->sendJsonError('Method not allowed', 405);
            return;
        }

        $session = SessionWrapperFactory::getInstance()->getActiveSession();

        $rawAuthUserId = $session->get('authUserID');
        $authUserId = is_numeric($rawAuthUserId) ? (int) $rawAuthUserId : 0;
        if ($authUserId <= 0) {
            $this->sendJsonError('Not authenticated', 401);
            return;
        }

        $decoded = json_decode($this->bodyReader->read(), true);
        $decoded = is_array($decoded) ? $decoded : [];

        $csrfToken = $decoded['csrf_token_form'] ?? null;
        if (!is_string($csrfToken) || !CsrfUtils::verifyCsrfToken($csrfToken, $session)) {
            $this->sendJsonError('CSRF verification failed', 403);
            return;
        }

        try {
            /** @var array<array-key, mixed> $decoded */
            $feedbackRequest = FeedbackProxyRequest::fromArray($decoded);
        } catch (FeedbackProxyRequestException) {
            $this->sendJsonError('Invalid request', 400);
            return;
        }

        $this->forwardToAgent($feedbackRequest);
    }

    private function forwardToAgent(FeedbackProxyRequest $feedbackRequest): void
    {
        $client = new Client();

        try {
            $response = $client->post(rtrim($this->agentUrl(), '/') . '/feedback', [
                'json' => [
                    'correlation_id' => $feedbackRequest->correlationId,
                    'thumb' => $feedbackRequest->thumb,
                    'comment' => $feedbackRequest->comment,
                ],
                'headers' => [
                    'Authorization' => 'Bearer ' . $feedbackRequest->token,
                ],
                'timeout' => self::UPSTREAM_TIMEOUT_SECONDS,
                'http_errors' => false,
            ]);
        } catch (GuzzleException) {
            $this->sendJsonError('Feedback service unavailable', 502);
            return;
        }

        // Relay the agent's status + JSON body transparently: app/feedback.py
        // already returns generic, non-leaking error details (see its
        // module docstring), so there is nothing further to sanitize here.
        header('Content-Type: application/json');
        http_response_code($response->getStatusCode());
        echo $response->getBody()->getContents();
    }

    private function agentUrl(): string
    {
        $configured = OEGlobalsBag::getInstance()->get(self::AGENT_URL_GLOBAL);

        return is_string($configured) && $configured !== '' ? $configured : self::DEFAULT_AGENT_URL;
    }

    private function sendJsonError(string $message, int $code): void
    {
        header('Content-Type: application/json');
        http_response_code($code);
        echo json_encode(['error' => $message]);
    }
}
