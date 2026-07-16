<?php

/**
 * Clinical Co-Pilot Chat Proxy Controller
 *
 * Same-origin SSE bridge for the P2.14 chat panel (plan §4.7/§5): the agent
 * service sits on the `copilot_internal` docker network only (no host port,
 * no browser-reachable URL -- see docker-compose.copilot.yml) so the browser
 * cannot call it directly. This controller streams the agent's
 * `POST /chat` SSE response through the OpenEMR origin instead: the browser
 * talks only to OpenEMR (no CORS, no new attack surface on the agent).
 *
 * Load-bearing runtime controls (NOT redundant "defense-in-depth"): with the
 * dev-token bridge the agent holds a powerful OpenEMR token and its own
 * `/chat` validator is only a non-empty dev-stub, so this session+CSRF gate
 * and the server-anchored `patient_id` below -- together with the planner's
 * patient-context binding -- keep a request scoped to the authenticated user
 * and the patient their panel was opened on. Agent-side DevAgentToken HMAC +
 * pid validation is the tracked hardening (issue #127).
 *
 * The forwarded `patient_id` is never taken from client input -- read from
 * the session via PatientSessionUtil, same as the rest of this module (see
 * CopilotPanelController) -- so the panel cannot be tricked into streaming
 * a different patient's conversation than the one it was opened on.
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
use OpenEMR\Common\Session\PatientSessionUtil;
use OpenEMR\Common\Session\SessionWrapperFactory;
use OpenEMR\Core\OEGlobalsBag;
use OpenEMR\Modules\ClinicalCopilot\Chat\ChatProxyRequest;
use OpenEMR\Modules\ClinicalCopilot\Chat\ChatProxyRequestException;

final class ChatProxyController
{
    /**
     * OpenEMR global (OEGlobalsBag) that overrides the agent base URL.
     * Matches TokenBrokerController's global/default exactly -- both
     * controllers resolve the same server-configured agent origin, never
     * anything from request input, so the panel cannot be redirected to an
     * attacker-chosen host.
     */
    private const AGENT_URL_GLOBAL = 'clinical_copilot_agent_url';

    private const DEFAULT_AGENT_URL = 'http://agent:8000';

    /**
     * Ceiling on the upstream request. Generous: the planner loop can take
     * several sequential model calls (P2.8, up to 6 turns) before the
     * agent emits its first post-`conversation` frame.
     *
     * Known carry-forward: Apache's own `Timeout` directive (60s in the dev
     * stack) can still close an idle connection during a long silent gap
     * between the agent's `conversation` frame and its next frame (the
     * agent batches everything after the planner loop completes -- see
     * app/chat.py's SSE frame contract). Observed real-model latencies
     * (single-question: ~9-29s) stay well under that ceiling; a keep-alive
     * ping mechanism would close the gap for pathological multi-turn
     * questions but is not implemented here (no raw curl_* -- see
     * ForbiddenCurlFunctionsRule -- and Guzzle's streaming body read() does
     * not offer a low-level per-chunk callback to interleave one).
     */
    private const UPSTREAM_TIMEOUT_SECONDS = 300;

    /** Bytes read per iteration while relaying the upstream SSE body. */
    private const STREAM_READ_CHUNK_BYTES = 8192;

    public function __construct(
        private readonly RawRequestBodyReader $bodyReader = new RawRequestBodyReader()
    ) {
    }

    public function handleRequest(): void
    {
        $method = filter_input(INPUT_SERVER, 'REQUEST_METHOD', FILTER_UNSAFE_RAW, FILTER_REQUIRE_SCALAR);
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
            $chatRequest = ChatProxyRequest::fromArray($decoded);
        } catch (ChatProxyRequestException) {
            $this->sendJsonError('Invalid request', 400);
            return;
        }

        $pid = PatientSessionUtil::getPid();
        if ($pid <= 0) {
            $this->sendJsonError('No patient in session', 400);
            return;
        }

        $this->streamFromAgent($chatRequest, $pid);
    }

    private function streamFromAgent(ChatProxyRequest $chatRequest, int $pid): void
    {
        header('Content-Type: text/event-stream');
        header('Cache-Control: no-cache');
        header('X-Accel-Buffering: no');
        header('Connection: keep-alive');

        // Disable every layer of output buffering so bytes reach Apache (and
        // the browser) as they are read from the upstream body, instead of
        // being held until the script ends.
        while (ob_get_level() > 0) {
            ob_end_flush();
        }
        ob_implicit_flush(true);
        set_time_limit(0);

        $client = new Client();

        try {
            $response = $client->post(rtrim($this->agentUrl(), '/') . '/chat', [
                'json' => [
                    'message' => $chatRequest->message,
                    'patient_id' => $pid,
                    'conversation_id' => $chatRequest->conversationId,
                ],
                'headers' => [
                    'Authorization' => 'Bearer ' . $chatRequest->token,
                ],
                'stream' => true,
                'timeout' => self::UPSTREAM_TIMEOUT_SECONDS,
                'http_errors' => false,
            ]);
        } catch (GuzzleException) {
            $this->emitErrorFrame(0);
            return;
        }

        if ($response->getStatusCode() !== 200) {
            // A non-200 upstream response is a JSON error body, not an SSE
            // frame -- emit a clean `error` frame instead of relaying it.
            $this->emitErrorFrame($response->getStatusCode());
            return;
        }

        $body = $response->getBody();
        while (!$body->eof()) {
            echo $body->read(self::STREAM_READ_CHUNK_BYTES);
            flush();
        }
    }

    private function emitErrorFrame(int $upstreamStatus): void
    {
        echo "event: error\ndata: " . json_encode(['status' => $upstreamStatus], JSON_THROW_ON_ERROR) . "\n\n";
        flush();
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
