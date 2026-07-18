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
 * @author    Francisco de Guzman <ciscodg@gmail.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Controller;

use GuzzleHttp\Client;
use GuzzleHttp\Exception\GuzzleException;
use GuzzleHttp\Exception\RequestException;
use GuzzleHttp\Handler\CurlMultiHandler;
use GuzzleHttp\HandlerStack;
use OpenEMR\Common\Csrf\CsrfUtils;
use OpenEMR\Common\Http\RawRequestBodyReader;
use OpenEMR\Common\Session\PatientSessionUtil;
use OpenEMR\Common\Session\SessionWrapperFactory;
use OpenEMR\Core\OEGlobalsBag;
use OpenEMR\Modules\ClinicalCopilot\Chat\ChatProxyRequest;
use OpenEMR\Modules\ClinicalCopilot\Chat\ChatProxyRequestException;
use OpenEMR\Modules\ClinicalCopilot\Http\FlushingOutputStream;
use Psr\Http\Message\ResponseInterface;

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
     * questions but is not implemented here -- FlushingOutputStream::write()
     * (#211) now fires per upstream chunk, which is where a ping would hook
     * in, but no such hook exists yet (separate follow-up).
     */
    private const UPSTREAM_TIMEOUT_SECONDS = 300;

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
            // The global launcher (P2.17) opens this panel on every page,
            // including ones with no patient selected. Signal that case with
            // a stable machine-readable reason so the panel can show the
            // "open a patient chart first" hint instead of a generic error.
            $this->sendJsonError('No patient in session', 400, 'no_patient_in_session');
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

        // PHP's `http://` stream wrapper (what Guzzle's default StreamHandler,
        // and raw fopen(), are both built on) buffers the ENTIRE upstream
        // response before exposing any of it via read() -- proven live for
        // issue #211 (see scratchpad u211-stream-timing.txt). CurlMultiHandler
        // drives curl_multi_exec() directly, so libcurl's CURLOPT_WRITEFUNCTION
        // calls the sink's write() incrementally as each chunk arrives -- the
        // only Guzzle-native path that streams (no raw curl_*; see
        // ForbiddenCurlFunctionsRule).
        $client = new Client([
            'handler' => HandlerStack::create(new CurlMultiHandler()),
        ]);

        $sink = new FlushingOutputStream();

        try {
            $promise = $client->postAsync(rtrim($this->agentUrl(), '/') . '/chat', [
                'json' => [
                    'message' => $chatRequest->message,
                    'patient_id' => $pid,
                    'conversation_id' => $chatRequest->conversationId,
                ],
                'headers' => [
                    'Authorization' => 'Bearer ' . $chatRequest->token,
                ],
                'stream' => true,
                'sink' => $sink,
                'timeout' => self::UPSTREAM_TIMEOUT_SECONDS,
                // Inspect the upstream status as soon as headers arrive --
                // BEFORE any body bytes reach the sink (and thus the
                // browser). A non-200 upstream response is a JSON error
                // body, not an SSE frame; throwing here aborts the transfer
                // so it never gets relayed, matching the prior (blocking)
                // post()-then-check behavior.
                'on_headers' => function (ResponseInterface $response): void {
                    if ($response->getStatusCode() !== 200) {
                        throw new \RuntimeException('non-200 upstream status');
                    }
                },
            ]);

            $promise->wait();
        } catch (GuzzleException $e) {
            $this->emitErrorFrame($this->upstreamErrorStatus($e));
            return;
        }
    }

    /**
     * Map a caught upstream failure to the status code carried in the SSE
     * `error` frame.
     *
     * The `on_headers` callback throws only for a non-200 upstream status, so
     * the response attached to a caught RequestException disambiguates the two
     * failure modes:
     *   - a non-200 response is that `on_headers` rejection -> report its real
     *     status (404, 500, ...);
     *   - a 200 response means headers already parsed 200 and the body
     *     transfer dropped mid-stream -> report the 0 transfer-error sentinel,
     *     never a self-contradictory `status: 200` inside an `error` frame.
     * Any exception with no response (agent down / DNS / refused connection,
     * e.g. ConnectException) is likewise a transport failure -> sentinel 0.
     */
    private function upstreamErrorStatus(GuzzleException $e): int
    {
        if ($e instanceof RequestException) {
            $response = $e->getResponse();
            if ($response !== null && $response->getStatusCode() !== 200) {
                return $response->getStatusCode();
            }
        }

        return 0;
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

    private function sendJsonError(string $message, int $code, ?string $reason = null): void
    {
        header('Content-Type: application/json');
        http_response_code($code);
        $payload = ['error' => $message];
        if ($reason !== null) {
            $payload['reason'] = $reason;
        }
        echo json_encode($payload);
    }
}
