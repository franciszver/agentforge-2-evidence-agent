<?php

/**
 * Clinical Co-Pilot Source Document Proxy Controller (P3.7)
 *
 * Same-origin bridge for the citation-overlay "View source page" link: the
 * agent service sits on the `copilot_internal` docker network only (no host
 * port, no browser-reachable URL -- see docker-compose.copilot.yml), so the
 * browser cannot call its `GET /documents/{source_id}` (app/documents.py)
 * directly. This controller relays that call through the OpenEMR origin
 * instead, exactly like ChatProxyController/FeedbackProxyController.
 *
 * Unlike those two, this is a plain browser navigation (the citation chip's
 * link is a real `<a target="_blank">`, opened in a new tab so the module
 * chat panel is never blocked/replaced by the PDF view) -- there is no
 * fetch() call available to attach a bearer the panel already brokered, and
 * no request body to carry a CSRF token in. So this controller mints its
 * OWN bearer server-side via AgentTokenBroker (the SAME broker
 * TokenBrokerController uses), keyed off the authenticated session, rather
 * than depending on browser-held state -- the session cookie IS the only
 * credential a plain navigation can carry.
 *
 * No CSRF gate: this is a read-only GET with no side effect (nothing is
 * mutated on the agent or in OpenEMR), so there is nothing for a
 * state-changing CSRF attack to trigger; the Same-Origin Policy stops a
 * cross-origin page from reading the response either way. Auth is still
 * required -- an unauthenticated session gets 401 before anything else runs.
 *
 * **Access control.** `source_id` is re-validated against
 * `LocalIngestionStore`'s own uuid4().hex naming (32 lowercase hex chars)
 * here too, even though `app.documents`'s endpoint already gates on the same
 * pattern -- defense in depth, same discipline as every other
 * two-independent-checks guard in this codebase. A malformed value is
 * rejected with 400 before any upstream call is made.
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
use GuzzleHttp\Exception\RequestException;
use OpenEMR\BC\ServiceContainer;
use OpenEMR\Common\Csrf\CsrfUtils;
use OpenEMR\Common\Session\PatientSessionUtil;
use OpenEMR\Common\Session\SessionWrapperFactory;
use OpenEMR\Core\OEGlobalsBag;
use OpenEMR\Modules\ClinicalCopilot\Auth\AgentTokenBroker;
use OpenEMR\Modules\ClinicalCopilot\Auth\BrokerOutcome;
use OpenEMR\Modules\ClinicalCopilot\Auth\BrokerResult;
use OpenEMR\Modules\ClinicalCopilot\Auth\GuzzleAuthorizationCodeExchanger;
use OpenEMR\Modules\ClinicalCopilot\Auth\OAuthConsentConfig;
use OpenEMR\Modules\ClinicalCopilot\Auth\QueryUtilsTokenStorageGateway;
use OpenEMR\Modules\ClinicalCopilot\Auth\UserOAuthTokenRepository;
use Symfony\Component\HttpFoundation\Session\SessionInterface;

final class SourceDocumentProxyController
{
    /** Matches LocalIngestionStore's own uuid4().hex naming exactly. */
    private const SOURCE_ID_PATTERN = '/^[0-9a-f]{32}$/';

    /** CsrfUtils subject used to derive the per-session dev-token signing key. Matches TokenBrokerController. */
    private const SIGNING_KEY_SUBJECT = 'clinical-copilot-agent-token';

    private const TOKEN_TTL_SECONDS = 3600;

    private const AGENT_URL_GLOBAL = 'clinical_copilot_agent_url';
    private const DEFAULT_AGENT_URL = 'http://agent:8000';

    private const UPSTREAM_TIMEOUT_SECONDS = 15;

    public function __construct(
        private readonly ?AgentTokenBroker $broker = null,
        /** Request method override for tests -- see FeedbackProxyControllerTest for why. */
        private readonly ?string $requestMethod = null,
    ) {
    }

    public function handleRequest(): void
    {
        $method = $this->requestMethod
            ?? filter_input(INPUT_SERVER, 'REQUEST_METHOD', FILTER_UNSAFE_RAW, FILTER_REQUIRE_SCALAR);
        if ($method !== 'GET') {
            $this->sendError('Method not allowed', 405);
            return;
        }

        $session = SessionWrapperFactory::getInstance()->getActiveSession();

        $rawAuthUserId = $session->get('authUserID');
        $authUserId = is_numeric($rawAuthUserId) ? (int) $rawAuthUserId : 0;
        if ($authUserId <= 0) {
            $this->sendError('Not authenticated', 401);
            return;
        }

        $sourceId = filter_input(INPUT_GET, 'source_id', FILTER_UNSAFE_RAW, FILTER_REQUIRE_SCALAR);
        if (!is_string($sourceId) || !preg_match(self::SOURCE_ID_PATTERN, $sourceId)) {
            $this->sendError('Invalid source_id', 400);
            return;
        }

        $result = $this->brokerToken($session, $authUserId);

        match ($result->outcome) {
            BrokerOutcome::Token => $this->streamDocument($sourceId, (string) $result->token),
            BrokerOutcome::ConsentRequired => $this->redirectToAuthorize(),
            BrokerOutcome::Error => $this->sendError('Unable to broker token', 500),
        };
    }

    private function streamDocument(string $sourceId, string $token): void
    {
        $client = new Client();

        try {
            $response = $client->get(rtrim($this->agentUrl(), '/') . '/documents/' . $sourceId, [
                'headers' => ['Authorization' => 'Bearer ' . $token],
                'timeout' => self::UPSTREAM_TIMEOUT_SECONDS,
            ]);
        } catch (RequestException $e) {
            $upstreamResponse = $e->getResponse();
            $status = $upstreamResponse !== null ? $upstreamResponse->getStatusCode() : 502;
            $this->sendError('Document not available', $status === 404 ? 404 : 502);
            return;
        } catch (GuzzleException) {
            $this->sendError('Document service unavailable', 502);
            return;
        }

        header('Content-Type: application/pdf');
        header('X-Content-Type-Options: nosniff');
        echo $response->getBody()->getContents();
    }

    private function redirectToAuthorize(): void
    {
        // Same destination the panel JS sends the user to when the token
        // broker signals consent_required (createChatController.ensureToken).
        header('Location: ' . 'oauth-authorize.php');
        http_response_code(302);
    }

    private function brokerToken(SessionInterface $session, int $authUserId): BrokerResult
    {
        $rawUsername = $session->get('authUser');
        $username = is_string($rawUsername) ? $rawUsername : '';
        $signingKey = CsrfUtils::collectCsrfToken($session, self::SIGNING_KEY_SUBJECT);

        return $this->resolveBroker()->broker(
            $authUserId,
            $username,
            PatientSessionUtil::getPid(),
            $signingKey,
            time(),
            self::TOKEN_TTL_SECONDS,
        );
    }

    /** Lazily builds the production broker (same wiring as public/ajax.php) unless a test double was injected. */
    private function resolveBroker(): AgentTokenBroker
    {
        if ($this->broker !== null) {
            return $this->broker;
        }

        $config = OAuthConsentConfig::fromEnvironment();
        $globals = OEGlobalsBag::getInstance();
        $verifySslRaw = $globals->get('clinical_copilot_oauth_verify_ssl');
        $verifySsl = ($verifySslRaw === null || $verifySslRaw === '')
            ? true
            : $globals->getBoolean('clinical_copilot_oauth_verify_ssl');

        return new AgentTokenBroker(
            $config,
            new GuzzleAuthorizationCodeExchanger($config, $verifySsl),
            new UserOAuthTokenRepository(ServiceContainer::getCrypto(), new QueryUtilsTokenStorageGateway()),
            $globals->getBoolean('database_encryption'),
        );
    }

    private function agentUrl(): string
    {
        $configured = OEGlobalsBag::getInstance()->get(self::AGENT_URL_GLOBAL);

        return is_string($configured) && $configured !== '' ? $configured : self::DEFAULT_AGENT_URL;
    }

    private function sendError(string $message, int $code): void
    {
        header('Content-Type: application/json');
        http_response_code($code);
        echo json_encode(['error' => $message]);
    }
}
