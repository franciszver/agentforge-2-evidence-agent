<?php

/**
 * Clinical Co-Pilot Token Broker Controller
 *
 * Brokers the OAuth handshake for the Co-Pilot panel (plan §4.1): it
 * authenticates the current OpenEMR session, verifies the CSRF token on every
 * request, and hands the panel the agent base URL plus a bearer token
 * representing the logged-in user.
 *
 * The which-bearer decision is delegated to AgentTokenBroker (#124 Phase 3).
 * With the consent flag OFF the broker mints a signed dev identity token (see
 * DevAgentToken) as the documented stand-in -- unchanged default. With the flag
 * ON it hands back the real per-user OpenEMR access token stored by the Phase 2b
 * consent flow (refreshing/rotating it on demand), or a "consent required"
 * signal that tells the panel to restart the authorize flow. No client secret or
 * password is ever read or returned here.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Controller;

use OpenEMR\Common\Csrf\CsrfUtils;
use OpenEMR\Common\Logging\EventAuditLogger;
use OpenEMR\Common\Session\PatientSessionUtil;
use OpenEMR\Common\Session\SessionWrapperFactory;
use OpenEMR\Core\OEGlobalsBag;
use OpenEMR\Modules\ClinicalCopilot\Auth\AgentTokenBroker;
use OpenEMR\Modules\ClinicalCopilot\Auth\BrokerOutcome;
use OpenEMR\Modules\ClinicalCopilot\Auth\BrokerResult;
use Symfony\Component\HttpFoundation\Session\SessionInterface;

final class TokenBrokerController
{
    public function __construct(
        private readonly AgentTokenBroker $broker,
    ) {
    }

    /**
     * OpenEMR global (OEGlobalsBag) that overrides the agent base URL. The
     * URL is taken only from server configuration - never from request input -
     * so the panel cannot be redirected to an attacker-chosen host.
     */
    private const AGENT_URL_GLOBAL = 'clinical_copilot_agent_url';

    /**
     * Default agent base URL: the docker-network alias of the agent service on
     * the internal copilot network (see docker-compose.copilot.yml).
     */
    private const DEFAULT_AGENT_URL = 'http://agent:8000';

    /** CsrfUtils subject used to derive the per-session token signing key. */
    private const SIGNING_KEY_SUBJECT = 'clinical-copilot-agent-token';

    /** Dev token lifetime in seconds. */
    private const TOKEN_TTL_SECONDS = 3600;

    /**
     * Audit event name recorded when a user opens the Co-Pilot on a chart.
     * Its own category (not one of OpenEMR's built-in event categories) so a
     * "who opened the Co-Pilot on which chart, and when" query is a single
     * predicate on the log table.
     */
    private const AUDIT_EVENT = 'copilot-open';

    public function handleRequest(): void
    {
        header('Content-Type: application/json');

        $method = filter_input(INPUT_SERVER, 'REQUEST_METHOD', FILTER_UNSAFE_RAW, FILTER_REQUIRE_SCALAR);
        if ($method !== 'POST') {
            $this->sendError('Method not allowed', 405);
            return;
        }

        $session = SessionWrapperFactory::getInstance()->getActiveSession();

        $csrfToken = filter_input(INPUT_POST, 'csrf_token_form', FILTER_UNSAFE_RAW, FILTER_REQUIRE_SCALAR);
        if (!CsrfUtils::verifyCsrfToken($csrfToken, $session)) {
            $this->sendError('CSRF verification failed', 403);
            return;
        }

        $rawAuthUserId = $session->get('authUserID');
        $authUserId = is_numeric($rawAuthUserId) ? (int) $rawAuthUserId : 0;
        if ($authUserId <= 0) {
            $this->sendError('Not authenticated', 401);
            return;
        }

        try {
            $result = $this->brokerToken($session, $authUserId);
        } catch (\JsonException) {
            // Guard the one realistic failure (DevAgentToken payload encoding);
            // never surface internal detail to the browser.
            $this->sendError('Unable to broker token', 500);
            return;
        }

        match ($result->outcome) {
            BrokerOutcome::Token => $this->respondWithToken($session, $result),
            BrokerOutcome::ConsentRequired => $this->respondConsentRequired(),
            BrokerOutcome::Error => $this->sendError('Unable to broker token', 500),
        };
    }

    /**
     * Emit the brokered bearer to the panel and record the chart-access audit
     * event (only reached on a genuine, successful token issuance).
     */
    private function respondWithToken(SessionInterface $session, BrokerResult $result): void
    {
        $this->recordChartAccess($session);

        echo json_encode([
            'agent_url' => $this->agentUrl(),
            'token' => $result->token,
        ]);
    }

    /**
     * Signal the panel that consent is required so it redirects the user into
     * the Phase 2b authorize flow. The panel knows its own authorize entry URL,
     * so the body carries only the flag -- never a token or any secret.
     */
    private function respondConsentRequired(): void
    {
        echo json_encode(['consent_required' => true]);
    }

    /**
     * Record the module's chart-access audit event (plan §4.2): who opened
     * the Co-Pilot on which patient chart, and when.
     *
     * The broker is the honest per-open trigger: the panel JS acquires a
     * token lazily and caches it for the panel session, so this fires once
     * when the user actually engages the Co-Pilot to start a conversation --
     * not on every dashboard render -- and it covers both the embedded
     * dashboard panel and the standalone PWA, which share this broker. The
     * audit logger supplies the timestamp. Only reached after authentication,
     * CSRF verification, and a successful token mint, so the open is genuine.
     */
    private function recordChartAccess(SessionInterface $session): void
    {
        $rawUsername = $session->get('authUser');
        $username = is_string($rawUsername) ? $rawUsername : '';

        $rawGroup = $session->get('authProvider');
        $group = is_string($rawGroup) ? $rawGroup : '';

        $pid = PatientSessionUtil::getPid();

        EventAuditLogger::getInstance()->newEvent(
            self::AUDIT_EVENT,
            $username,
            $group,
            1,
            'Opened Clinical Co-Pilot on patient chart',
            $pid > 0 ? $pid : null,
        );
    }

    /**
     * Gather the authenticated session's identity context and delegate the
     * which-bearer decision to the broker. The DevAgentToken signing material is
     * only consumed on the flag-off path; the broker ignores it when the consent
     * flag is on.
     */
    private function brokerToken(SessionInterface $session, int $authUserId): BrokerResult
    {
        $rawUsername = $session->get('authUser');
        $username = is_string($rawUsername) ? $rawUsername : '';

        $signingKey = CsrfUtils::collectCsrfToken($session, self::SIGNING_KEY_SUBJECT);

        return $this->broker->broker(
            $authUserId,
            $username,
            PatientSessionUtil::getPid(),
            $signingKey,
            time(),
            self::TOKEN_TTL_SECONDS,
        );
    }

    private function agentUrl(): string
    {
        $configured = OEGlobalsBag::getInstance()->get(self::AGENT_URL_GLOBAL);

        return is_string($configured) && $configured !== '' ? $configured : self::DEFAULT_AGENT_URL;
    }

    private function sendError(string $message, int $code): void
    {
        http_response_code($code);
        echo json_encode(['error' => $message]);
    }
}
