<?php

/**
 * Clinical Co-Pilot Token Broker E2E Test
 *
 * Exercises the P2.13 token-broker endpoint end-to-end against the running
 * dev stack. The broker (public/ajax.php) is CSRF-gated and session-scoped:
 * a logged-in panel acquires a dev bearer token + the agent base URL, while
 * a tampered CSRF token or a non-POST method is rejected with no token in the
 * response body.
 *
 * The CSRF rejection / wrong-method cases are exercised here (not as an
 * isolated unit test) because ajax.php bootstraps globals.php, which requires
 * a live session and database; the pure token-minting logic is unit-tested in
 * DevAgentTokenTest.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Tests\E2e;

use OpenEMR\Common\Database\QueryUtils;
use OpenEMR\Tests\E2e\Base\BaseTrait;
use OpenEMR\Tests\E2e\Login\LoginTestData;
use OpenEMR\Tests\E2e\Login\LoginTrait;
use PHPUnit\Framework\Attributes\Test;
use Symfony\Component\Panther\PantherTestCase;

class ClinicalCopilotTokenBrokerTest extends PantherTestCase
{
    use BaseTrait;
    use LoginTrait;

    private const DEMO_PATIENT_PID = 1;

    private const BROKER_PATH = '/interface/modules/custom_modules/oe-module-clinical-copilot/public/ajax.php';

    /** Audit event name the broker records when a chart's Co-Pilot is opened. */
    private const AUDIT_EVENT = 'copilot-open';

    #[Test]
    public function testBrokerBehaviour(): void
    {
        $this->base();
        try {
            $this->login(LoginTestData::username, LoginTestData::password);
            $this->client->request('GET', '/interface/patient_file/summary/demographics.php?set_pid=' . self::DEMO_PATIENT_PID);
            $this->client->waitFor("//*[@id='copilot-open-chat-btn']", 15);

            $csrfToken = $this->client->executeScript('return window.CopilotContext ? window.CopilotContext.csrfToken : null;');
            $this->assertIsString($csrfToken, 'panel context must expose a CSRF token for the broker');
            $this->assertNotSame('', $csrfToken, 'CSRF token must be non-empty');

            // Snapshot the log high-water mark so the audit assertion only
            // sees the row this test's broker call writes.
            $maxLogIdBefore = $this->maxLogId();

            // Positive: a valid CSRF token yields a token + agent URL.
            $ok = $this->callBroker('POST', $csrfToken);
            $this->assertSame(200, $ok['status'], 'valid CSRF POST should succeed');
            $okBody = json_decode((string) $ok['body'], true);
            $this->assertIsArray($okBody);
            $this->assertArrayNotHasKey('error', $okBody);
            $this->assertIsString($okBody['token'] ?? null, 'broker must return a bearer token');
            $this->assertNotSame('', $okBody['token'], 'token must be non-empty');
            $this->assertIsString($okBody['agent_url'] ?? null, 'broker must return the agent base URL');
            $this->assertNotSame('', $okBody['agent_url'], 'agent URL must be non-empty');

            // Chart-access audit trail (P2.17): a successful open records a
            // copilot-open event naming WHO (the logged-in user) and WHICH
            // patient (the panel pid), timestamped by the audit logger.
            $auditRow = $this->latestChartAccessEvent($maxLogIdBefore);
            $this->assertNotNull($auditRow, 'broker must record a copilot-open audit event on a successful open');
            $this->assertSame(LoginTestData::username, $auditRow['user'], 'audit event must name the logged-in user');
            $this->assertSame(self::DEMO_PATIENT_PID, $auditRow['patient_id'], 'audit event must name the opened patient');
            $this->assertSame('1', $auditRow['success'], 'a successful open must be logged as success');

            // Negative: a tampered CSRF token is rejected with no token leaked.
            $bad = $this->callBroker('POST', $csrfToken . 'tampered');
            $this->assertSame(403, $bad['status'], 'tampered CSRF must be rejected');
            $badBody = json_decode((string) $bad['body'], true);
            $this->assertIsArray($badBody);
            $this->assertArrayNotHasKey('token', $badBody, 'no token material in a rejected response');

            // Negative: a non-POST method is rejected.
            $get = $this->callBroker('GET', $csrfToken);
            $this->assertSame(405, $get['status'], 'non-POST method must be rejected');
            $getBody = json_decode((string) $get['body'], true);
            $this->assertIsArray($getBody);
            $this->assertArrayNotHasKey('token', $getBody, 'no token material in a rejected response');
        } finally {
            $this->client->quit();
        }
    }

    /**
     * Issue a same-origin request to the broker from the logged-in page
     * context (so the session cookie is attached) and return its HTTP status
     * and raw body.
     *
     * @return array{status: int, body: string}
     */
    private function callBroker(string $method, string $csrfToken): array
    {
        $script = <<<'JS'
            var url = arguments[0];
            var method = arguments[1];
            var csrf = arguments[2];
            var xhr = new XMLHttpRequest();
            xhr.open(method, url, false);
            try {
                if (method === 'POST') {
                    xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
                    xhr.send('action=get_token&csrf_token_form=' + encodeURIComponent(csrf));
                } else {
                    xhr.send(null);
                }
            } catch (e) {
                return JSON.stringify({ status: -1, body: String(e) });
            }
            return JSON.stringify({ status: xhr.status, body: xhr.responseText });
            JS;

        $raw = $this->client->executeScript($script, [self::BROKER_PATH, $method, $csrfToken]);
        $decoded = json_decode(is_string($raw) ? $raw : '', true);

        $status = is_array($decoded) && is_int($decoded['status'] ?? null) ? $decoded['status'] : 0;
        $body = is_array($decoded) && is_string($decoded['body'] ?? null) ? $decoded['body'] : '';

        return ['status' => $status, 'body' => $body];
    }

    /**
     * Current maximum id in the audit `log` table, used as a high-water mark
     * so the audit assertion ignores rows written before the broker call.
     */
    private function maxLogId(): int
    {
        $row = QueryUtils::querySingleRow('SELECT COALESCE(MAX(id), 0) AS max_id FROM log');
        $maxId = is_array($row) ? ($row['max_id'] ?? 0) : 0;

        return is_numeric($maxId) ? (int) $maxId : 0;
    }

    /**
     * The copilot-open audit event written after the given log id, if any.
     *
     * @return array{user: string, patient_id: int, success: string}|null
     */
    private function latestChartAccessEvent(int $afterLogId): ?array
    {
        $row = QueryUtils::querySingleRow(
            'SELECT `user`, `patient_id`, `success` FROM `log`'
            . ' WHERE `event` = ? AND `id` > ? ORDER BY `id` DESC LIMIT 1',
            [self::AUDIT_EVENT, $afterLogId]
        );
        if (!is_array($row)) {
            return null;
        }

        $patientId = $row['patient_id'] ?? null;
        $success = $row['success'] ?? null;

        return [
            'user' => is_string($row['user']) ? $row['user'] : '',
            'patient_id' => is_numeric($patientId) ? (int) $patientId : 0,
            'success' => is_scalar($success) ? (string) $success : '',
        ];
    }
}
