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

use OpenEMR\Tests\E2e\Base\BaseTrait;
use OpenEMR\Tests\E2e\Login\LoginTestData;
use OpenEMR\Tests\E2e\Login\LoginTrait;
use PHPUnit\Framework\Attributes\Test;
use Symfony\Component\DomCrawler\Crawler;
use Symfony\Component\Panther\PantherTestCase;

class ClinicalCopilotTokenBrokerTest extends PantherTestCase
{
    use BaseTrait;
    use LoginTrait;

    private ?Crawler $crawler = null;

    private const DEMO_PATIENT_PID = 1;

    private const BROKER_PATH = '/interface/modules/custom_modules/oe-module-clinical-copilot/public/ajax.php';

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
}
