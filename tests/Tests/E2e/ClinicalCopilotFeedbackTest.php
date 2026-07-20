<?php

/**
 * Clinical Co-Pilot Feedback Buttons E2E Test (P4.4)
 *
 * The DoD scenario: log in, open a patient chart, open the Co-Pilot panel,
 * send a real message, wait for the streamed assistant response and its
 * feedback widget (carrying the P4.1 correlation id delivered on the
 * `conversation` SSE frame -- see app/chat.py), click thumbs-down, and
 * assert the UI reflects a recorded submission (buttons disabled, the
 * down thumb selected, an accessible status message).
 *
 * This test asserts the UI-visible outcome of the round trip through the
 * real dev stack: browser -> public/feedback-proxy.php (P4.4, session +
 * CSRF gated) -> agent POST /feedback (P4.3) -> TraceStore.record_feedback_span
 * (P4.2). It cannot itself query the agent's SQLite trace store
 * (/data/traces.db lives in the `agent` container, on the `copilot_internal`
 * network only, with no shared volume and no docker socket reachable from
 * inside the `openemr` container this test runs in -- see
 * docker-compose.copilot.yml). The correlation id is written to STDERR
 * below so a live run can join it against the trace store directly (e.g.
 * `docker exec development-easy-agent-1 sqlite3 /data/traces.db
 * "SELECT * FROM spans WHERE correlation_id='<id>' AND span_type='feedback'"`)
 * -- the P4.4 task's required live verification, done once outside this
 * automated suite against the rebuilt agent.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Tests\E2e;

use Facebook\WebDriver\WebDriver;
use Facebook\WebDriver\WebDriverBy;
use Facebook\WebDriver\WebDriverDimension;
use OpenEMR\Tests\E2e\Base\BaseTrait;
use OpenEMR\Tests\E2e\Login\LoginTestData;
use OpenEMR\Tests\E2e\Login\LoginTrait;
use PHPUnit\Framework\Attributes\Test;
use Symfony\Component\Panther\PantherTestCase;

class ClinicalCopilotFeedbackTest extends PantherTestCase
{
    use BaseTrait;
    use LoginTrait;

    private const DEMO_PATIENT_PID = 1;

    /**
     * Real 4B-model round trip through the planner loop plus the quarantine
     * summarizer -- same generous ceiling as ClinicalCopilotLiveClinicalAnswerTest.
     */
    private const ASSISTANT_REPLY_TIMEOUT_SECONDS = 180;

    #[Test]
    public function testThumbsDownIsRecordedByTheFeedbackWidget(): void
    {
        $this->base();
        try {
            $this->client->manage()->window()->setSize(new WebDriverDimension(1366, 768));
            $this->login(LoginTestData::username, LoginTestData::password);
            $this->openPatientDashboard();
            $this->openChatPanel();

            $this->sendMessage('What medications is this patient taking?');

            $this->client->wait(self::ASSISTANT_REPLY_TIMEOUT_SECONDS, 500)->until(
                fn(WebDriver $driver) => count($driver->findElements(
                    WebDriverBy::cssSelector('.copilot-feedback')
                )) > 0
            );

            $widget = $this->client->findElement(WebDriverBy::cssSelector('.copilot-feedback'));
            $correlationId = $widget->getAttribute('data-correlation-id');
            $this->assertIsString($correlationId);
            $this->assertNotSame('', $correlationId, 'the feedback widget must carry the response correlation id');

            // Echoed for a live, out-of-band join against the agent's trace
            // store (see this class's docblock) -- not itself an assertion.
            fwrite(STDERR, "\nCOPILOT_FEEDBACK_CORRELATION_ID=" . $correlationId . "\n");

            $downBtn = $this->client->findElement(WebDriverBy::cssSelector('.copilot-feedback-down'));
            $downBtn->click();

            $this->client->wait(15, 200)->until(
                fn(WebDriver $driver) => str_contains(
                    (string) $driver->findElement(WebDriverBy::cssSelector('.copilot-feedback'))->getAttribute('class'),
                    'copilot-feedback-submitted'
                )
            );

            $this->assertNotNull(
                $downBtn->getAttribute('disabled'),
                'the down button must disable after a successful submit (no double-submit)'
            );
            $upBtn = $this->client->findElement(WebDriverBy::cssSelector('.copilot-feedback-up'));
            $this->assertNotNull($upBtn->getAttribute('disabled'), 'the up button must disable too, not just the clicked one');

            $this->assertStringContainsString(
                'copilot-feedback-selected',
                (string) $downBtn->getAttribute('class'),
                'the clicked (down) thumb must be visibly marked as the recorded choice'
            );

            $status = $this->client->findElement(WebDriverBy::cssSelector('.copilot-feedback-status'));
            $this->assertStringContainsStringIgnoringCase(
                'thanks',
                $status->getText(),
                'an accessible status message must confirm the feedback was recorded'
            );
        } finally {
            $this->client->quit();
        }
    }

    private function openPatientDashboard(): void
    {
        $this->client->request('GET', '/interface/patient_file/summary/demographics.php?set_pid=' . self::DEMO_PATIENT_PID);
    }

    private function openChatPanel(): void
    {
        $this->client->waitFor("//*[@id='copilot-open-chat-btn']", 15);
        $button = $this->client->findElement(WebDriverBy::id('copilot-open-chat-btn'));
        $button->click();
        $this->client->waitFor("//*[@id='copilot-chat-panel' and not(contains(@class,'copilot-hidden'))]", 10);
    }

    private function sendMessage(string $text): void
    {
        $input = $this->client->findElement(WebDriverBy::id('copilot-chat-input'));
        $input->sendKeys($text);
        $send = $this->client->findElement(WebDriverBy::id('copilot-chat-send-btn'));
        $send->click();
    }
}
