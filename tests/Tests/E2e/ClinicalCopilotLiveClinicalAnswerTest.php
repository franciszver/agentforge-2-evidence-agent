<?php

/**
 * Clinical Co-Pilot Live Clinical Answer E2E Test (issue #126, finding F4)
 *
 * The end-to-end proof that ONE clinical use case answers GENUINELY through
 * the real browser UI with real clinical data. Logs in, opens demo patient
 * Phil Belford's chart (pid 1), opens the Co-Pilot panel, asks the UC2
 * medication-list question, and asserts the streamed assistant answer names
 * an ACTUAL medication from Phil's record (Lisinopril / Norvasc) -- a canary
 * proving genuine retrieval through a real OpenEMR token, not an auth error
 * and not an empty answer.
 *
 * Before the dev-token bridge (this issue) the runtime path could not fetch
 * any clinical data: the browser's DevAgentToken is an identity assertion,
 * not a real OpenEMR token, so tool calls auth-failed and the answer could
 * never contain a real record value. This test fails red on that path and
 * passes once the agent obtains a real OpenEMR token server-side.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Tests\E2e;

use Facebook\WebDriver\WebDriverBy;
use Facebook\WebDriver\WebDriverDimension;
use OpenEMR\Tests\E2e\Base\BaseTrait;
use OpenEMR\Tests\E2e\Login\LoginTestData;
use OpenEMR\Tests\E2e\Login\LoginTrait;
use PHPUnit\Framework\Attributes\Test;
use Symfony\Component\DomCrawler\Crawler;
use Symfony\Component\Panther\PantherTestCase;

class ClinicalCopilotLiveClinicalAnswerTest extends PantherTestCase
{
    use BaseTrait;
    use LoginTrait;

    private ?Crawler $crawler = null;

    private const DEMO_PATIENT_PID = 1;

    /**
     * Real medications on demo patient Phil Belford's record (lists table,
     * type='medication'): both active. The live answer must surface at least
     * one of these verbatim -- the canary that retrieval genuinely reached
     * OpenEMR with a real, resource-scoped token.
     */
    private const REAL_MEDICATIONS = ['lisinopril', 'norvasc'];

    /**
     * Real 4B-model round trip through the planner loop plus the quarantine
     * summarizer -- generous headroom for CI/dev-box variance (same ceiling
     * as ClinicalCopilotChatPanelTest).
     */
    private const ASSISTANT_REPLY_TIMEOUT_SECONDS = 180;

    #[Test]
    public function testAnswerContainsRealMedicationFromPatientRecord(): void
    {
        $this->base();
        try {
            $this->client->manage()->window()->setSize(new WebDriverDimension(1366, 768));
            $this->login(LoginTestData::username, LoginTestData::password);
            $this->openPatientDashboard();
            $this->openChatPanel();

            $this->sendMessage('What medications is this patient taking?');

            $this->client->wait(self::ASSISTANT_REPLY_TIMEOUT_SECONDS, 500)->until(
                fn(\Facebook\WebDriver\WebDriver $driver) => count($driver->findElements(
                    WebDriverBy::xpath("//*[contains(@class,'copilot-chat-message-assistant')]")
                )) > 0
            );
            $assistantBubble = $this->client->findElement(
                WebDriverBy::xpath("//*[contains(@class,'copilot-chat-message-assistant')]")
            );
            $answer = strtolower(trim($assistantBubble->getText()));

            $this->assertNotSame('', $answer, 'a real assistant answer should have streamed in');
            $this->assertStringNotContainsStringIgnoringCase(
                'error',
                $answer,
                'the answer must be a genuine clinical answer, not an auth/tool error'
            );

            $matched = false;
            foreach (self::REAL_MEDICATIONS as $medication) {
                if (str_contains($answer, $medication)) {
                    $matched = true;
                    break;
                }
            }
            $this->assertTrue(
                $matched,
                'the streamed answer must name a real medication from the patient record ('
                . implode(' or ', self::REAL_MEDICATIONS) . '); got: ' . $answer
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
