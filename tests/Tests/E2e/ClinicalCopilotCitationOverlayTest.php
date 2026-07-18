<?php

/**
 * Clinical Co-Pilot Citation Overlay E2E Test (P3.7)
 *
 * Exercises the document-citation "click-to-source" chip in the real chat
 * panel: tapping it reveals the cited page/quote and a real `<a>` link to
 * the source PDF at the cited page.
 *
 * Same scoping note as ClinicalCopilotVerificationUiTest (P3.8): the
 * answer->claims pipeline that would feed a live `verification` frame
 * carrying real `document_citations` is not wired into the live planner yet
 * (see app.rendering/app.extraction module docstrings and the P3.7 PR
 * description) -- `Claim.document_citations` is populated by tests only
 * today, never by a real chat turn. So this drives the panel's real,
 * production render path (`window.CopilotChat.renderVerification`, the same
 * documented pure-render seam ClinicalCopilotVerificationUiTest uses) with a
 * representative payload injected via executeScript, in the real browser at
 * the real DOM level -- it exercises the actual `<a>` element's real href/
 * target/rel attributes and the actual click-to-reveal behavior, which a
 * jsdom-based Jest test (tests/js/clinical-copilot-citation-overlay.test.js)
 * cannot fully confirm.
 *
 * Deliberately NOT exercised here: the source PDF actually opening/rendering
 * behind the link (that would need a real ingested document_store, wired
 * through source-doc-proxy.php to a real agent GET /documents/{source_id} --
 * out of scope until the ingestion-to-chat wiring above lands). This test
 * only confirms the link's own attributes are correct, not the page it
 * points to.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <ciscodg@gmail.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Tests\E2e;

use Facebook\WebDriver\WebDriver;
use Facebook\WebDriver\WebDriverBy;
use OpenEMR\Tests\E2e\Base\BaseTrait;
use OpenEMR\Tests\E2e\Login\LoginTestData;
use OpenEMR\Tests\E2e\Login\LoginTrait;
use PHPUnit\Framework\Attributes\Test;
use Symfony\Component\Panther\PantherTestCase;

class ClinicalCopilotCitationOverlayTest extends PantherTestCase
{
    use BaseTrait;
    use LoginTrait;

    private const DEMO_PATIENT_PID = 1;

    /**
     * A "verified" response whose sole claim is backed by a document
     * citation (P3.1/P3.6 shape) rather than a structured-tool SourceRef --
     * exercises the document-citation chip and its view-source link.
     *
     * @return array<string, mixed>
     */
    private function documentCitedPayload(): array
    {
        return [
            'verdict' => 'verified',
            'segments' => [
                [
                    'type' => 'claim',
                    'text' => 'Glucose is 105 mg/dL.',
                    'citations' => [],
                    'document_citations' => [
                        [
                            'source_type' => 'lab_pdf',
                            'source_id' => str_repeat('a1b2', 8), // 32 hex chars
                            'page_or_section' => 'page 2',
                            'field_or_chunk_id' => 'Glucose',
                            'quote_or_value' => 'Glucose: 105 mg/dL',
                        ],
                    ],
                ],
            ],
            'warnings' => [
                'allergy_conflicts' => [],
                'blocking_interactions' => [],
                'warning_interactions' => [],
            ],
        ];
    }

    #[Test]
    public function testDocumentCitationChipRevealsQuoteAndSourcePageLink(): void
    {
        $this->base();
        try {
            $this->login(LoginTestData::username, LoginTestData::password);
            $this->openPatientDashboard();
            $this->openChatPanel();

            $this->renderVerification($this->documentCitedPayload());

            $this->client->waitFor('.copilot-document-citation-chip', 5);
            $chip = $this->client->findElement(WebDriverBy::cssSelector('.copilot-document-citation-chip'));
            $this->assertSame('Glucose', $chip->getText());

            $record = $this->client->findElement(WebDriverBy::cssSelector('.copilot-citation-record'));
            $this->assertFalse($record->isDisplayed(), 'the citation record is hidden until the chip is tapped');

            $chip->click();
            $this->client->wait(5, 200)->until(
                fn(WebDriver $driver) => $driver->findElement(
                    WebDriverBy::cssSelector('.copilot-citation-record')
                )->isDisplayed()
            );

            $this->assertStringContainsString(
                'Glucose: 105 mg/dL',
                $record->getText(),
                'the revealed record shows the citation\'s own literal quote -- never a fabricated value'
            );
            $this->assertStringContainsString('page 2', $record->getText());

            $link = $this->client->findElement(WebDriverBy::cssSelector('.copilot-citation-source-link'));
            $this->assertStringContainsString(
                'source_id=' . str_repeat('a1b2', 8),
                $link->getAttribute('href') ?? '',
            );
            $this->assertStringContainsString(
                '#page=2',
                $link->getAttribute('href') ?? '',
                'the link opens the real source PDF at the cited page -- the honest page-level fallback, never a fabricated bounding box (see app/documents.py\'s module docstring for the capability decision)'
            );
            $this->assertSame('_blank', $link->getAttribute('target'));
            $this->assertStringContainsString('noopener', $link->getAttribute('rel') ?? '');
        } finally {
            $this->client->quit();
        }
    }

    /**
     * Inject a representative verification payload through the panel's real
     * render path (same seam ClinicalCopilotVerificationUiTest uses), with a
     * source-doc-proxy base URL matching what initFromDom wires in
     * production.
     *
     * @param array<string, mixed> $payload
     */
    private function renderVerification(array $payload): void
    {
        $script = <<<'JS'
            var messages = document.getElementById('copilot-chat-messages');
            window.CopilotChat.renderVerification(messages, arguments[0], 'source-doc-proxy.php');
            JS;
        $this->client->executeScript($script, [$payload]);
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
}
