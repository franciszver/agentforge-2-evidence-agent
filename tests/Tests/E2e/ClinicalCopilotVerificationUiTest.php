<?php

/**
 * Clinical Co-Pilot Verification UI E2E Test (P3.8)
 *
 * Exercises the verification-layer UI -- verdict badge, tappable citation
 * chips, and warning banner -- in the real chat panel at both the 360px phone
 * viewport and desktop width.
 *
 * The answer->claims/meds extraction pipeline that would feed a live
 * ``verification`` frame with real data is not built yet (the live frame is
 * the pending, verdict-null payload that renders nothing), so this scenario
 * drives the panel's real, production render path
 * (``window.CopilotChat.renderVerification``, the documented pure-render seam)
 * with representative verification payloads injected via executeScript. This
 * exercises the actual UI component in the actual browser at both viewports --
 * it does not stub or fake the rendering.
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
use Symfony\Component\Panther\PantherTestCase;

class ClinicalCopilotVerificationUiTest extends PantherTestCase
{
    use BaseTrait;
    use LoginTrait;

    private const DEMO_PATIENT_PID = 1;

    /**
     * A "verified" response with one cited claim -- exercises the green
     * verdict badge and a tappable citation chip whose tap reveals the
     * underlying record.
     *
     * @return array<string, mixed>
     */
    private function verifiedPayload(): array
    {
        return [
            'verdict' => 'verified',
            'segments' => [
                [
                    'type' => 'claim',
                    'text' => 'She takes lisinopril 10 mg daily.',
                    'citations' => [
                        [
                            'tool_call_id' => 'call-1',
                            'record_id' => 'med-42',
                            'field' => 'dose',
                            'value' => '10 mg',
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

    /**
     * A "blocked" response carrying a recorded-allergy conflict -- exercises
     * the red verdict badge and the prominent warning banner.
     *
     * @return array<string, mixed>
     */
    private function blockedPayload(): array
    {
        return [
            'verdict' => 'blocked',
            'segments' => [],
            'warnings' => [
                'allergy_conflicts' => [
                    ['medication_name' => 'Ibuprofen', 'allergy_substance' => 'NSAID'],
                ],
                'blocking_interactions' => [],
                'warning_interactions' => [],
            ],
        ];
    }

    #[Test]
    public function testVerificationUiRendersAtPhoneViewport(): void
    {
        $this->base();
        try {
            $this->client->manage()->window()->setSize(new WebDriverDimension(360, 800));
            $this->login(LoginTestData::username, LoginTestData::password);
            $this->openPatientDashboard();
            $this->openChatPanel();

            $this->assertVerificationUiRenders();
        } finally {
            $this->client->quit();
        }
    }

    #[Test]
    public function testVerificationUiRendersAtDesktopViewport(): void
    {
        $this->base();
        try {
            $this->client->manage()->window()->setSize(new WebDriverDimension(1366, 768));
            $this->login(LoginTestData::username, LoginTestData::password);
            $this->openPatientDashboard();
            $this->openChatPanel();

            $this->assertVerificationUiRenders();
        } finally {
            $this->client->quit();
        }
    }

    /**
     * Renders the representative payloads through the panel's real render
     * path and asserts: the verdict badge appears per verdict, a citation
     * chip tap reveals the underlying record, and the warning banner appears
     * on a conflict. Viewport-agnostic so both breakpoints run the same
     * checks.
     */
    private function assertVerificationUiRenders(): void
    {
        // --- Verified case: badge + tappable citation chip. ---
        $this->renderVerification($this->verifiedPayload());

        $this->client->waitFor('.copilot-verdict-badge', 5);
        $badge = $this->client->findElement(WebDriverBy::cssSelector('.copilot-verdict-badge.copilot-verdict-verified'));
        $this->assertStringContainsString(
            'Verified',
            $badge->getText(),
            'the verified verdict badge should render with its text label (not color-only)'
        );

        $record = $this->client->findElement(WebDriverBy::cssSelector('.copilot-citation-record'));
        $this->assertFalse(
            $record->isDisplayed(),
            'the underlying record is hidden until the citation chip is tapped'
        );

        $chip = $this->client->findElement(WebDriverBy::cssSelector('.copilot-citation-chip'));
        $chip->click();
        $this->client->wait(5, 200)->until(
            fn(\Facebook\WebDriver\WebDriver $driver) => $driver->findElement(
                WebDriverBy::cssSelector('.copilot-citation-record')
            )->isDisplayed()
        );
        $this->assertStringContainsString(
            '10 mg',
            $record->getText(),
            'tapping the citation chip reveals the underlying record value'
        );

        // --- Blocked case: red badge + prominent warning banner. ---
        $this->renderVerification($this->blockedPayload());

        $this->client->waitFor('.copilot-warning-banner', 5);
        $banner = $this->client->findElement(WebDriverBy::cssSelector('.copilot-warning-banner'));
        $this->assertTrue($banner->isDisplayed(), 'the warning banner should be visible on a conflict');
        $this->assertStringContainsString(
            'Ibuprofen',
            $banner->getText(),
            'the warning banner should list the conflicting medication'
        );
        $this->assertGreaterThan(
            0,
            count($this->client->findElements(WebDriverBy::cssSelector('.copilot-verdict-badge.copilot-verdict-blocked'))),
            'the blocked verdict badge should render'
        );
    }

    /**
     * Inject a representative verification payload through the panel's real
     * render path (the documented pure-render seam).
     *
     * @param array<string, mixed> $payload
     */
    private function renderVerification(array $payload): void
    {
        $script = <<<'JS'
            var messages = document.getElementById('copilot-chat-messages');
            window.CopilotChat.renderVerification(messages, arguments[0]);
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
