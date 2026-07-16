<?php

/**
 * Clinical Co-Pilot Patient Dashboard UI Injection Test
 *
 * Verifies the Clinical Co-Pilot module injects a Co-Pilot card (via
 * PatientDemographics RenderEvent::EVENT_SECTION_LIST_RENDER_BEFORE) and a
 * persistent open-chat button (via PageHeadingRenderEvent) onto the patient
 * dashboard, and that
 * both carry the current session context (pid, encounter, authUserID)
 * escaped into the page for the panel JS to read. Runs at both a desktop
 * and a phone viewport since the elements must render at both.
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
use OpenEMR\Common\Database\QueryUtils;
use OpenEMR\Tests\E2e\Base\BaseTrait;
use OpenEMR\Tests\E2e\Login\LoginTestData;
use OpenEMR\Tests\E2e\Login\LoginTrait;
use PHPUnit\Framework\Attributes\Test;
use Symfony\Component\Panther\PantherTestCase;

class ClinicalCopilotPatientDashboardUiTest extends PantherTestCase
{
    use BaseTrait;
    use LoginTrait;

    private const DEMO_PATIENT_PID = 1;

    private const CARD_XPATH = "//*[@id='copilot-card']";

    private const BUTTON_XPATH = "//*[@id='copilot-open-chat-btn']";

    #[Test]
    public function testCopilotCardAndOpenChatButtonRenderWithSessionContextAtDesktopViewport(): void
    {
        $this->runScenario(1366, 768);
    }

    #[Test]
    public function testCopilotCardAndOpenChatButtonRenderWithSessionContextAtPhoneViewport(): void
    {
        $this->runScenario(360, 640);
    }

    private function runScenario(int $width, int $height): void
    {
        $this->base();
        try {
            $this->client->manage()->window()->setSize(new WebDriverDimension($width, $height));
            $this->login(LoginTestData::username, LoginTestData::password);
            $this->openPatientDashboard(self::DEMO_PATIENT_PID);

            $this->client->waitFor(self::CARD_XPATH, 15);
            $card = $this->client->findElement(WebDriverBy::xpath(self::CARD_XPATH));
            $this->assertSame(
                (string) self::DEMO_PATIENT_PID,
                $card->getAttribute('data-copilot-pid'),
                'Co-Pilot card should carry the current patient pid'
            );

            $this->client->waitFor(self::BUTTON_XPATH, 15);
            $button = $this->client->findElement(WebDriverBy::xpath(self::BUTTON_XPATH));
            $this->assertTrue($button->isDisplayed(), 'Open-chat button should be visible in the page heading');

            $context = $this->client->executeScript('return window.CopilotContext;');
            $this->assertIsArray($context, 'window.CopilotContext should be defined by the injected panel script');
            // The server emits these as JSON integers, so they arrive as PHP ints.
            $this->assertSame(
                self::DEMO_PATIENT_PID,
                $context['pid'] ?? null,
                'window.CopilotContext.pid should match the opened patient'
            );

            $adminUserId = $this->getAdminUserId();
            $this->assertSame(
                $adminUserId,
                $context['authUserID'] ?? null,
                'window.CopilotContext.authUserID should match the logged-in user'
            );
        } finally {
            $this->client->quit();
        }
    }

    private function openPatientDashboard(int $pid): void
    {
        $this->client->request('GET', '/interface/patient_file/summary/demographics.php?set_pid=' . $pid);
    }

    private function getAdminUserId(): int
    {
        $row = QueryUtils::querySingleRow('SELECT id FROM users WHERE username = ?', [LoginTestData::username]);
        $id = is_array($row) ? ($row['id'] ?? null) : null;
        if (!is_numeric($id)) {
            $this->fail('Could not resolve admin user id from users table');
        }
        return (int) $id;
    }
}
