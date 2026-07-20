<?php

/**
 * Module Manager Enable Clinical Co-Pilot Test
 *
 * Drives the Module Manager UI through register -> install -> enable for
 * the Clinical Co-Pilot module and verifies it reaches an enabled state
 * (mod_active=1 in the `modules` table), not merely that it is discoverable.
 *
 * Each step clicks the real UI button, then confirms the persisted database
 * state rather than racing the page's client-side reload - the reload
 * timing after each ajax action is unreliable to observe via polling DOM
 * text under Selenium/Chrome, even though the underlying action completes
 * quickly and reliably server-side.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2025 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Tests\E2e;

use Facebook\WebDriver\WebDriverBy;
use Facebook\WebDriver\WebDriverElement;
use Facebook\WebDriver\WebDriverExpectedCondition;
use OpenEMR\Common\Database\QueryUtils;
use OpenEMR\Tests\E2e\Base\BaseTrait;
use OpenEMR\Tests\E2e\Login\LoginTestData;
use OpenEMR\Tests\E2e\Login\LoginTrait;
use PHPUnit\Framework\Attributes\Test;
use Symfony\Component\Panther\PantherTestCase;

class OfModuleManagerEnableClinicalCopilotTest extends PantherTestCase
{
    use BaseTrait;
    use LoginTrait;

    private const MODULE_ROW_XPATH = "//tr[contains(., 'Clinical Co-Pilot')]";
    private const MODULE_DIRECTORY = 'oe-module-clinical-copilot';
    // Module Manager is mounted at this exact path (no trailing "/index").
    // The page's Action column links use relative URLs (e.g.
    // "./Installer/manage") that resolve against the current path, so a
    // literal "/index" segment would double up to
    // ".../Installer/Installer/manage" and 404.
    //
    // This is an upstream OpenEMR quirk (Installer page emitting
    // relative ajax URLs), not a Co-Pilot bug. Decision (2026-07-16,
    // issue #115): do not file an upstream openemr/openemr issue for
    // it - this repo is a downstream fork focused on the Co-Pilot, and
    // the workaround here (omitting "/index") is proportionate and
    // sufficient. Revisit only if the quirk starts affecting non-E2E
    // code paths.
    private const MODULE_MANAGER_URL = '/interface/modules/zend_modules/public/Installer';

    #[Test]
    public function testModuleEnablesViaModuleManagerWithoutError(): void
    {
        $this->base();
        try {
            $this->login(LoginTestData::username, LoginTestData::password);
            $this->goToModuleManager();

            $moduleRow = $this->getModuleDbRow();
            $this->assertNotFalse(
                $moduleRow,
                'Clinical Co-Pilot module should be auto-registered in the database after visiting Module Manager'
            );
            $this->assertNotEmpty(
                $this->getModuleUiRowText(),
                'Clinical Co-Pilot module should be visible in the Module Manager UI'
            );

            if ($this->toDbInt($moduleRow['sql_run']) === 0) {
                $this->clickModuleButton('Install');
                $this->waitForModuleDbState('sql_run', 1);
                $this->goToModuleManager();
            }

            $moduleRow = $this->getModuleDbRow();
            $this->assertNotFalse(
                $moduleRow,
                'Clinical Co-Pilot module should still be registered in the database after install'
            );
            if ($this->toDbInt($moduleRow['mod_active']) === 0) {
                $this->clickModuleButton('Enable');
                $this->waitForModuleDbState('mod_active', 1);
                $this->goToModuleManager();
            }

            $this->assertStringContainsString(
                'Active',
                $this->getModuleUiRowText(),
                'Clinical Co-Pilot module should show Active status in the Module Manager UI after enabling'
            );

            // Verify no critical error alerts surfaced during the flow.
            $errorElements = $this->client->findElements(
                WebDriverBy::xpath("//div[contains(@class, 'alert-danger')]")
            );
            $this->assertEmpty(
                $errorElements,
                'Module Manager should not show critical error alerts after enabling the module'
            );

            // Final, authoritative check: the enabled state is persisted -
            // the actual acceptance criterion, independent of UI rendering.
            $finalRow = $this->getModuleDbRow();
            $this->assertSame(
                1,
                $finalRow !== false ? $this->toDbInt($finalRow['mod_active']) : -1,
                'Clinical Co-Pilot module should have mod_active=1 in the database after enabling'
            );
        } catch (\Throwable $e) {
            $this->client->quit();
            throw $e;
        }
        $this->client->quit();
    }

    private function goToModuleManager(): void
    {
        $this->client->request('GET', self::MODULE_MANAGER_URL);
        $this->client->waitFor("//span[contains(text(), 'Custom Module Listings')]", 10);
    }

    /**
     * @return array<mixed>|false Row with mod_id, sql_run, mod_active keys
     */
    private function getModuleDbRow(): array|false
    {
        return QueryUtils::querySingleRow(
            "SELECT mod_id, sql_run, mod_active FROM modules WHERE mod_directory = ?",
            [self::MODULE_DIRECTORY]
        );
    }

    private function toDbInt(mixed $value): int
    {
        if (!is_numeric($value)) {
            $this->fail('Expected a numeric module column value, got ' . get_debug_type($value));
        }
        return (int) $value;
    }

    private function getModuleUiRowText(): string
    {
        $rows = $this->client->findElements(WebDriverBy::xpath(self::MODULE_ROW_XPATH));
        return ($rows[0] ?? null)?->getText() ?? '';
    }

    private function clickModuleButton(string $buttonValue): void
    {
        $button = $this->client->wait(10)->until(
            WebDriverExpectedCondition::elementToBeClickable(
                WebDriverBy::xpath(self::MODULE_ROW_XPATH . "//input[@value='{$buttonValue}']")
            )
        );
        if (!$button instanceof WebDriverElement) {
            $this->fail('Expected a clickable WebDriverElement for the "' . $buttonValue . '" button');
        }
        $button->click();
    }

    /**
     * Poll the database directly for the expected column value. The click
     * triggers a server-side ajax action that completes quickly, but the
     * client-side page reload that follows it is not reliably observable
     * via DOM polling, so the database is the synchronization point.
     */
    private function waitForModuleDbState(string $column, int $expectedValue, int $timeoutSeconds = 20): void
    {
        $deadline = microtime(true) + $timeoutSeconds;
        do {
            $row = $this->getModuleDbRow();
            if ($row !== false && $this->toDbInt($row[$column]) === $expectedValue) {
                return;
            }
            usleep(500_000);
        } while (microtime(true) < $deadline);

        $this->fail("Timed out waiting for Clinical Co-Pilot module '{$column}' to become {$expectedValue}");
    }
}
