<?php

/**
 * BaseTrait trait
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Brady Miller <brady.g.miller@gmail.com>
 * @author    Michael A. Smith <michael@opencoreemr.com>
 * @copyright Copyright (c) 2024 Brady Miller <brady.g.miller@gmail.com>
 * @copyright Copyright (c) 2025-2026 OpenCoreEMR Inc <https://opencoreemr.com/>
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Tests\E2e\Base;

use Facebook\WebDriver\Exception\Internal\UnexpectedResponseException;
use Facebook\WebDriver\Exception\StaleElementReferenceException;
use Facebook\WebDriver\Exception\TimeoutException;
use Facebook\WebDriver\Exception\UnexpectedAlertOpenException;
use Facebook\WebDriver\Exception\WebDriverException;
use Facebook\WebDriver\JavaScriptExecutor;
use Facebook\WebDriver\Remote\DesiredCapabilities;
use Facebook\WebDriver\WebDriver;
use Facebook\WebDriver\WebDriverBy;
use Facebook\WebDriver\WebDriverElement;
use Facebook\WebDriver\WebDriverExpectedCondition;
use OpenEMR\Common\Database\QueryUtils;
use OpenEMR\Tests\E2e\Xpaths\XpathsConstants;
use Symfony\Component\Panther\Client;
use Symfony\Component\Panther\DomCrawler\Crawler as PantherCrawler;

trait BaseTrait
{
    private Client $client;

    /**
     * Set on every navigation/interaction that returns a crawler (login,
     * requests, refreshCrawler). Declared here so every E2E class composing
     * this trait gets a real typed property instead of a per-class dynamic
     * property (PHP 8.2+ deprecation) or a per-class PHPStan baseline entry.
     *
     * Typed as Panther's Crawler (not the base DomCrawler\Crawler) because
     * Client::request()/refreshCrawler() both declare that narrower return
     * type, and callers rely on Panther-only methods like getElement().
     */
    private PantherCrawler $crawler;

    private function base(): void
    {
        // getenv() returns string|false (never null), so a default is applied
        // via an explicit false-check rather than ?? (which PHPStan flags as
        // meaningless on a non-nullable left side).
        $useGridEnv = getenv("SELENIUM_USE_GRID", true);
        $useGrid = $useGridEnv !== false ? $useGridEnv : "false";

        if ($useGrid === "true") {
            // Use Selenium Grid (consistent testing environment with goal of stability)
            $seleniumHostEnv = getenv("SELENIUM_HOST", true);
            $seleniumHost = $seleniumHostEnv !== false ? $seleniumHostEnv : "selenium";
            $e2eBaseUrl = getenv("SELENIUM_BASE_URL", true) ?: "http://openemr";
            $forceHeadlessEnv = getenv("SELENIUM_FORCE_HEADLESS", true);
            $forceHeadless = $forceHeadlessEnv !== false ? $forceHeadlessEnv : "false";
            // Implicit wait must be 0 when using explicit waits (waitFor,
            // waitForVisibility, wait()->until()). A non-zero implicit wait
            // causes each findElement() call inside an explicit wait condition
            // to block for the full implicit wait duration before throwing,
            // consuming the entire explicit wait timeout in a single attempt
            // instead of retrying.
            $implicitWait = (int)(getenv("SELENIUM_IMPLICIT_WAIT") ?: 0);
            $pageLoadTimeout = (int)(getenv("SELENIUM_PAGE_LOAD_TIMEOUT") ?: 60);

            $capabilities = DesiredCapabilities::chrome();

            $chromeArgs = [
                '--window-size=1920,1080',  // Matches SE_SCREEN_WIDTH/HEIGHT
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu'
            ];

            // Add headless if forced (but VNC won't work in headless mode)
            if ($forceHeadless === "true") {
                $chromeArgs[] = '--headless';
            }

            $capabilities->setCapability('goog:chromeOptions', [
                'args' => $chromeArgs
            ]);

            $capabilities->setCapability('unhandledPromptBehavior', 'accept');
            $capabilities->setCapability('pageLoadStrategy', 'normal');

            $seleniumUrl = "http://$seleniumHost:4444/wd/hub";
            $this->client = Client::createSeleniumClient($seleniumUrl, $capabilities, $e2eBaseUrl);

            $this->client->manage()->timeouts()->implicitlyWait($implicitWait);
            $this->client->manage()->timeouts()->pageLoadTimeout($pageLoadTimeout);
        } else {
            // Use local ChromeDriver (not a consistent testing environment, which is thus not stable, good luck :) )
            $this->client = static::createPantherClient(['external_base_uri' => "http://localhost"]);
            $this->client->manage()->window()->maximize();
        }
    }

    /**
     * Wait for the application to be fully initialized after login.
     *
     * Verifies Knockout.js has applied bindings by checking that the
     * #mainMenu div has children (rendered by the menu template).
     * Without this gate, tests that immediately navigate menus can
     * fail because the page HTML loaded but the JS framework hasn't
     * finished rendering.
     *
     * @param int $timeout Seconds to wait before giving up
     * @return bool True if app initialized, false if timeout
     */
    private function waitForAppReady(int $timeout = 30): bool
    {
        try {
            $this->client->wait($timeout)->until(
                fn(WebDriver&JavaScriptExecutor $driver) => $driver->executeScript(
                    'return document.getElementById("mainMenu")?.children.length > 0'
                )
            );
            // Log state on success to verify hypothesis that koAvailable
            // is always true when the menu renders successfully
            $state = $this->client->executeScript(<<<'JS_WRAP'
                return JSON.stringify({
                    koAvailable: typeof ko !== 'undefined',
                    mainMenuChildren: document.getElementById('mainMenu')?.children.length ?? 0
                });
            JS_WRAP);
            $stateText = is_string($state) ? $state : get_debug_type($state);
            fwrite(STDERR, "[E2E] waitForAppReady succeeded: {$stateText}\n");
            return true;
        } catch (TimeoutException) {
            return false;
        }
    }

    /**
     * Create a TimeoutException with diagnostic information about the page state.
     *
     * Call this after waitForAppReady() returns false to get a detailed exception
     * with information about why the app didn't initialize.
     */
    private function createAppReadyTimeoutException(): TimeoutException
    {
        try {
            $result = $this->client->executeScript(<<<'JS_WRAP'
                return JSON.stringify({
                    url: location.href,
                    readyState: document.readyState,
                    title: document.title,
                    koAvailable: typeof ko !== 'undefined',
                    mainMenuExists: document.getElementById('mainMenu') !== null,
                    mainMenuChildren: document.getElementById('mainMenu')?.children.length ?? 0,
                    bodyLength: document.body?.innerHTML?.length ?? 0
                });
            JS_WRAP);
            $diagnostics = is_string($result) ? $result : get_debug_type($result);
        } catch (WebDriverException) {
            // executeScript() failures surface as WebDriverException.
            // Narrow to it (rather than \Throwable or the broader
            // \Exception, which still overlaps \ErrorException) so
            // genuine programming errors still propagate.
            $diagnostics = 'unable to gather diagnostics (executeScript failed)';
        }
        return new TimeoutException(
            "waitForAppReady() timed out after retry. Page state: {$diagnostics}"
        );
    }

    private function switchToIFrame(string $xpath): void
    {
        $selector = WebDriverBy::xpath($xpath);
        $iframe = $this->client->findElement($selector);
        $this->client->switchTo()->frame($iframe);
        $this->crawler = $this->client->refreshCrawler();
    }

    private function assertActiveTab(string $text, string $loading = "Loading", bool $looseTabTitle = false): void
    {
        // Retry loop to handle page transitions that can cause the active tab
        // element to become stale. After accepting a JS alert dialog (e.g.,
        // "Create Visit" when a visit already exists), the page may reload and
        // replace the active tab element during waitForElementToNotContain.
        $maxRetries = 3;
        $lastException = null;

        for ($attempt = 1; $attempt <= $maxRetries; $attempt++) {
            try {
                // Wait for the active tab element to exist (handles page transitions)
                $this->client->waitFor(XpathsConstants::ACTIVE_TAB);

                // Wait for each loading indicator to disappear from the live DOM
                foreach (explode('||', $loading) as $loadingText) {
                    $this->client->waitForElementToNotContain(XpathsConstants::ACTIVE_TAB, $loadingText);
                }

                // Success - exit retry loop
                $lastException = null;
                break;
            } catch (UnexpectedResponseException | StaleElementReferenceException $e) {
                // Element became stale during the page transition - retry
                $lastException = $e;
                if ($attempt < $maxRetries) {
                    usleep(500_000); // 500ms before retry
                }
            } catch (UnexpectedAlertOpenException $e) {
                // An alert appeared after the goToMainMenuLink() wait window.
                // Accept it and retry (the page may reload after accepting).
                try {
                    $this->client->getWebDriver()->switchTo()->alert()->accept();
                } catch (WebDriverException) {
                    // Alert already dismissed.
                }
                $lastException = $e;
                if ($attempt < $maxRetries) {
                    usleep(500_000); // 500ms before retry
                }
            }
        }

        if ($lastException !== null) {
            throw $lastException;
        }

        $this->crawler = $this->client->refreshCrawler();
        if ($looseTabTitle) {
            $this->assertTrue(str_contains($this->crawler->filterXPath(XpathsConstants::ACTIVE_TAB)->text(), $text), "[$text] tab load FAILED");
        } else {
            $this->assertSame($text, $this->crawler->filterXPath(XpathsConstants::ACTIVE_TAB)->text(), "[$text] tab load FAILED");
        }
    }

    private function assertActivePopup(string $text): void
    {
        $this->client->waitForElementToContain(XpathsConstants::MODAL_TITLE, $text);
        $this->crawler = $this->client->refreshCrawler();
        $this->assertSame($text, $this->crawler->filterXPath(XpathsConstants::MODAL_TITLE)->text(), "[$text] popup load FAILED");
    }

    private function goToMainMenuLink(string $menuLink, bool $acceptAlert = false): void
    {
        // ensure on main page (ie. not in an iframe)
        $this->client->switchTo()->defaultContent();
        // go to and click the menu link
        $menuLinkSequenceArray = explode('||', $menuLink);
        $counter = 0;
        foreach ($menuLinkSequenceArray as $menuLinkItem) {
            if ($counter == 0) {
                if (count($menuLinkSequenceArray) > 1) {
                    // start clicking through a dropdown/nested menu item
                    $menuLink = '//div[@id="mainMenu"]/div/div/div/div[text()="' . $menuLinkItem . '"]';
                } else {
                    // just clicking a simple/single menu item
                    $menuLink = '//div[@id="mainMenu"]/div/div/div[text()="' . $menuLinkItem . '"]';
                }
            } elseif ($counter == 1) {
                if (count($menuLinkSequenceArray) == 2) {
                    // click the nested menu item
                    $menuLink = '//div[@id="mainMenu"]/div/div/div/div[text()="' . $menuLinkSequenceArray[0] . '"]/../ul/li/div[text()="' . $menuLinkItem . '"]';
                } else {
                    // continue clicking through a dropdown/nested menu item
                    $menuLink = '//div[@id="mainMenu"]/div/div/div/div[text()="' . $menuLinkSequenceArray[0] . '"]/../ul/li/div/div[text()="' . $menuLinkItem . '"]';
                }
            } else { // $counter > 1
                // click the nested menu item
                $menuLink = '//div[@id="mainMenu"]/div/div/div/div[text()="' . $menuLinkSequenceArray[0] . '"]/../ul/li/div/div[text()="' . $menuLinkSequenceArray[1] . '"]/../ul/li/div[text()="' . $menuLinkItem . '"]';
            }

            // Use elementToBeClickable + direct WebDriver click instead of
            // Panther's refreshCrawler/filterXPath/click, which can fail
            // with stale DOM references if the page updates between the
            // crawler snapshot and the click
            $element = $this->client->wait(30)->until(
                WebDriverExpectedCondition::elementToBeClickable(
                    WebDriverBy::xpath($menuLink)
                )
            );
            if (!$element instanceof WebDriverElement) {
                $this->fail('Expected a clickable WebDriverElement for menu link: ' . $menuLink);
            }
            $element->click();
            $counter++;
        }

        if ($acceptAlert) {
            // Accept any JavaScript alert/confirm that appears after clicking.
            // Some menu items (e.g., "Create Visit") show a confirm dialog if
            // a visit already exists for the patient today. Handle immediately
            // after clicking to prevent the alert from blocking subsequent
            // WebDriver operations.
            try {
                $this->client->wait(2)->until(function (WebDriver $driver) {
                    try {
                        $driver->switchTo()->alert()->accept();
                        return true;
                    } catch (WebDriverException) {
                        // No alert present.
                        return false;
                    }
                });
            } catch (TimeoutException) {
                // No alert appeared, which is fine
            }
        }
    }

    private function goToUserMenuLink(string $menuTreeIcon): void
    {
        $menuLink = XpathsConstants::USER_MENU_ICON;
        $menuLink2 = '//ul[@id="userdropdown"]//i[contains(@class, "' . $menuTreeIcon . '")]';
        $element = $this->client->wait(10)->until(
            WebDriverExpectedCondition::elementToBeClickable(
                WebDriverBy::xpath($menuLink)
            )
        );
        if (!$element instanceof WebDriverElement) {
            $this->fail('Expected a clickable WebDriverElement for menu link: ' . $menuLink);
        }
        $element->click();
        $element2 = $this->client->wait(10)->until(
            WebDriverExpectedCondition::elementToBeClickable(
                WebDriverBy::xpath($menuLink2)
            )
        );
        if (!$element2 instanceof WebDriverElement) {
            $this->fail('Expected a clickable WebDriverElement for menu link: ' . $menuLink2);
        }
        $element2->click();
    }

    private function isUserExist(string $username): bool
    {
        $usernameDatabase = QueryUtils::querySingleRow(
            "SELECT `username` FROM `users` WHERE `username` = ?",
            [$username]
        );
        return is_array($usernameDatabase) && ($usernameDatabase['username'] ?? '') === $username;
    }

    private function isPatientExist(string $firstname, string $lastname, string $dob, string $sex): bool
    {
        $patientDatabase = QueryUtils::querySingleRow(
            "SELECT `fname` FROM `patient_data` WHERE `fname` = ? AND `lname` = ? AND `DOB` = ? AND `sex` = ?",
            [$firstname, $lastname, $dob, $sex]
        );
        return is_array($patientDatabase)
            && ($patientDatabase['fname'] ?? '') !== ''
            && $patientDatabase['fname'] === $firstname;
    }

    private function isEncounterExist(string $firstname, string $lastname, string $dob, string $sex): bool
    {
        $patientDatabase = QueryUtils::querySingleRow(
            "SELECT `patient_data`.`fname`
                                     FROM `patient_data`
                                     INNER JOIN `form_encounter`
                                     ON `patient_data`.`pid` = `form_encounter`.`pid`
                                     WHERE `patient_data`.`fname` = ? AND `patient_data`.`lname` = ? AND `patient_data`.`DOB` = ? AND `patient_data`.`sex` = ?",
            [$firstname, $lastname, $dob, $sex]
        );
        return is_array($patientDatabase)
            && ($patientDatabase['fname'] ?? '') !== ''
            && $patientDatabase['fname'] === $firstname;
    }

    private function logOut(): void
    {
        $this->client->switchTo()->defaultContent();
        $this->goToUserMenuLink('fa-sign-out-alt');
        $this->client->waitFor('//input[@id="authUser"]');
        $title = $this->client->getTitle();
        $this->assertSame('OpenEMR Login', $title, 'Logout FAILED');
    }
}
