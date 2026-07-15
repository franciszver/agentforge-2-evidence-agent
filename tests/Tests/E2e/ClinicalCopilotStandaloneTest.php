<?php

/**
 * Clinical Co-Pilot Standalone PWA Route E2E Test
 *
 * Exercises the P2.15 standalone /copilot surface (plan §4.7) end-to-end
 * against the running dev stack: the module's public/copilot.php page is
 * the real dev URL behind the plan's "/copilot" shorthand (custom modules
 * have no router -- every public/*.php file IS a route, same as the P2.13
 * token broker and P2.14 chat proxy). It reuses the exact same chat assets
 * as the embedded P2.14 panel (copilot-chat.js, copilot.css, token broker,
 * chat proxy) -- this test only proves the standalone shell wires them up
 * correctly; the live SSE round trip itself is already covered by
 * ClinicalCopilotChatPanelTest and is not re-exercised here.
 *
 * Also covers the PWA installability surface: a <link rel="manifest">
 * pointing at a valid manifest.json (display: standalone, icons), and the
 * service worker script being served with a script-capable content type
 * (browsers refuse to register a service worker whose response has the
 * wrong MIME type).
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

class ClinicalCopilotStandaloneTest extends PantherTestCase
{
    use BaseTrait;
    use LoginTrait;

    private ?Crawler $crawler = null;

    private const DEMO_PATIENT_PID = 1;

    private const MODULE_PUBLIC_PATH = '/interface/modules/custom_modules/oe-module-clinical-copilot/public';

    private const STANDALONE_ROUTE = self::MODULE_PUBLIC_PATH . '/copilot.php';

    private const MANIFEST_PATH = self::MODULE_PUBLIC_PATH . '/manifest.json';

    private const SW_PATH = self::MODULE_PUBLIC_PATH . '/copilot-sw.js';

    #[Test]
    public function testStandaloneRouteRendersChatUiAndPwaSurfaceWithPatientInContext(): void
    {
        $this->base();
        try {
            $this->client->manage()->window()->setSize(new WebDriverDimension(360, 800));
            $this->login(LoginTestData::username, LoginTestData::password);
            // Bind a patient into session context before opening the
            // standalone route, same as opening the embedded panel from a
            // patient's dashboard.
            $this->client->request('GET', '/interface/patient_file/summary/demographics.php?set_pid=' . self::DEMO_PATIENT_PID);

            $this->crawler = $this->client->request('GET', self::STANDALONE_ROUTE);
            $this->client->waitFor("//*[@id='copilot-chat-input']", 15);

            $input = $this->client->findElement(WebDriverBy::id('copilot-chat-input'));
            $send = $this->client->findElement(WebDriverBy::id('copilot-chat-send-btn'));
            $this->assertTrue($input->isDisplayed(), 'chat input should be visible on the standalone route');
            $this->assertTrue($send->isDisplayed(), 'send button should be visible on the standalone route');

            $manifestLink = $this->crawler->filter("link[rel='manifest']");
            $this->assertGreaterThan(0, $manifestLink->count(), '<link rel="manifest"> must be present');
            $this->assertStringContainsString(
                'manifest.json',
                (string) $manifestLink->attr('href'),
                'manifest link must point at the module\'s manifest.json'
            );

            $manifest = $this->fetchJson(self::MANIFEST_PATH);
            $this->assertSame('standalone', $manifest['display'] ?? null, 'manifest must declare display: standalone');
            $this->assertIsArray($manifest['icons'] ?? null, 'manifest must declare icons');
            $this->assertGreaterThan(0, count($manifest['icons']), 'manifest must declare at least one icon');
            $this->assertNotSame('', $manifest['name'] ?? '', 'manifest must declare a name');
            $this->assertNotSame('', $manifest['short_name'] ?? '', 'manifest must declare a short_name');
        } finally {
            $this->client->quit();
        }
    }

    #[Test]
    public function testServiceWorkerScriptIsServedWithScriptContentType(): void
    {
        $this->base();
        try {
            $this->login(LoginTestData::username, LoginTestData::password);

            $result = $this->fetchRaw(self::SW_PATH);
            $this->assertSame(200, $result['status'], 'service worker script must be served');
            $this->assertMatchesRegularExpression(
                '#(java|ecma)script#i',
                $result['contentType'],
                'service worker response must have a script-capable content type or registration is refused by the browser'
            );
        } finally {
            $this->client->quit();
        }
    }

    #[Test]
    public function testStandaloneRouteWithoutPatientContextShowsOpenChartState(): void
    {
        $this->base();
        try {
            $this->login(LoginTestData::username, LoginTestData::password);
            // Deliberately skip visiting a patient dashboard: a fresh login
            // carries no pid in session, exercising the "no patient bound"
            // branch honestly rather than faking one.

            $this->crawler = $this->client->request('GET', self::STANDALONE_ROUTE);
            $this->client->waitFor("//*[@id='copilot-standalone-empty-state']", 15);

            $emptyState = $this->client->findElement(WebDriverBy::id('copilot-standalone-empty-state'));
            $this->assertTrue($emptyState->isDisplayed(), 'the "open a chart first" state should render with no patient in session');

            $this->assertCount(
                0,
                $this->client->findElements(WebDriverBy::id('copilot-chat-input')),
                'the chat form must not render when there is no patient in session context'
            );
        } finally {
            $this->client->quit();
        }
    }

    /**
     * @return array<array-key, mixed>
     */
    private function fetchJson(string $path): array
    {
        $script = <<<'JS'
            var url = arguments[0];
            var xhr = new XMLHttpRequest();
            xhr.open('GET', url, false);
            xhr.send(null);
            return xhr.responseText;
            JS;
        $raw = $this->client->executeScript($script, [$path]);
        $decoded = json_decode(is_string($raw) ? $raw : '', true);
        return is_array($decoded) ? $decoded : [];
    }

    /**
     * @return array{status: int, contentType: string}
     */
    private function fetchRaw(string $path): array
    {
        $script = <<<'JS'
            var url = arguments[0];
            var xhr = new XMLHttpRequest();
            xhr.open('GET', url, false);
            xhr.send(null);
            return JSON.stringify({
                status: xhr.status,
                contentType: xhr.getResponseHeader('Content-Type') || ''
            });
            JS;
        $raw = $this->client->executeScript($script, [$path]);
        $decoded = json_decode(is_string($raw) ? $raw : '', true);

        $status = is_array($decoded) && is_int($decoded['status'] ?? null) ? $decoded['status'] : 0;
        $contentType = is_array($decoded) && is_string($decoded['contentType'] ?? null) ? $decoded['contentType'] : '';

        return ['status' => $status, 'contentType' => $contentType];
    }
}
