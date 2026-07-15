<?php

/**
 * Clinical Co-Pilot Chat Panel E2E Test
 *
 * Exercises the P2.14 chat panel end-to-end against the running dev stack:
 * opening the panel, layout at the 360px phone breakpoint (thumb-reach send
 * button, visible input) and at desktop width, and a full live round trip
 * through the same-origin chat proxy (public/chat-proxy.php) to the real
 * agent/Ollama service -- a user bubble appears immediately and a real,
 * live-model assistant response streams in. Also exercises the proxy's
 * CSRF/method/request-shape gating (same discipline as
 * ClinicalCopilotTokenBrokerTest for P2.13's broker).
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

class ClinicalCopilotChatPanelTest extends PantherTestCase
{
    use BaseTrait;
    use LoginTrait;

    private ?Crawler $crawler = null;

    private const DEMO_PATIENT_PID = 1;

    private const CHAT_PROXY_PATH = '/interface/modules/custom_modules/oe-module-clinical-copilot/public/chat-proxy.php';

    /**
     * Real 4B-model round trip: single tool-call-free question, observed
     * ~9-29s in manual testing. Generous headroom for CI/dev-box variance
     * and the (up to 6-turn) planner loop on tool-bearing questions.
     */
    private const ASSISTANT_REPLY_TIMEOUT_SECONDS = 150;

    #[Test]
    public function testChatPanelLiveRoundTripAndThumbReachLayoutAtPhoneViewport(): void
    {
        $this->base();
        try {
            $this->client->manage()->window()->setSize(new WebDriverDimension(360, 800));
            $this->login(LoginTestData::username, LoginTestData::password);
            $this->openPatientDashboard();
            $this->openChatPanel();

            $this->assertThumbReachLayout();

            $this->sendMessage('What medications is this patient taking?');

            $this->client->waitFor("//*[contains(@class,'copilot-chat-message-user')]", 15);
            $userBubble = $this->client->findElement(
                WebDriverBy::xpath("//*[contains(@class,'copilot-chat-message-user')]")
            );
            $this->assertSame(
                'What medications is this patient taking?',
                $userBubble->getText(),
                'user bubble should render the exact text the user sent'
            );

            $this->client->wait(self::ASSISTANT_REPLY_TIMEOUT_SECONDS, 500)->until(
                fn(\Facebook\WebDriver\WebDriver $driver) => count($driver->findElements(
                    WebDriverBy::xpath("//*[contains(@class,'copilot-chat-message-assistant')]")
                )) > 0
            );
            $assistantBubble = $this->client->findElement(
                WebDriverBy::xpath("//*[contains(@class,'copilot-chat-message-assistant')]")
            );
            $this->assertNotSame(
                '',
                trim($assistantBubble->getText()),
                'a real assistant response should have streamed in from the live agent'
            );
        } finally {
            $this->client->quit();
        }
    }

    #[Test]
    public function testChatPanelIsUsableAtDesktopViewport(): void
    {
        $this->base();
        try {
            $this->client->manage()->window()->setSize(new WebDriverDimension(1366, 768));
            $this->login(LoginTestData::username, LoginTestData::password);
            $this->openPatientDashboard();
            $this->openChatPanel();

            $input = $this->client->findElement(WebDriverBy::id('copilot-chat-input'));
            $send = $this->client->findElement(WebDriverBy::id('copilot-chat-send-btn'));
            $this->assertTrue($input->isDisplayed(), 'chat input should be visible at desktop width');
            $this->assertTrue($send->isDisplayed(), 'send button should be visible at desktop width');
        } finally {
            $this->client->quit();
        }
    }

    #[Test]
    public function testChatProxyGating(): void
    {
        $this->base();
        try {
            $this->login(LoginTestData::username, LoginTestData::password);
            $this->openPatientDashboard();
            $this->client->waitFor("//*[@id='copilot-open-chat-btn']", 15);

            $csrfToken = $this->client->executeScript('return window.CopilotContext ? window.CopilotContext.csrfToken : null;');
            $this->assertIsString($csrfToken, 'panel context must expose a CSRF token');

            // Negative: tampered CSRF is rejected, no stream started.
            $tampered = $this->callChatProxy($csrfToken . 'tampered', 'dev-token', 'hello');
            $this->assertSame(403, $tampered['status'], 'tampered CSRF must be rejected');

            // Negative: non-POST method is rejected.
            $get = $this->callChatProxy($csrfToken, 'dev-token', 'hello', 'GET');
            $this->assertSame(405, $get['status'], 'non-POST method must be rejected');

            // Negative: a blank message is rejected by ChatProxyRequest.
            $blank = $this->callChatProxy($csrfToken, 'dev-token', '');
            $this->assertSame(400, $blank['status'], 'a blank message must be rejected');

            // Negative: a missing token is rejected by ChatProxyRequest.
            $script = <<<'JS'
                var url = arguments[0];
                var csrf = arguments[1];
                var xhr = new XMLHttpRequest();
                xhr.open('POST', url, false);
                xhr.setRequestHeader('Content-Type', 'application/json');
                xhr.send(JSON.stringify({ csrf_token_form: csrf, message: 'hello' }));
                return JSON.stringify({ status: xhr.status, body: xhr.responseText });
                JS;
            $raw = $this->client->executeScript($script, [self::CHAT_PROXY_PATH, $csrfToken]);
            $decoded = json_decode(is_string($raw) ? $raw : '', true);
            $status = is_array($decoded) && is_int($decoded['status'] ?? null) ? $decoded['status'] : null;
            $this->assertSame(400, $status, 'a missing token must be rejected');
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

    /**
     * Every interactive element the thumb needs (input, send button) must
     * render fully inside the viewport at the 360px breakpoint -- no
     * hover-dependent discovery, no off-screen controls.
     *
     * Reads the actual document viewport (window.innerWidth/innerHeight)
     * rather than trusting the WebDriverDimension passed to
     * manage()->window()->setSize(): that call sizes the outer browser
     * window (chrome/borders included), which is not reliably equal to
     * the content viewport across browsers/headless configurations.
     */
    private function assertThumbReachLayout(): void
    {
        $input = $this->client->findElement(WebDriverBy::id('copilot-chat-input'));
        $send = $this->client->findElement(WebDriverBy::id('copilot-chat-send-btn'));

        $this->assertTrue($input->isDisplayed(), 'chat input should be visible at the phone breakpoint');
        $this->assertTrue($send->isDisplayed(), 'send button should be visible at the phone breakpoint');

        $viewportWidth = $this->executeScriptAsInt('return window.innerWidth;');
        $viewportHeight = $this->executeScriptAsInt('return window.innerHeight;');

        $sendLocation = $send->getLocation();
        $sendSize = $send->getSize();
        $this->assertLessThanOrEqual(
            $viewportWidth,
            $sendLocation->getX() + $sendSize->getWidth(),
            'send button must render within the viewport width (reachable, not clipped off-screen)'
        );
        $this->assertLessThanOrEqual(
            $viewportHeight,
            $sendLocation->getY() + $sendSize->getHeight(),
            'send button must render within the viewport height'
        );
        // Thumb zone: the bottom-fixed panel keeps the send control in the
        // lower half of the screen, reachable one-handed without stretching.
        $this->assertGreaterThan(
            $viewportHeight / 2,
            $sendLocation->getY(),
            'send button should sit in the lower (thumb-reachable) half of the viewport'
        );
    }

    private function executeScriptAsInt(string $script): int
    {
        $result = $this->client->executeScript($script);
        if (!is_numeric($result)) {
            $this->fail('executeScript() did not return a numeric value for: ' . $script);
        }
        return (int) $result;
    }

    /**
     * @return array{status: int, body: string}
     */
    private function callChatProxy(string $csrfToken, string $token, string $message, string $method = 'POST'): array
    {
        $script = <<<'JS'
            var url = arguments[0];
            var method = arguments[1];
            var csrf = arguments[2];
            var token = arguments[3];
            var message = arguments[4];
            var xhr = new XMLHttpRequest();
            xhr.open(method, url, false);
            try {
                if (method === 'POST') {
                    xhr.setRequestHeader('Content-Type', 'application/json');
                    xhr.send(JSON.stringify({ csrf_token_form: csrf, token: token, message: message }));
                } else {
                    xhr.send(null);
                }
            } catch (e) {
                return JSON.stringify({ status: -1, body: String(e) });
            }
            return JSON.stringify({ status: xhr.status, body: xhr.responseText });
            JS;

        $raw = $this->client->executeScript($script, [self::CHAT_PROXY_PATH, $method, $csrfToken, $token, $message]);
        $decoded = json_decode(is_string($raw) ? $raw : '', true);

        $status = is_array($decoded) && is_int($decoded['status'] ?? null) ? $decoded['status'] : 0;
        $body = is_array($decoded) && is_string($decoded['body'] ?? null) ? $decoded['body'] : '';

        return ['status' => $status, 'body' => $body];
    }
}
