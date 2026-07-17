<?php

/**
 * Bootstrap Registration Test for Clinical Co-Pilot Module
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2025 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Tests\Isolated\Modules\ClinicalCopilot;

use OpenEMR\Core\ModulesClassLoader;
use OpenEMR\Events\Main\Tabs\RenderEvent;
use OpenEMR\Events\PatientDemographics\RenderEvent as PatientDemographicsRenderEvent;
use OpenEMR\Events\UserInterface\PageHeadingRenderEvent;
use OpenEMR\Menu\MenuEvent;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Symfony\Component\EventDispatcher\EventDispatcher;

class BootstrapRegistrationTest extends TestCase
{
    private string $projectDir;
    private string $moduleBootstrapPath;

    protected function setUp(): void
    {
        $this->projectDir = dirname(__DIR__, 5);
        $this->moduleBootstrapPath = $this->projectDir . '/interface/modules/custom_modules/oe-module-clinical-copilot/src';

        $classLoader = new ModulesClassLoader($this->projectDir);
        $classLoader->registerNamespaceIfNotExists(
            'OpenEMR\\Modules\\ClinicalCopilot\\',
            $this->moduleBootstrapPath
        );
    }

    #[Test]
    public function testBootstrapClassExists(): void
    {
        $this->assertTrue(class_exists('OpenEMR\\Modules\\ClinicalCopilot\\Bootstrap'), 'Bootstrap class should exist after namespace registration');
    }

    #[Test]
    public function testBootstrapRegistersPatientDemographicsAndPageHeadingListeners(): void
    {
        $eventDispatcher = new EventDispatcher();
        $bootstrapClass = 'OpenEMR\\Modules\\ClinicalCopilot\\Bootstrap';
        $bootstrap = new $bootstrapClass($eventDispatcher);

        $bootstrap->subscribeToEvents();

        $this->assertNotEmpty(
            $eventDispatcher->getListeners(PatientDemographicsRenderEvent::EVENT_SECTION_LIST_RENDER_BEFORE),
            'Bootstrap should register a listener for PatientDemographics RenderEvent::EVENT_SECTION_LIST_RENDER_BEFORE (copilot card injection)'
        );
        $this->assertNotEmpty(
            $eventDispatcher->getListeners(PageHeadingRenderEvent::EVENT_PAGE_HEADING_RENDER),
            'Bootstrap should register a listener for PageHeadingRenderEvent::EVENT_PAGE_HEADING_RENDER (open-chat button injection)'
        );
    }

    #[Test]
    public function testBootstrapRegistersGlobalLauncherListener(): void
    {
        $eventDispatcher = new EventDispatcher();
        $bootstrapClass = 'OpenEMR\\Modules\\ClinicalCopilot\\Bootstrap';
        $bootstrap = new $bootstrapClass($eventDispatcher);

        $bootstrap->subscribeToEvents();

        $this->assertNotEmpty(
            $eventDispatcher->getListeners(RenderEvent::EVENT_BODY_RENDER_POST),
            'Bootstrap should register a listener for RenderEvent::EVENT_BODY_RENDER_POST '
                . '(global floating launcher injected into the outer frameset chrome, P2.17)'
        );
    }

    #[Test]
    public function testBootstrapRegistersMenuUpdateListener(): void
    {
        $eventDispatcher = new EventDispatcher();
        $bootstrapClass = 'OpenEMR\\Modules\\ClinicalCopilot\\Bootstrap';
        $bootstrap = new $bootstrapClass($eventDispatcher);

        $bootstrap->subscribeToEvents();

        $this->assertNotEmpty(
            $eventDispatcher->getListeners(MenuEvent::MENU_UPDATE),
            'Bootstrap should register a listener for MenuEvent::MENU_UPDATE '
                . '(top-level "Co-Pilot" nav entry, issue #201)'
        );
    }

    #[Test]
    public function testMenuUpdateListenerAddsTopLevelCoPilotMenuItem(): void
    {
        // xlt() (used to translate the "Co-Pilot" label) resolves through
        // xl(), which hits the database unless translation is disabled --
        // same pattern as HscPrivateXlOrWarnTest.
        $GLOBALS['disable_translation'] = true;

        $eventDispatcher = new EventDispatcher();
        $bootstrapClass = 'OpenEMR\\Modules\\ClinicalCopilot\\Bootstrap';
        $bootstrap = new $bootstrapClass($eventDispatcher);

        $bootstrap->subscribeToEvents();

        // Seed with a Calendar-like sibling to prove the new item lands at
        // the SAME (top) level as existing entries, not nested as a child.
        $calendar = new \stdClass();
        $calendar->label = 'Calendar';
        $calendar->menu_id = 'cal0';
        $calendar->target = 'cal';
        $calendar->url = '/interface/main/main_info.php';
        $calendar->children = [];
        $calendar->requirement = 0;
        $calendar->acl_req = ['patients', 'appt'];

        $updatedEvent = $eventDispatcher->dispatch(new MenuEvent([$calendar]), MenuEvent::MENU_UPDATE);

        unset($GLOBALS['disable_translation']);

        $menu = $updatedEvent->getMenu();
        $this->assertCount(2, $menu, 'Co-Pilot item should be appended as a top-level sibling, not nested');

        $coPilotItem = $menu[1];
        $this->assertInstanceOf(\stdClass::class, $coPilotItem);
        $this->assertSame('Co-Pilot', $coPilotItem->label);
        $this->assertIsString($coPilotItem->url);
        $this->assertStringContainsString(
            '/interface/modules/custom_modules/oe-module-clinical-copilot/public/copilot.php',
            $coPilotItem->url
        );
        // 'blank' is a special target value recognized by menuActionClick()
        // (tabs_view_model.js) that opens the URL in a genuine new browser
        // tab/window instead of an in-app SPA tab -- required because the
        // standalone PWA's service worker/install prompt cannot register
        // from inside an iframe.
        $this->assertSame('blank', $coPilotItem->target);
        $this->assertSame(0, $coPilotItem->requirement);
        $this->assertSame(['patients', 'demo'], $coPilotItem->acl_req);
        $this->assertSame([], $coPilotItem->children);

        // The pre-existing Calendar item must be untouched (additive only).
        $this->assertSame($calendar, $menu[0]);
    }
}
