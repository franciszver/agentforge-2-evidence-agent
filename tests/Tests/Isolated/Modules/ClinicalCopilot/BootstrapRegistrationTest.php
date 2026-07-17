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
}
