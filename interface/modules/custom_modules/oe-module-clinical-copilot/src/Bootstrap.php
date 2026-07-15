<?php

/**
 * Clinical Co-Pilot Module Bootstrap Class
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2025 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot;

use OpenEMR\Common\Session\PatientSessionUtil;
use OpenEMR\Events\PatientDemographics\RenderEvent as PatientDemographicsRenderEvent;
use OpenEMR\Events\UserInterface\PageHeadingRenderEvent;
use OpenEMR\Modules\ClinicalCopilot\Controller\CopilotPanelController;
use Symfony\Component\EventDispatcher\EventDispatcherInterface;

class Bootstrap
{
    const MODULE_INSTALLATION_PATH = "/interface/modules/custom_modules/oe-module-clinical-copilot";
    const MODULE_NAME = "oe-module-clinical-copilot";

    /**
     * Page ID for the patient demographics/dashboard screen, as dispatched
     * by OemrUI::pageHeading() (see interface/patient_file/summary/demographics.php).
     */
    private const PATIENT_DASHBOARD_PAGE_ID = 'core.mrd';

    /**
     * The module's CSS/JS asset tags only need to be emitted once per
     * page, from whichever of the two listeners below fires first.
     */
    private bool $assetsRendered = false;

    public function __construct(
        /**
         * @var EventDispatcherInterface The object responsible for sending and subscribing to events.
         * Public so the scaffold carries the wiring for the first real event
         * subscription (P2.12) without a write-only private property.
         */
        public readonly EventDispatcherInterface $eventDispatcher
    ) {
    }

    /**
     * Subscribe to events.
     *
     * @return void
     */
    public function subscribeToEvents(): void
    {
        $this->eventDispatcher->addListener(
            PatientDemographicsRenderEvent::EVENT_SECTION_LIST_RENDER_BEFORE,
            $this->renderCopilotCard(...)
        );
        $this->eventDispatcher->addListener(
            PageHeadingRenderEvent::EVENT_PAGE_HEADING_RENDER,
            $this->renderOpenChatButton(...)
        );
    }

    /**
     * Inject the Co-Pilot card onto the patient dashboard.
     *
     * EVENT_SECTION_LIST_RENDER_BEFORE is dispatched exactly once,
     * unconditionally, before the dashboard card list (see
     * interface/patient_file/summary/demographics.php), so the card renders
     * regardless of which other cards the current user's ACLs allow.
     *
     * @param PatientDemographicsRenderEvent $event
     * @return void
     */
    public function renderCopilotCard(PatientDemographicsRenderEvent $event): void
    {
        // The event's pid is untyped and can carry a raw request value
        // (see demographics.php), so normalize before comparing: a
        // non-numeric string would otherwise slip past a bare `<= 0`
        // check under PHP 8 string-comparison semantics.
        $pid = $event->getPid();
        if (!is_numeric($pid) || (int) $pid <= 0) {
            return;
        }

        $controller = new CopilotPanelController();
        echo $this->renderAssetsOnce($controller);
        echo $controller->renderCard();
    }

    /**
     * Inject the persistent open-chat button into the patient dashboard's
     * page heading.
     *
     * @param PageHeadingRenderEvent $event
     * @return PageHeadingRenderEvent
     */
    public function renderOpenChatButton(PageHeadingRenderEvent $event): PageHeadingRenderEvent
    {
        if ($event->getPageId() !== self::PATIENT_DASHBOARD_PAGE_ID) {
            return $event;
        }
        // Same "no widget without a patient" gate as the card; this event
        // carries no pid, so read the normalized session value.
        if (PatientSessionUtil::getPid() <= 0) {
            return $event;
        }

        $controller = new CopilotPanelController();
        $event->appendTitleNavContent($this->renderAssetsOnce($controller) . $controller->renderOpenChatButton());

        return $event;
    }

    private function renderAssetsOnce(CopilotPanelController $controller): string
    {
        if ($this->assetsRendered) {
            return '';
        }
        $this->assetsRendered = true;

        return $controller->renderAssetTags();
    }
}
