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
use OpenEMR\Events\Main\Tabs\RenderEvent;
use OpenEMR\Events\PatientDemographics\RenderEvent as PatientDemographicsRenderEvent;
use OpenEMR\Events\UserInterface\PageHeadingRenderEvent;
use OpenEMR\Menu\MenuEvent;
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
     * page, from whichever of the listeners below fires first.
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
        $this->eventDispatcher->addListener(
            RenderEvent::EVENT_BODY_RENDER_POST,
            $this->renderGlobalLauncher(...)
        );
        $this->eventDispatcher->addListener(
            MenuEvent::MENU_UPDATE,
            $this->addTopNavMenuItem(...)
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

    /**
     * Inject a fixed-position floating launcher into the outer frameset
     * chrome (interface/main/tabs/main.php) so the Co-Pilot is discoverable
     * on every page -- calendar, patients, admin -- not just the patient
     * dashboard the other two listeners are gated on.
     *
     * EVENT_BODY_RENDER_POST fires in main.php's own document, a separate
     * top-level frame from the patient-content iframe the other two
     * listeners render into, so this reuses the exact same element ids
     * (copilot-open-chat-btn / copilot-chat-panel) as renderOpenChatButton()
     * without any DOM collision -- and copilot.js's existing toggle wiring
     * (which queries those ids) works here unmodified.
     *
     * The launcher always renders the real chat panel, never a patient-gated
     * empty-state baked in at render time: main.php is a long-lived SPA shell
     * whose own document never reloads when the user selects a patient (only
     * the content iframe navigates), so any has-patient decision made here
     * would be frozen at login (no patient) forever. Instead the chat binds
     * to the *current* patient at send time -- ChatProxyController reads the
     * pid from the session per request, never from client input -- and
     * cleanly answers "open a patient chart first" when none is selected
     * (see copilot-chat.js's no_patient_in_session handling).
     *
     * @param RenderEvent $event
     * @return void
     */
    public function renderGlobalLauncher(RenderEvent $event): void
    {
        $controller = new CopilotPanelController();
        echo $this->renderAssetsOnce($controller);
        echo '<div class="copilot-global-launcher">';
        echo $controller->renderOpenChatButton();
        echo '</div>';
    }

    private function renderAssetsOnce(CopilotPanelController $controller): string
    {
        if ($this->assetsRendered) {
            return '';
        }
        $this->assetsRendered = true;

        return $controller->renderAssetTags();
    }

    /**
     * Add a top-level "Co-Pilot" entry to OpenEMR's primary nav menu
     * (issue #201), launching the standalone full-page PWA.
     *
     * Sibling modules (e.g. oe-module-dashboard-context, oe-module-weno)
     * all nest their menu item as a CHILD of an existing node (Admin,
     * Modules, etc.) by walking $menu and pushing onto ->children. Issue
     * #201 explicitly calls for a genuinely top-level item next to
     * Calendar/Patient, so this appends directly onto the top-level $menu
     * array returned by MenuEvent::getMenu() instead of nesting under an
     * existing entry -- see interface/main/tabs/menu/menus/standard.json
     * for the node shape those top-level siblings use (label/menu_id/
     * target/url/children/requirement/acl_req), which buildTopNavMenuItem()
     * mirrors.
     *
     * @param MenuEvent $event
     * @return MenuEvent
     */
    public function addTopNavMenuItem(MenuEvent $event): MenuEvent
    {
        $menu = $event->getMenu();
        $menu[] = $this->buildTopNavMenuItem();
        $event->setMenu($menu);

        return $event;
    }

    /**
     * Build the top-level "Co-Pilot" menu node.
     *
     * - target 'blank' is a special value recognized by menuActionClick()
     *   (interface/main/tabs/js/tabs_view_model.js) that opens the url in a
     *   genuine new browser tab via window.open(), instead of the app's own
     *   knockout-driven SPA tab system (all other menu items load their url
     *   into an in-frameset iframe tab). A real top-level browsing context
     *   is required here because CopilotStandaloneController's manifest
     *   link and service worker registration only work outside an iframe
     *   -- see docs/IMPLEMENTATION_PLAN.md's "PWA install prompt inside
     *   iframe" row.
     * - requirement 0 mirrors Calendar's top-level entry: always enabled,
     *   not gated on a patient/encounter being selected, since this is a
     *   global launcher reachable from anywhere.
     * - acl_req ["patients", "demo"] matches the baseline clinical-read
     *   access already required to reach the patient dashboard where the
     *   embedded Co-Pilot panel lives (see Finder's identical acl_req in
     *   standard.json) -- the standalone route itself enforces no
     *   additional ACL beyond a logged-in session.
     */
    private function buildTopNavMenuItem(): \stdClass
    {
        $menuItem = new \stdClass();
        $menuItem->requirement = 0;
        $menuItem->target = 'blank';
        $menuItem->menu_id = 'copilot0';
        $menuItem->label = xlt('Co-Pilot');
        $menuItem->url = self::MODULE_INSTALLATION_PATH . '/public/copilot.php';
        $menuItem->children = [];
        $menuItem->acl_req = ['patients', 'demo'];
        $menuItem->global_req = [];

        return $menuItem;
    }
}
