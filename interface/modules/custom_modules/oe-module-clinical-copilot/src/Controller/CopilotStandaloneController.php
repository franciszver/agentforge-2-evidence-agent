<?php

/**
 * Clinical Co-Pilot Standalone PWA Page Controller
 *
 * Renders the full standalone page behind the plan's "/copilot" route
 * (§4.7): the real dev URL is public/copilot.php (custom modules have no
 * router in this codebase -- every public/*.php file directly under the
 * module IS a route; see public/ajax.php and public/chat-proxy.php for the
 * same pattern). This page is served under the same session-gated
 * globals.php bootstrap as the rest of the module (no $ignoreAuth) and
 * reuses the exact chat markup/assets/behavior as the P2.14 embedded panel
 * via CopilotPanelController -- it never forks the chat implementation, only
 * the page shell (manifest link, service worker registration, and the
 * honest "no patient" state) around it.
 *
 * Install prompts do not fire inside iframes/embedded panels, so this route
 * is what makes the same chat app installable to a phone home screen.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Controller;

use OpenEMR\Common\Session\PatientSessionUtil;
use OpenEMR\Core\OEGlobalsBag;
use OpenEMR\Modules\ClinicalCopilot\Bootstrap;

final class CopilotStandaloneController
{
    private readonly int $pid;

    private readonly string $moduleUrl;

    private readonly CopilotPanelController $panel;

    public function __construct()
    {
        $this->pid = PatientSessionUtil::getPid();
        $this->moduleUrl = OEGlobalsBag::getInstance()->getWebRoot() . Bootstrap::MODULE_INSTALLATION_PATH;
        $this->panel = new CopilotPanelController();
    }

    /**
     * Pure decision: is there a patient bound to the current session?
     * P2.16 will harden/expand patient-binding; this route keeps it honest
     * and simple for now -- no patient in session means no conversation to
     * have, so the page says so rather than rendering a chat form that
     * silently has nothing to talk about.
     */
    public static function hasPatientContext(int $pid): bool
    {
        return $pid > 0;
    }

    public function renderPage(): string
    {
        $manifestUrl = $this->moduleUrl . '/public/manifest.json';
        $swUrl = $this->moduleUrl . '/public/copilot-sw.js';

        ob_start();
        ?>
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
    <title><?php echo xlt('Co-Pilot'); ?></title>
    <meta name="theme-color" content="#0b5a8a">
    <link rel="manifest" href="<?php echo attr($manifestUrl); ?>">
        <?php echo $this->panel->renderAssetTags(); ?>
</head>
<body class="copilot-standalone-body">
        <?php if (self::hasPatientContext($this->pid)) { ?>
            <?php echo $this->panel->renderChatPanel(true); ?>
        <?php } else { ?>
    <div id="copilot-standalone-empty-state" class="copilot-standalone-empty-state">
        <p><?php echo xlt('Open a patient chart first to start a Co-Pilot conversation.'); ?></p>
    </div>
        <?php } ?>
<script>
    if ('serviceWorker' in navigator) {
        // Default scope is the script's own directory (public/), which
        // covers this page (also under public/) -- the SW cannot be scoped
        // any narrower than the directory it is served from, so it lives
        // alongside copilot.php rather than under assets/js/.
        navigator.serviceWorker.register(<?php echo json_encode($swUrl, JSON_THROW_ON_ERROR); ?>, { scope: <?php echo json_encode($this->moduleUrl . '/public/', JSON_THROW_ON_ERROR); ?> });
    }
</script>
</body>
</html>
        <?php
        return $this->endCapture();
    }

    /**
     * Close the output buffer opened by ob_start() and return its
     * contents. ob_get_clean() only returns false when no buffer is
     * active, which cannot happen on this path; the check narrows
     * string|false to string for renderPage()'s return type.
     */
    private function endCapture(): string
    {
        $html = ob_get_clean();
        return is_string($html) ? $html : '';
    }
}
