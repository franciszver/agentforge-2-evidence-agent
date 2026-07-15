<?php

/**
 * Clinical Co-Pilot Panel Controller
 *
 * Renders the inert UI shells injected onto the patient dashboard: the
 * Co-Pilot card and the persistent open-chat button, plus the module's
 * CSS/JS asset tags and the escaped session context they read.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Controller;

use OpenEMR\Common\Csrf\CsrfUtils;
use OpenEMR\Common\Session\EncounterSessionUtil;
use OpenEMR\Common\Session\PatientSessionUtil;
use OpenEMR\Common\Session\SessionWrapperFactory;
use OpenEMR\Core\OEGlobalsBag;
use OpenEMR\Modules\ClinicalCopilot\Bootstrap;

final class CopilotPanelController
{
    private readonly int $pid;

    private readonly int $encounter;

    private readonly int $authUserId;

    private readonly string $moduleUrl;

    private readonly string $csrfToken;

    public function __construct()
    {
        $this->pid = PatientSessionUtil::getPid();
        $this->encounter = EncounterSessionUtil::getEncounter();

        $session = SessionWrapperFactory::getInstance()->getActiveSession();
        $rawAuthUserId = $session->get('authUserID');
        $this->authUserId = is_numeric($rawAuthUserId) ? (int) $rawAuthUserId : 0;

        // CSRF token the panel JS sends to the token broker (public/ajax.php).
        $this->csrfToken = CsrfUtils::collectCsrfToken($session);

        $this->moduleUrl = OEGlobalsBag::getInstance()->getWebRoot() . Bootstrap::MODULE_INSTALLATION_PATH;
    }

    /**
     * Render the Co-Pilot card injected onto the patient dashboard.
     * Inert shell only - no chat behavior (see P2.14).
     */
    public function renderCard(): string
    {
        ob_start();
        ?>
        <div class="card copilot-card" id="copilot-card" data-copilot-pid="<?php echo attr((string) $this->pid); ?>">
            <div class="card-body p-1">
                <h6 class="card-title mb-0"><?php echo xlt('Co-Pilot'); ?></h6>
                <div class="copilot-card-body text-muted small">
                    <?php echo xlt('Clinical Co-Pilot assistant'); ?>
                </div>
            </div>
        </div>
        <?php
        return $this->endCapture();
    }

    /**
     * Render the persistent open-chat button injected into the page
     * heading, plus an inert placeholder panel it toggles.
     */
    public function renderOpenChatButton(): string
    {
        ob_start();
        ?>
        <button type="button"
                id="copilot-open-chat-btn"
                class="btn btn-sm btn-outline-primary copilot-open-chat-btn"
                title="<?php echo xla('Open Clinical Co-Pilot'); ?>">
            <i class="fa fa-comment-medical"></i> <?php echo xlt('Co-Pilot'); ?>
        </button>
        <div id="copilot-chat-panel" class="copilot-chat-panel copilot-hidden" aria-hidden="true"></div>
        <?php
        return $this->endCapture();
    }

    /**
     * Render the module's CSS/JS asset tags and the current session
     * context (pid, encounter, authUserID, csrfToken) for the panel JS to read.
     *
     * The context values are ints parsed at the session boundary, and the
     * JSON_HEX_* flags escape <, >, &, ', and " so no value can break out
     * of the inline script context (e.g. via a literal close-script tag)
     * even if a string field is ever added to the context. js_escape() is
     * not used because it is a bare json_encode() without these flags and
     * its declared signature accepts only strings.
     */
    public function renderAssetTags(): string
    {
        $context = [
            'pid' => $this->pid,
            'encounter' => $this->encounter,
            'authUserID' => $this->authUserId,
            'csrfToken' => $this->csrfToken,
        ];
        $contextJson = json_encode(
            $context,
            JSON_THROW_ON_ERROR | JSON_HEX_TAG | JSON_HEX_AMP | JSON_HEX_APOS | JSON_HEX_QUOT
        );
        ob_start();
        ?>
        <link rel="stylesheet" href="<?php echo attr($this->moduleUrl . '/public/assets/css/copilot.css'); ?>">
        <script>
            window.CopilotContext = <?php echo $contextJson; ?>;
        </script>
        <script src="<?php echo attr($this->moduleUrl . '/public/assets/js/copilot.js'); ?>" defer></script>
        <?php
        return $this->endCapture();
    }

    /**
     * Close the output buffer opened by ob_start() and return its
     * contents. ob_get_clean() only returns false when no buffer is
     * active, which cannot happen on these paths; the check narrows
     * string|false to string for the render methods' return types.
     */
    private function endCapture(): string
    {
        $html = ob_get_clean();
        return is_string($html) ? $html : '';
    }
}
