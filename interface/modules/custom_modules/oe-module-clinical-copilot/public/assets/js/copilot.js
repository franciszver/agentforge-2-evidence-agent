/**
 * Clinical Co-Pilot module - open-chat button toggle.
 * P2.12 scope: inert UI shell only. No chat behavior (see P2.14), no token
 * brokering (see P2.13), no agent calls. Reads window.CopilotContext
 * (pid, encounter, authUserID), set by the server via json_encode()
 * before this script loads.
 */
(function () {
    'use strict';

    function init() {
        var button = document.getElementById('copilot-open-chat-btn');
        var panel = document.getElementById('copilot-chat-panel');
        if (!button || !panel) {
            return;
        }
        button.addEventListener('click', function () {
            var hidden = panel.classList.toggle('copilot-hidden');
            panel.setAttribute('aria-hidden', hidden ? 'true' : 'false');
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
