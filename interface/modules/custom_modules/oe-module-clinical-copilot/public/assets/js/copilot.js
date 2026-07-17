/**
 * Clinical Co-Pilot module - open-chat button toggle.
 * Toggles the panel's visibility; the chat behavior itself lives in
 * copilot-chat.js (token brokering, agent calls). For the P2.17 global
 * launcher only, opening the panel also resets the conversation
 * (window.CopilotChat.resetActiveConversation) so the never-reloaded
 * main.php shell never carries one patient's conversation into another's.
 */
(function () {
    'use strict';

    function init() {
        var button = document.getElementById('copilot-open-chat-btn');
        var panel = document.getElementById('copilot-chat-panel');
        if (!button || !panel) {
            return;
        }
        // Whether this button/panel pair is the P2.17 global launcher (in the
        // never-reloaded main.php shell) rather than the in-context heading
        // panel (which lives in the per-patient content iframe, reloaded on
        // every patient switch). Only the launcher needs a per-open reset.
        var isGlobalLauncher = !!button.closest('.copilot-global-launcher');

        button.addEventListener('click', function () {
            var hidden = panel.classList.toggle('copilot-hidden');
            panel.setAttribute('aria-hidden', hidden ? 'true' : 'false');

            // Opening the launcher starts a fresh conversation bound to the
            // CURRENT patient: main.php never reloads on patient switch, so a
            // panel opened earlier for another patient (or for none) must not
            // carry its conversation across -- the agent binds a conversation
            // to its patient and rejects a mismatched pid. The in-context
            // heading panel is left untouched (its iframe already resets it).
            if (!hidden && isGlobalLauncher &&
                window.CopilotChat && typeof window.CopilotChat.resetActiveConversation === 'function') {
                window.CopilotChat.resetActiveConversation();
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
