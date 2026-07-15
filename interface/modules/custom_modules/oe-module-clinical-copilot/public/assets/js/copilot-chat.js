/**
 * Clinical Co-Pilot module - chat panel (P2.14).
 *
 * Populates the panel toggled by copilot.js (P2.12) with a working chat:
 * input + send, message list, streaming assistant responses. Talks to the
 * agent's POST /chat SSE endpoint (P2.10) through the module's same-origin
 * proxy (public/chat-proxy.php, P2.14) rather than directly -- the agent has
 * no browser-reachable URL (see docker-compose.copilot.yml: copilot_internal
 * is `internal: true`, no host port). The bearer token is acquired once from
 * the P2.13 token broker (public/ajax.php) and cached for the panel session.
 *
 * "Streaming" here means incremental rendering of the SSE stream's named
 * events as they arrive over the wire (conversation ack, then the tool_call/
 * answer/done frames once the planner loop completes) -- the agent's
 * `answer` frame carries the complete final text in one event, not
 * token-by-token deltas (see app/chat.py's SSE frame contract), so there is
 * no typewriter effect to render; the UI reflects the real granularity of
 * the backend contract instead of faking one it doesn't have.
 *
 * Security: assistant/user text is always rendered via `textContent`
 * (appendMessage below), never `innerHTML` -- model output and patient
 * record text can carry adversarial content and must render as inert text.
 */
(function (window, document) {
    'use strict';

    // `document.currentScript` is only valid during this script's own
    // synchronous top-level execution (even with `defer`) -- it is null by
    // the time a later DOMContentLoaded callback runs, so it must be
    // captured here, not inside initFromDom().
    var CURRENT_SCRIPT_SRC = document.currentScript ? document.currentScript.src : '';

    // -------------------------------------------------------------------
    // Pure: incremental SSE frame parser.
    //
    // Server frames (app/chat.py's `_sse()`) look like:
    //   event: <name>\ndata: <json>\n\n
    // A `push(chunk)` call feeds one arbitrarily-sized piece of the byte
    // stream in and returns the frames that became complete as a result;
    // any trailing partial frame is retained internally for the next push.
    // Lines starting with ":" are SSE comments (used by the proxy as
    // keep-alive pings to survive Apache's idle timeout during long model
    // "thinking" gaps) and never produce a frame.
    // -------------------------------------------------------------------
    function parseBlock(rawBlock) {
        var lines = rawBlock.split('\n');
        var eventName = 'message'; // SSE default event name per spec
        var dataLines = [];
        for (var i = 0; i < lines.length; i++) {
            var line = lines[i];
            if (line.indexOf(':') === 0) {
                continue; // comment / keep-alive line
            }
            if (line.indexOf('event:') === 0) {
                eventName = line.slice('event:'.length).trim();
            } else if (line.indexOf('data:') === 0) {
                dataLines.push(line.slice('data:'.length).trim());
            }
        }
        if (dataLines.length === 0 && eventName === 'message') {
            return null; // pure comment block; nothing to emit
        }

        var rawData = dataLines.join('\n');
        var data = null;
        var error = null;
        try {
            data = rawData === '' ? {} : JSON.parse(rawData);
        } catch {
            error = 'malformed JSON in SSE data: ' + rawData;
        }
        return { event: eventName, data: data, error: error };
    }

    function createSSEFrameParser() {
        var buffer = '';
        return {
            push: function (chunk) {
                buffer += chunk;
                var frames = [];
                var boundary;
                while ((boundary = buffer.indexOf('\n\n')) !== -1) {
                    var rawBlock = buffer.slice(0, boundary);
                    buffer = buffer.slice(boundary + 2);
                    var frame = parseBlock(rawBlock);
                    if (frame) {
                        frames.push(frame);
                    }
                }
                return frames;
            }
        };
    }

    // -------------------------------------------------------------------
    // Drives a ReadableStreamDefaultReader-shaped reader (only `.read()`
    // is used, so tests can pass a fake) through the parser, invoking
    // `onFrame` for each complete frame in arrival order.
    // -------------------------------------------------------------------
    function consumeSSEStream(reader, onFrame) {
        var decoder = new TextDecoder();
        var parser = createSSEFrameParser();

        function pump() {
            return reader.read().then(function (result) {
                if (result.value) {
                    var chunk = decoder.decode(result.value, { stream: true });
                    var frames = parser.push(chunk);
                    for (var i = 0; i < frames.length; i++) {
                        onFrame(frames[i]);
                    }
                }
                if (result.done) {
                    return undefined;
                }
                return pump();
            });
        }

        return pump();
    }

    // -------------------------------------------------------------------
    // DOM rendering. `textContent` only -- see module docstring.
    // -------------------------------------------------------------------
    function appendMessage(container, role, text) {
        var el = document.createElement('div');
        el.className = 'copilot-chat-message copilot-chat-message-' + role;
        el.textContent = text;
        container.appendChild(el);
        container.scrollTop = container.scrollHeight;
        return el;
    }

    // -------------------------------------------------------------------
    // Orchestration.
    // -------------------------------------------------------------------
    var UNAVAILABLE_MESSAGE = 'Sorry, the Co-Pilot is unavailable right now.';

    function createChatController(options) {
        var cachedToken = null;
        var conversationId = null;

        function ensureToken() {
            if (cachedToken) {
                return Promise.resolve(cachedToken);
            }
            return options.fetchImpl(options.brokerUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: 'csrf_token_form=' + encodeURIComponent(options.context.csrfToken)
            }).then(function (resp) {
                return resp.json().then(function (data) {
                    if (!resp.ok || typeof data.token !== 'string' || data.token === '') {
                        throw new Error('token broker request failed');
                    }
                    cachedToken = data.token;
                    return cachedToken;
                });
            });
        }

        function sendMessage(text) {
            appendMessage(options.messagesEl, 'user', text);

            return ensureToken().then(function (token) {
                return options.fetchImpl(options.proxyUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        csrf_token_form: options.context.csrfToken,
                        token: token,
                        message: text,
                        conversation_id: conversationId
                    })
                });
            }).then(function (resp) {
                if (!resp.ok || !resp.body) {
                    throw new Error('chat proxy request failed');
                }
                var answerText = '';
                var hadError = false;
                return consumeSSEStream(resp.body.getReader(), function (frame) {
                    if (frame.event === 'conversation' && frame.data && typeof frame.data.conversation_id === 'string') {
                        conversationId = frame.data.conversation_id;
                    } else if (frame.event === 'answer' && frame.data && typeof frame.data.answer === 'string') {
                        answerText = frame.data.answer;
                    } else if (frame.event === 'error') {
                        hadError = true;
                    }
                }).then(function () {
                    if (answerText) {
                        appendMessage(options.messagesEl, 'assistant', answerText);
                    } else if (hadError) {
                        appendMessage(options.messagesEl, 'assistant', UNAVAILABLE_MESSAGE);
                    }
                });
            }).catch(function () {
                appendMessage(options.messagesEl, 'assistant', UNAVAILABLE_MESSAGE);
            });
        }

        function handleSubmit(evt) {
            evt.preventDefault();
            var text = options.inputEl.value.trim();
            if (!text) {
                return;
            }
            options.inputEl.value = '';
            options.inputEl.style.height = 'auto';
            sendMessage(text);
        }

        function init() {
            options.formEl.addEventListener('submit', handleSubmit);
        }

        return { init: init, sendMessage: sendMessage };
    }

    function initFromDom() {
        var panel = document.getElementById('copilot-chat-panel');
        var messagesEl = document.getElementById('copilot-chat-messages');
        var formEl = document.getElementById('copilot-chat-form');
        var inputEl = document.getElementById('copilot-chat-input');
        if (!panel || !messagesEl || !formEl || !inputEl || !window.CopilotContext) {
            return;
        }

        // This script's own URL gives us the module's public/ base path
        // without needing the server to thread it through CopilotContext.
        var baseUrl = CURRENT_SCRIPT_SRC.replace(/\/assets\/js\/copilot-chat\.js(\?.*)?$/, '');

        var controller = createChatController({
            messagesEl: messagesEl,
            formEl: formEl,
            inputEl: inputEl,
            context: window.CopilotContext,
            brokerUrl: baseUrl + '/ajax.php',
            proxyUrl: baseUrl + '/chat-proxy.php',
            fetchImpl: window.fetch.bind(window)
        });
        controller.init();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initFromDom);
    } else {
        initFromDom();
    }

    window.CopilotChat = {
        createSSEFrameParser: createSSEFrameParser,
        consumeSSEStream: consumeSSEStream,
        appendMessage: appendMessage,
        createChatController: createChatController
    };
})(window, document);
