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
 *
 * P4.4: a thumbs up/down feedback widget renders under each assistant
 * response, tied to that response's P4.1 correlation id (delivered on the
 * `conversation` frame) and posted to the same-origin feedback proxy
 * (public/feedback-proxy.php) -- see attachFeedbackHandlers below.
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
    // P3.8 verification layer: verdict badge, citation chips, warning banner.
    //
    // Renders the `verification` SSE frame (app/chat.py
    // build_verification_payload). Every text field here derives from the
    // patient record / model output, so -- exactly like appendMessage -- it is
    // rendered via textContent only, never innerHTML: a `<script>` or
    // `<img onerror>` payload in a claim, citation, or warning renders inert.
    // No hover-dependent interaction: a citation chip is a real <button> the
    // user taps to reveal the underlying record.
    // -------------------------------------------------------------------
    var VERDICT_BADGES = {
        verified: { label: 'Verified', icon: '✓', className: 'copilot-verdict-verified' },
        partially_verified: { label: 'Partially verified', icon: '⚠', className: 'copilot-verdict-partial' },
        blocked: { label: 'Blocked', icon: '✕', className: 'copilot-verdict-blocked' }
    };

    function verdictBadgeInfo(verdict) {
        if (typeof verdict !== 'string' || !Object.prototype.hasOwnProperty.call(VERDICT_BADGES, verdict)) {
            return null;
        }
        return VERDICT_BADGES[verdict];
    }

    function renderVerdictBadge(verdict) {
        var info = verdictBadgeInfo(verdict);
        if (!info) {
            return null;
        }
        var badge = document.createElement('span');
        badge.className = 'copilot-verdict-badge ' + info.className;
        badge.setAttribute('role', 'status');

        var icon = document.createElement('span');
        icon.className = 'copilot-verdict-badge-icon';
        icon.setAttribute('aria-hidden', 'true');
        icon.textContent = info.icon;
        badge.appendChild(icon);

        // The accessible signal is the text label, not the colour alone.
        var label = document.createElement('span');
        label.className = 'copilot-verdict-badge-label';
        label.textContent = info.label;
        badge.appendChild(label);

        return badge;
    }

    function appendRecordField(record, label, value) {
        if (value === null || value === undefined || value === '') {
            return;
        }
        var row = document.createElement('div');
        row.className = 'copilot-citation-record-row';

        var key = document.createElement('span');
        key.className = 'copilot-citation-record-key';
        key.textContent = label;
        row.appendChild(key);

        var val = document.createElement('span');
        val.className = 'copilot-citation-record-value';
        val.textContent = String(value);
        row.appendChild(val);

        record.appendChild(row);
    }

    function buildCitationChip(citation) {
        citation = citation || {};

        var chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'copilot-citation-chip';
        chip.textContent = citation.field ? String(citation.field) : 'source';
        chip.setAttribute('aria-expanded', 'false');

        var record = document.createElement('div');
        record.className = 'copilot-citation-record copilot-hidden';
        appendRecordField(record, 'Field', citation.field);
        appendRecordField(record, 'Value', citation.value);
        appendRecordField(record, 'Record', citation.record_id);
        appendRecordField(record, 'Source', citation.tool_call_id);

        chip.addEventListener('click', function () {
            var nowHidden = record.classList.toggle('copilot-hidden');
            chip.setAttribute('aria-expanded', nowHidden ? 'false' : 'true');
        });

        return { chip: chip, record: record };
    }

    function renderClaimSegment(segment) {
        var claim = document.createElement('div');
        claim.className = 'copilot-claim';

        var text = document.createElement('div');
        text.className = 'copilot-claim-text';
        text.textContent = segment.text ? String(segment.text) : '';
        claim.appendChild(text);

        var citations = Array.isArray(segment.citations) ? segment.citations : [];
        if (citations.length > 0) {
            var chips = document.createElement('div');
            chips.className = 'copilot-claim-chips';
            var records = document.createElement('div');
            records.className = 'copilot-claim-records';

            for (var i = 0; i < citations.length; i++) {
                var built = buildCitationChip(citations[i]);
                chips.appendChild(built.chip);
                records.appendChild(built.record);
            }
            claim.appendChild(chips);
            claim.appendChild(records);
        }
        return claim;
    }

    function renderNoticeSegment(segment) {
        var notice = document.createElement('div');
        notice.className = 'copilot-claim copilot-notice';
        notice.textContent = segment.text ? String(segment.text) : '';
        return notice;
    }

    function renderWarningBanner(warnings) {
        warnings = warnings || {};
        var allergies = Array.isArray(warnings.allergy_conflicts) ? warnings.allergy_conflicts : [];
        var blocking = Array.isArray(warnings.blocking_interactions) ? warnings.blocking_interactions : [];
        if (allergies.length === 0 && blocking.length === 0) {
            return null;
        }

        var banner = document.createElement('div');
        banner.className = 'copilot-warning-banner';
        banner.setAttribute('role', 'alert');

        var heading = document.createElement('div');
        heading.className = 'copilot-warning-banner-heading';
        var icon = document.createElement('span');
        icon.className = 'copilot-warning-banner-icon';
        icon.setAttribute('aria-hidden', 'true');
        icon.textContent = '⚠';
        heading.appendChild(icon);
        var headingText = document.createElement('span');
        headingText.textContent = 'Safety warning';
        heading.appendChild(headingText);
        banner.appendChild(heading);

        var list = document.createElement('ul');
        list.className = 'copilot-warning-banner-list';
        for (var i = 0; i < allergies.length; i++) {
            var conflict = allergies[i] || {};
            var allergyItem = document.createElement('li');
            allergyItem.textContent =
                'Allergy conflict: ' + conflict.medication_name + ' vs recorded allergy to ' + conflict.allergy_substance;
            list.appendChild(allergyItem);
        }
        for (var j = 0; j < blocking.length; j++) {
            var interaction = blocking[j] || {};
            var interactionItem = document.createElement('li');
            interactionItem.textContent =
                'Interaction (' + interaction.severity + '): ' +
                interaction.drug_a + ' + ' + interaction.drug_b + ' — ' + interaction.description;
            list.appendChild(interactionItem);
        }
        banner.appendChild(list);

        return banner;
    }

    // Assembles the full verification block (banner + badge + claims) and
    // appends it to `container`. Returns the block element, or null for the
    // pending/degenerate payload (verdict absent) -- nothing to render yet.
    function renderVerification(container, data) {
        if (!data || !data.verdict) {
            return null;
        }

        var block = document.createElement('div');
        block.className = 'copilot-verification';

        var banner = renderWarningBanner(data.warnings);
        if (banner) {
            block.appendChild(banner);
        }

        var badge = renderVerdictBadge(data.verdict);
        if (badge) {
            block.appendChild(badge);
        }

        var segments = Array.isArray(data.segments) ? data.segments : [];
        if (segments.length > 0) {
            var claims = document.createElement('div');
            claims.className = 'copilot-claims';
            for (var i = 0; i < segments.length; i++) {
                var segment = segments[i];
                if (segment && segment.type === 'claim') {
                    claims.appendChild(renderClaimSegment(segment));
                } else if (segment && segment.type === 'notice') {
                    claims.appendChild(renderNoticeSegment(segment));
                }
            }
            block.appendChild(claims);
        }

        container.appendChild(block);
        container.scrollTop = container.scrollHeight;
        return block;
    }

    // -------------------------------------------------------------------
    // P4.4 feedback buttons: thumbs up/down per assistant response, tied to
    // that response's P4.1 correlation id (delivered on the `conversation`
    // frame -- see app/chat.py's SSE frame contract, and the `feedbackUrl`
    // proxy request in createChatController below). Real <button> elements
    // (>=44px tap target, no hover dependence -- see copilot.css), and
    // guarded against double-submit: both buttons disable the instant either
    // is clicked, and a click is a no-op while a submission is already
    // pending or done.
    // -------------------------------------------------------------------

    // Pure: the exact JSON body posted to public/feedback-proxy.php.
    function buildFeedbackPayload(context, token, correlationId, thumb, comment) {
        var payload = {
            csrf_token_form: context.csrfToken,
            token: token,
            correlation_id: correlationId,
            thumb: thumb
        };
        if (comment) {
            payload.comment = comment;
        }
        return payload;
    }

    // Pure: builds the (unattached) feedback widget DOM for one response.
    function renderFeedbackWidget(container, correlationId) {
        var wrapper = document.createElement('div');
        wrapper.className = 'copilot-feedback';
        wrapper.setAttribute('data-correlation-id', correlationId || '');

        var upBtn = document.createElement('button');
        upBtn.type = 'button';
        upBtn.className = 'copilot-feedback-btn copilot-feedback-up';
        upBtn.setAttribute('aria-label', 'Helpful');
        upBtn.setAttribute('data-thumb', 'up');
        upBtn.textContent = '👍';

        var downBtn = document.createElement('button');
        downBtn.type = 'button';
        downBtn.className = 'copilot-feedback-btn copilot-feedback-down';
        downBtn.setAttribute('aria-label', 'Not helpful');
        downBtn.setAttribute('data-thumb', 'down');
        downBtn.textContent = '👎';

        var status = document.createElement('span');
        status.className = 'copilot-feedback-status';
        status.setAttribute('aria-live', 'polite');

        var commentWrap = document.createElement('div');
        commentWrap.className = 'copilot-feedback-comment copilot-hidden';
        var commentInput = document.createElement('textarea');
        commentInput.className = 'copilot-feedback-comment-input';
        commentInput.setAttribute('placeholder', 'What went wrong? (optional)');
        commentInput.setAttribute('maxlength', '2000');
        commentInput.setAttribute('aria-label', 'Feedback comment');
        var commentSendBtn = document.createElement('button');
        commentSendBtn.type = 'button';
        commentSendBtn.className = 'copilot-feedback-comment-send';
        commentSendBtn.textContent = 'Send';
        commentWrap.appendChild(commentInput);
        commentWrap.appendChild(commentSendBtn);

        wrapper.appendChild(upBtn);
        wrapper.appendChild(downBtn);
        wrapper.appendChild(status);
        wrapper.appendChild(commentWrap);
        container.appendChild(wrapper);

        return {
            wrapper: wrapper,
            upBtn: upBtn,
            downBtn: downBtn,
            status: status,
            commentWrap: commentWrap,
            commentInput: commentInput,
            commentSendBtn: commentSendBtn
        };
    }

    // Pure DOM mutators: the three states a feedback widget can be in.
    function applyFeedbackPendingState(widget) {
        widget.upBtn.disabled = true;
        widget.downBtn.disabled = true;
    }

    function applyFeedbackSuccessState(widget, thumb) {
        applyFeedbackPendingState(widget);
        widget.wrapper.classList.add('copilot-feedback-submitted');
        var selectedBtn = thumb === 'up' ? widget.upBtn : widget.downBtn;
        selectedBtn.classList.add('copilot-feedback-selected');
        widget.status.textContent = 'Thanks for your feedback';
    }

    function applyFeedbackErrorState(widget) {
        widget.upBtn.disabled = false;
        widget.downBtn.disabled = false;
        widget.status.textContent = 'Could not send feedback. Try again.';
    }

    // Orchestration: wires the widget's buttons to POST public/feedback-proxy.php.
    // `options` needs `context`, `ensureToken` (the chat controller's cached
    // bearer-token seam), `fetchImpl`, and `feedbackUrl`.
    function attachFeedbackHandlers(widget, correlationId, options) {
        var state = 'idle'; // idle | pending | done -- the no-double-submit guard

        function post(thumb, comment) {
            return options.ensureToken().then(function (token) {
                var payload = buildFeedbackPayload(options.context, token, correlationId, thumb, comment);
                return options.fetchImpl(options.feedbackUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
            }).then(function (resp) {
                if (!resp.ok) {
                    throw new Error('feedback request failed');
                }
            });
        }

        function submitThumb(thumb) {
            if (state !== 'idle') {
                return Promise.resolve();
            }
            state = 'pending';
            applyFeedbackPendingState(widget);

            return post(thumb, null).then(function () {
                state = 'done';
                applyFeedbackSuccessState(widget, thumb);
                if (thumb === 'down') {
                    widget.commentWrap.classList.remove('copilot-hidden');
                }
            }).catch(function () {
                state = 'idle';
                applyFeedbackErrorState(widget);
            });
        }

        function submitComment() {
            var comment = widget.commentInput.value.trim();
            if (!comment || widget.commentSendBtn.disabled) {
                return Promise.resolve();
            }
            widget.commentSendBtn.disabled = true;

            return post('down', comment).then(function () {
                widget.commentWrap.classList.add('copilot-hidden');
                widget.status.textContent = 'Thanks for the detail';
            }).catch(function () {
                widget.commentSendBtn.disabled = false;
                widget.status.textContent = 'Could not send comment. Try again.';
            });
        }

        widget.upBtn.addEventListener('click', function () {
            submitThumb('up');
        });
        widget.downBtn.addEventListener('click', function () {
            submitThumb('down');
        });
        widget.commentSendBtn.addEventListener('click', submitComment);

        return { submitThumb: submitThumb, submitComment: submitComment };
    }

    // -------------------------------------------------------------------
    // Orchestration.
    // -------------------------------------------------------------------
    var UNAVAILABLE_MESSAGE = 'Sorry, the Co-Pilot is unavailable right now.';

    function createChatController(options) {
        var cachedToken = null;
        var conversationId = null;
        var redirect = options.redirectImpl || function (url) {
            window.location.assign(url);
        };

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
                    // Flag-on Phase 3: the broker asks the user to (re)authorize.
                    // Send them into the authorize flow and stop this turn — the
                    // page is navigating away, so we return a promise that never
                    // settles rather than surfacing an "unavailable" message.
                    if (data && data.consent_required === true) {
                        redirect(options.authorizeUrl);
                        return new Promise(function () {});
                    }
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
            // Per-turn, not per-conversation (unlike conversationId above): a
            // fresh P4.1 correlation id arrives on every `conversation` frame,
            // one per response, so the feedback widget below is tied to THIS
            // answer specifically.
            var responseCorrelationId = null;

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
                var verificationData = null;
                var hadError = false;
                return consumeSSEStream(resp.body.getReader(), function (frame) {
                    if (frame.event === 'conversation' && frame.data) {
                        if (typeof frame.data.conversation_id === 'string') {
                            conversationId = frame.data.conversation_id;
                        }
                        if (typeof frame.data.correlation_id === 'string') {
                            responseCorrelationId = frame.data.correlation_id;
                        }
                    } else if (frame.event === 'answer' && frame.data && typeof frame.data.answer === 'string') {
                        answerText = frame.data.answer;
                    } else if (frame.event === 'verification' && frame.data) {
                        verificationData = frame.data;
                    } else if (frame.event === 'error') {
                        hadError = true;
                    }
                }).then(function () {
                    if (answerText) {
                        appendMessage(options.messagesEl, 'assistant', answerText);
                        // Pending verification payloads (verdict null) render
                        // nothing; a populated one adds the badge/chips/banner.
                        renderVerification(options.messagesEl, verificationData);
                        if (responseCorrelationId) {
                            var widget = renderFeedbackWidget(options.messagesEl, responseCorrelationId);
                            attachFeedbackHandlers(widget, responseCorrelationId, {
                                context: options.context,
                                ensureToken: ensureToken,
                                fetchImpl: options.fetchImpl,
                                feedbackUrl: options.feedbackUrl
                            });
                        }
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
            feedbackUrl: baseUrl + '/feedback-proxy.php',
            authorizeUrl: baseUrl + '/oauth-authorize.php',
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
        createChatController: createChatController,
        verdictBadgeInfo: verdictBadgeInfo,
        renderVerdictBadge: renderVerdictBadge,
        renderWarningBanner: renderWarningBanner,
        renderVerification: renderVerification,
        buildFeedbackPayload: buildFeedbackPayload,
        renderFeedbackWidget: renderFeedbackWidget,
        applyFeedbackPendingState: applyFeedbackPendingState,
        applyFeedbackSuccessState: applyFeedbackSuccessState,
        applyFeedbackErrorState: applyFeedbackErrorState,
        attachFeedbackHandlers: attachFeedbackHandlers
    };
})(window, document);
