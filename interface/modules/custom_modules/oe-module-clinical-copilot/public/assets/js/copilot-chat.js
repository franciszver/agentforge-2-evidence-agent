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
 * events as they arrive over the wire (conversation ack, tool_call frames,
 * reasoning_delta frames, then answer/verification/done once the planner
 * loop completes). The `answer` frame still only ever carries the complete,
 * VERIFIED final text in one event -- that is a deliberate safety property
 * (see the P213 reasoning zone below), not a missing feature: the
 * `extract(FinalAnswer)` call that produces it cannot stream (schema decode
 * needs the whole JSON). The `reasoning_delta` frame is what streams
 * token-by-token, into its own clearly-labeled surface.
 *
 * P213 reasoning zone: the model's free-text reasoning (`reasoning_delta`
 * frames, `app/chat.py`'s SSE frame contract) types into a separate
 * "Reasoning locally (Qwen3-4B)..." zone as it arrives, distinct from the
 * answer bubble -- see createReasoningZone below. This is UNVERIFIED,
 * provisional model text; the answer bubble renders ONLY from the `answer`
 * frame's already-verified text, never from any reasoning_delta text. The
 * #208 generic "Reasoning locally..." spinner stage hands off to this live
 * zone the instant the first reasoning_delta frame arrives, so the two are
 * never shown redundantly at once. Once the answer renders, the reasoning
 * zone stays visible above it (no collapse/toggle -- the simplest option
 * that satisfies "don't hide it, don't show a redundant spinner").
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
    // #208 staged progress indicator: replaces the ~18-20s silent dead-air
    // between send and the answer with a spinner + status line that
    // advances through the real pipeline stages. Every status string below
    // is STATIC copy -- never interpolated with response/record data (no
    // PHI in this element, ever).
    //
    // Investigation finding (driving this design -- see the #208 PR): the
    // dev stack does NOT deliver SSE frames incrementally end-to-end. A
    // live capture (chrome devtools against a real /chat request) showed
    // every frame -- including `conversation`, which app/chat.py yields
    // BEFORE the planner even runs -- arriving in a single batch at the
    // very end of the wait; even the fetch() response headers do not
    // resolve until the whole request completes. So there is no real
    // signal available to mark the boundary between "dispatching tool
    // calls" and "the long local-model inference" -- by the time a
    // `tool_call` frame is observed, the entire wait is already over. The
    // REASONING_FALLBACK_MS timer below is what actually delivers the
    // "Reasoning locally..." stage during the long gap; a `tool_call` frame
    // (should the stack ever start flushing incrementally) advances the
    // same stage immediately and just pre-empts the timer.
    // -------------------------------------------------------------------
    var THINKING_STAGE_LABELS = {
        consulting: 'Consulting the chart…',
        reasoning: 'Reasoning locally (Qwen3-4B)…',
        verifying: 'Verifying claims against the record…'
    };
    var THINKING_GENERIC_LABEL = 'Thinking…';

    // How long to hold "Consulting the chart..." before assuming the wait
    // has moved into the long local-inference stage, absent any earlier
    // real signal (see the investigation note above). Small relative to
    // observed real-model latencies (~9-29s per chat-proxy's own comment).
    var THINKING_REASONING_FALLBACK_MS = 1500;

    function createThinkingIndicator(container) {
        var el = document.createElement('div');
        el.className = 'copilot-thinking';
        el.setAttribute('role', 'status');
        el.setAttribute('aria-live', 'polite');

        var spinner = document.createElement('span');
        spinner.className = 'copilot-thinking-spinner';
        spinner.setAttribute('aria-hidden', 'true');
        el.appendChild(spinner);

        var status = document.createElement('span');
        status.className = 'copilot-thinking-status';
        el.appendChild(status);

        container.appendChild(el);
        container.scrollTop = container.scrollHeight;

        return {
            setStage: function (stage) {
                status.textContent = Object.prototype.hasOwnProperty.call(THINKING_STAGE_LABELS, stage)
                    ? THINKING_STAGE_LABELS[stage]
                    : THINKING_GENERIC_LABEL;
            },
            remove: function () {
                if (el.parentNode) {
                    el.parentNode.removeChild(el);
                }
            }
        };
    }

    // -------------------------------------------------------------------
    // #213 live reasoning zone: the `reasoning_delta` SSE frame's text
    // types into this SEPARATE, clearly-labeled surface -- never the answer
    // bubble. This is the owner's non-negotiable UX rule: unverified,
    // provisional model text must never occupy the authoritative answer
    // slot (see the module docstring). `append` uses `textContent +=`, not
    // innerHTML, so streamed reasoning renders as inert text exactly like
    // every other model/record-derived string in this file.
    // -------------------------------------------------------------------
    function createReasoningZone(container) {
        var el = document.createElement('div');
        el.className = 'copilot-reasoning copilot-reasoning-active';
        el.setAttribute('role', 'status');
        el.setAttribute('aria-live', 'polite');

        var label = document.createElement('div');
        label.className = 'copilot-reasoning-label';
        label.textContent = THINKING_STAGE_LABELS.reasoning;
        el.appendChild(label);

        var text = document.createElement('div');
        text.className = 'copilot-reasoning-text';
        el.appendChild(text);

        container.appendChild(el);
        container.scrollTop = container.scrollHeight;

        // Reading scrollHeight forces a synchronous reflow; a reasoning
        // response streams hundreds of tokens, so coalesce the auto-scroll
        // to at most one reflow per animation frame instead of one per
        // token. The text append itself stays synchronous (callers/tests
        // observe it immediately). Falls back to an immediate scroll where
        // requestAnimationFrame is unavailable.
        var raf = window.requestAnimationFrame ? window.requestAnimationFrame.bind(window) : null;
        var scrollPending = false;
        function scheduleScroll() {
            if (!raf) {
                container.scrollTop = container.scrollHeight;
                return;
            }
            if (scrollPending) {
                return;
            }
            scrollPending = true;
            raf(function () {
                scrollPending = false;
                container.scrollTop = container.scrollHeight;
            });
        }

        return {
            append: function (delta) {
                text.textContent += delta;
                scheduleScroll();
            },
            // Stops the "actively streaming" cursor treatment once the
            // verified answer has arrived -- the zone itself stays visible
            // (see the module docstring's post-answer treatment note).
            finalize: function () {
                el.classList.remove('copilot-reasoning-active');
            },
            remove: function () {
                if (el.parentNode) {
                    el.parentNode.removeChild(el);
                }
            }
        };
    }

    // -------------------------------------------------------------------
    // #219 paced reveal: buffers `reasoning_delta` text and drains it into
    // the zone at a steady, readable rate instead of appending it the
    // instant it arrives. The tokens have genuinely already arrived (see
    // the #213 section of the module docstring) -- this paces the *reveal*
    // for readability, so the zone stays smoothly "typing" whether Ollama
    // delivers deltas spread out, in a sub-20ms burst, or as a single frame
    // at EOF. ~40 chars/sec (2 chars every 50ms) sits in the 30-45 chars/sec
    // readable target.
    //
    // Design decision (documented per the issue's acceptance criteria): the
    // verified `answer` frame must never be blocked by more than a small,
    // fixed cap. REVEAL_CAP_MS bounds how long waitForDrain() below may wait
    // before force-flushing whatever remains of the buffer -- NOT a
    // fast-forward-on-answer-arrival approach, because in this app's
    // request/response shape the whole SSE stream (reasoning_delta frames
    // AND the answer frame) is fully received before createChatController's
    // frame handler ever sees the `answer` frame (consumeSSEStream resolves
    // its promise only after the reader reports done) -- fast-forwarding
    // the instant the answer frame is *observed* would collapse the paced
    // reveal to ~0ms and defeat the feature. Bounding total drain time
    // instead keeps the reveal genuinely paced while still guaranteeing the
    // answer renders within REVEAL_CAP_MS.
    //
    // `prefers-reduced-motion` skips pacing entirely -- push() reveals the
    // full delta immediately and never starts a timer (same media-query
    // string the #208/#213 CSS already checks).
    // -------------------------------------------------------------------
    var REVEAL_TICK_MS = 50;
    var REVEAL_CHARS_PER_TICK = 2; // ~40 chars/sec
    var REVEAL_CAP_MS = 1500;

    function prefersReducedMotion() {
        return typeof window.matchMedia === 'function' &&
            window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    }

    function createReasoningRevealer(zone) {
        var buffer = '';
        var timer = null;
        var startedAt = 0;
        var waiters = [];

        function notifyWaiters() {
            var pending = waiters;
            waiters = [];
            for (var i = 0; i < pending.length; i++) {
                pending[i]();
            }
        }

        function stopTimer() {
            if (timer !== null) {
                clearInterval(timer);
                timer = null;
            }
        }

        function tick() {
            var chunk = buffer.slice(0, REVEAL_CHARS_PER_TICK);
            buffer = buffer.slice(chunk.length);
            if (chunk) {
                zone.append(chunk);
            }
            var capped = (Date.now() - startedAt) >= REVEAL_CAP_MS;
            if (capped && buffer.length > 0) {
                // Force-flush the remainder rather than let a large backlog
                // keep gating the answer past the documented cap.
                zone.append(buffer);
                buffer = '';
            }
            if (buffer.length === 0) {
                stopTimer();
                notifyWaiters();
            }
        }

        return {
            // Buffers `delta` and (re)starts the drain timer if it is not
            // already running.
            push: function (delta) {
                if (prefersReducedMotion()) {
                    zone.append(delta);
                    return;
                }
                buffer += delta;
                if (timer === null) {
                    startedAt = Date.now();
                    timer = setInterval(tick, REVEAL_TICK_MS);
                }
            },
            // Resolves once the buffer has fully drained, or REVEAL_CAP_MS
            // has elapsed since the current drain started (whichever comes
            // first -- tick() above force-flushes any remainder at the cap
            // before resolving). Resolves immediately if nothing is
            // pending.
            waitForDrain: function () {
                if (timer === null) {
                    return Promise.resolve();
                }
                return new Promise(function (resolve) {
                    waiters.push(resolve);
                });
            },
            // Hard stop for reset()/error paths, where the zone itself is
            // about to be detached: clears the timer and drops the buffer
            // without revealing it, so no further tick can mutate the (soon
            // detached) zone -- same "null the handle" cleanup discipline as
            // #208's stopThinking().
            stop: function () {
                stopTimer();
                buffer = '';
                notifyWaiters();
            }
        };
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
        verified: {
            label: 'Verified',
            icon: '✓',
            className: 'copilot-verdict-verified',
            meaning: 'Every claim matched an exact record value.'
        },
        partially_verified: {
            label: 'Partially verified',
            icon: '⚠',
            className: 'copilot-verdict-partial',
            meaning: 'Some claims could not be fully confirmed.'
        },
        blocked: {
            label: 'Blocked',
            icon: '✕',
            className: 'copilot-verdict-blocked',
            meaning: 'A safety conflict stopped the answer.'
        }
    };

    // Fixed display order for the first-open legend (P2.20) -- matches the
    // severity progression a user would expect: best case first, worst case
    // last.
    var VERDICT_ORDER = ['verified', 'partially_verified', 'blocked'];

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

    // -------------------------------------------------------------------
    // P2.20 first-open "about" state: a compact legend of the verdict
    // badges, reusing VERDICT_BADGES/renderVerdictBadge (above) so the
    // legend's badge markup and copy never diverge from what a real answer
    // renders. Populates `container` (the server-rendered, initially empty
    // <ul id="copilot-chat-about-legend">) once on init.
    // -------------------------------------------------------------------
    function renderAboutLegend(container) {
        for (var i = 0; i < VERDICT_ORDER.length; i++) {
            var key = VERDICT_ORDER[i];
            var row = document.createElement('li');
            row.className = 'copilot-about-legend-row';

            var badge = renderVerdictBadge(key);
            if (badge) {
                row.appendChild(badge);
            }

            var meaning = document.createElement('span');
            meaning.className = 'copilot-about-legend-meaning';
            meaning.textContent = VERDICT_BADGES[key].meaning;
            row.appendChild(meaning);

            container.appendChild(row);
        }
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
        // #172: input-side PHI deterrent (complements #157's export-side
        // scrub of this same comment). A placeholder alone disappears the
        // moment the clinician starts typing, so this persistent hint stays
        // visible for the life of the comment box -- textContent only, no
        // patient/model data ever flows into it (static copy).
        var commentHint = document.createElement('div');
        commentHint.className = 'copilot-feedback-comment-hint';
        commentHint.textContent = 'Feedback about the response only -- please avoid patient names, dates of birth, or other identifying details.';
        var commentInput = document.createElement('textarea');
        commentInput.className = 'copilot-feedback-comment-input';
        commentInput.setAttribute('placeholder', 'What went wrong with the response? (optional)');
        commentInput.setAttribute('maxlength', '2000');
        commentInput.setAttribute('aria-label', 'Feedback comment');
        var commentSendBtn = document.createElement('button');
        commentSendBtn.type = 'button';
        commentSendBtn.className = 'copilot-feedback-comment-send';
        commentSendBtn.textContent = 'Send';
        commentWrap.appendChild(commentHint);
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
            commentHint: commentHint,
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
    // Shown when the proxy reports no patient bound to the session (the P2.17
    // global launcher opens this panel on every page, including ones with no
    // patient selected). The pid is resolved server-side per request
    // (ChatProxyController), so this is the honest, always-current answer to
    // "you opened the Co-Pilot without a patient" -- no stale render-time
    // bake-in in the outer SPA shell.
    var NO_PATIENT_MESSAGE = 'Open a patient chart first to start a Co-Pilot conversation.';

    function createChatController(options) {
        var cachedToken = null;
        var conversationId = null;
        // #208: the thinking indicator's stop() for the in-flight send, if
        // any -- null when no send is outstanding. Lets reset() clear a
        // mid-flight indicator (and its fallback timer) even though the
        // indicator itself is a local var inside sendMessage.
        var activeThinking = null;
        // #213: same pattern as activeThinking, for the live reasoning zone
        // -- null when no send has streamed any reasoning yet. Lets reset()
        // clear a mid-flight zone even though it is a local var inside
        // sendMessage.
        var activeReasoningZone = null;
        // #221: per-turn currency guard. sendMessage() captures
        // `myTurn = ++turnSeq` at the start of a turn; reset() bumps turnSeq
        // again, so an in-flight turn's `myTurn` no longer equals `turnSeq`.
        // Every DOM-mutating continuation in sendMessage checks this before
        // writing, so a turn superseded by reset() (a patient switch firing
        // mid-wait, whose abandoned stream keeps delivering frames after the
        // transcript has moved on) cannot write its stale
        // answer/verification/reasoning into the current transcript. reset()
        // is the ONLY supersession: the send-lock below prevents a second,
        // overlapping send from starting while one is in flight, so no turn
        // is ever superseded by another sendMessage(). See issue #221.
        var turnSeq = 0;
        var redirect = options.redirectImpl || function (url) {
            window.location.assign(url);
        };

        // #221 send-lock: while a turn is streaming, the message input and its
        // submit button are disabled so the user cannot start a second,
        // overlapping send. That is what makes reset() the only supersession
        // (see turnSeq above): a superseded turn skips its own cleanup, so if
        // an overlapping send could supersede an in-flight turn it would strand
        // that turn's spinner/reasoning zone in the transcript. Re-enabled on
        // every sendMessage() exit path and by reset(). The submit button is
        // looked up from the form so no extra option needs threading through;
        // it is absent in unit tests that pass a bare <form>, hence the guard.
        function setSendEnabled(enabled) {
            options.inputEl.disabled = !enabled;
            var submitBtn = options.formEl.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = !enabled;
            }
        }

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
            // #221: this turn's identity -- see turnSeq above. Only reset()
            // supersedes an in-flight turn (the send-lock prevents overlapping
            // sends); stale() reports whether reset() has since fired.
            var myTurn = ++turnSeq;
            function stale() {
                return myTurn !== turnSeq;
            }
            // #221 send-lock: disable input/submit for the duration of this
            // turn; every exit path below re-enables it (see setSendEnabled).
            setSendEnabled(false);
            // The first-open "about" explainer (P2.20) gives way to the
            // transcript as soon as a conversation starts -- a no-op on
            // every send after the first (already hidden) and if no about
            // element was wired (e.g. tests that omit it).
            if (options.aboutEl) {
                options.aboutEl.classList.add('copilot-hidden');
            }
            appendMessage(options.messagesEl, 'user', text);
            // Per-turn, not per-conversation (unlike conversationId above): a
            // fresh P4.1 correlation id arrives on every `conversation` frame,
            // one per response, so the feedback widget below is tied to THIS
            // answer specifically.
            var responseCorrelationId = null;

            // #208: shown immediately on send -- no more silent dead-air
            // while the request is in flight. See the createThinkingIndicator
            // docstring for why a fallback timer (not just frame arrival)
            // drives the "Reasoning locally..." transition.
            var thinking = createThinkingIndicator(options.messagesEl);
            thinking.setStage('consulting');
            var reasoningFallbackTimer = setTimeout(function () {
                thinking.setStage('reasoning');
            }, THINKING_REASONING_FALLBACK_MS);
            function stopThinking() {
                clearTimeout(reasoningFallbackTimer);
                thinking.remove();
                activeThinking = null;
            }
            activeThinking = { stop: stopThinking };

            // #213: created lazily on the FIRST reasoning_delta frame (see
            // the frame handler below), so a turn with no reasoning ever
            // streamed (or a fallback planner double with no reasoning_delta
            // support -- see app.planner's module docstring) never shows an
            // empty zone.
            var reasoningZone = null;
            // #219: paces reasoningZone.append() calls -- see
            // createReasoningRevealer above. null until the first
            // reasoning_delta frame creates the zone, alongside it.
            var reasoningRevealer = null;
            function clearReasoningZone() {
                if (reasoningRevealer) {
                    reasoningRevealer.stop();
                    reasoningRevealer = null;
                }
                if (reasoningZone) {
                    reasoningZone.remove();
                    reasoningZone = null;
                }
                activeReasoningZone = null;
            }

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
                // The proxy rejects a request with no patient in session as a
                // 400 JSON body (not an SSE stream). Surface the specific
                // "open a patient first" hint for that case rather than the
                // generic unavailable message; any other JSON error stays
                // generic.
                if (resp.status === 400) {
                    return resp.json().then(function (data) {
                        // #221: this turn may have been superseded (e.g. a
                        // patient switch called reset()) while the 400 body
                        // was being parsed -- do not write into the current
                        // transcript on its behalf (reset() re-enables input).
                        if (stale()) {
                            return;
                        }
                        stopThinking();
                        setSendEnabled(true);
                        var noPatient = data && data.reason === 'no_patient_in_session';
                        appendMessage(
                            options.messagesEl,
                            'assistant',
                            noPatient ? NO_PATIENT_MESSAGE : UNAVAILABLE_MESSAGE
                        );
                    }).catch(function () {
                        if (stale()) {
                            return;
                        }
                        stopThinking();
                        setSendEnabled(true);
                        appendMessage(options.messagesEl, 'assistant', UNAVAILABLE_MESSAGE);
                    });
                }
                if (!resp.ok || !resp.body) {
                    throw new Error('chat proxy request failed');
                }
                var answerText = '';
                var verificationData = null;
                var hadError = false;
                return consumeSSEStream(resp.body.getReader(), function (frame) {
                    // #221: this turn's stream is not aborted on reset() -- the
                    // abandoned request keeps delivering frames after a patient
                    // switch cleared the transcript. A superseded turn's late
                    // frame (e.g. reasoning_delta/tool_call) must not touch the
                    // DOM or recreate the thinking/reasoning zone.
                    if (stale()) {
                        return;
                    }
                    if (frame.event === 'conversation' && frame.data) {
                        if (typeof frame.data.conversation_id === 'string') {
                            conversationId = frame.data.conversation_id;
                        }
                        if (typeof frame.data.correlation_id === 'string') {
                            responseCorrelationId = frame.data.correlation_id;
                        }
                    } else if (frame.event === 'tool_call') {
                        // The planner has moved past the chart-lookup tool
                        // dispatch it is documented to run before any real
                        // frame is emitted -- see the createThinkingIndicator
                        // docstring for why the fallback timer, not this
                        // frame, does the actual work of the transition in
                        // the current dev stack.
                        thinking.setStage('reasoning');
                    } else if (frame.event === 'reasoning_delta' && frame.data && typeof frame.data.text === 'string') {
                        if (!reasoningZone) {
                            // #213/#208 handoff: the generic spinner gives
                            // way to the live typing zone the instant real
                            // reasoning tokens start arriving -- never shown
                            // redundantly alongside it.
                            stopThinking();
                            reasoningZone = createReasoningZone(options.messagesEl);
                            reasoningRevealer = createReasoningRevealer(reasoningZone);
                            activeReasoningZone = { stop: clearReasoningZone };
                        }
                        // #219: paced reveal, not an immediate append -- see
                        // createReasoningRevealer.
                        reasoningRevealer.push(frame.data.text);
                    } else if (frame.event === 'answer' && frame.data && typeof frame.data.answer === 'string') {
                        answerText = frame.data.answer;
                    } else if (frame.event === 'verification' && frame.data) {
                        verificationData = frame.data;
                        thinking.setStage('verifying');
                    } else if (frame.event === 'error') {
                        hadError = true;
                    }
                }).then(function () {
                    // #221: guards the hadError/no-op branches below (the
                    // answer branch's own render point is guarded again
                    // after the additional waitForDrain() await, since
                    // reset() can fire during that wait even when it did not
                    // fire before it).
                    if (stale()) {
                        return;
                    }
                    if (answerText) {
                        // #219: the verified answer must not be blocked on
                        // the full paced reveal -- wait only up to
                        // REVEAL_CAP_MS (waitForDrain force-flushes and
                        // resolves at the cap; resolves immediately if no
                        // reasoning was ever streamed).
                        var drainWait = reasoningRevealer ? reasoningRevealer.waitForDrain() : Promise.resolve();
                        return drainWait.then(function () {
                            // #221: reset() (a patient switch) may have fired
                            // WHILE waiting for the paced reveal to drain --
                            // this is the primary race the guard defends
                            // against: the answer/verification must not
                            // render into a transcript that has since been
                            // cleared for a different patient (reset()
                            // re-enables input on that path).
                            if (stale()) {
                                return;
                            }
                            stopThinking();
                            setSendEnabled(true);
                            // #213: the verified answer has arrived -- stop
                            // the zone's "actively streaming" cursor
                            // treatment, but keep it visible above the
                            // answer bubble (minimal post-answer treatment;
                            // see the module docstring).
                            if (reasoningZone) {
                                reasoningZone.finalize();
                            }
                            activeReasoningZone = null;
                            appendMessage(options.messagesEl, 'assistant', answerText);
                            // Pending verification payloads (verdict null)
                            // render nothing; a populated one adds the
                            // badge/chips/banner.
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
                        });
                    }
                    stopThinking();
                    setSendEnabled(true);
                    if (hadError) {
                        // #213: an errored turn never gets a verified answer,
                        // so any in-flight (necessarily unverified/partial)
                        // reasoning text is cleared rather than left stuck
                        // above the error bubble -- same treatment as the
                        // #208 spinner on this branch.
                        clearReasoningZone();
                        appendMessage(options.messagesEl, 'assistant', UNAVAILABLE_MESSAGE);
                        // Self-heal: drop the conversation id so the NEXT send
                        // starts a fresh conversation instead of retrying the
                        // same failed one. Load-bearing for the global launcher
                        // (P2.17): if the panel is left OPEN across a patient
                        // switch, no open-time reset fires, and the stale
                        // conversation id + new session pid is hard-rejected by
                        // the agent's pid-binding check (chat.py) as an `error`
                        // frame. Without this clear, every subsequent send in
                        // that still-open panel repeats the identical rejection
                        // and the panel wedges permanently.
                        conversationId = null;
                    } else {
                        // #213: neither a verified answer nor an error frame
                        // (e.g. an empty-string answer payload, or a clean
                        // mid-stream truncation after reasoning). Any zone
                        // built from the streamed reasoning would otherwise be
                        // orphaned in the DOM with its cursor still blinking,
                        // and activeReasoningZone left dangling -- clear it,
                        // like the hadError branch does.
                        clearReasoningZone();
                    }
                });
            }).catch(function () {
                // #221: a superseded turn's request/stream failure must not
                // paint the generic error bubble into a transcript that has
                // since moved on to a different patient/turn (reset()
                // re-enables input on that path).
                if (stale()) {
                    return;
                }
                stopThinking();
                setSendEnabled(true);
                clearReasoningZone();
                appendMessage(options.messagesEl, 'assistant', UNAVAILABLE_MESSAGE);
            });
        }

        function handleSubmit(evt) {
            evt.preventDefault();
            // #221 send-lock: a send is already streaming (input disabled) --
            // ignore the re-submit rather than starting a second, overlapping
            // turn that would supersede and orphan the first turn's spinner.
            if (options.inputEl.disabled) {
                return;
            }
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

        // Start a fresh conversation: drop the cached conversation id (the
        // next send opens a new agent conversation bound to the CURRENT
        // patient) and clear the visible transcript. Used by the P2.17 global
        // launcher, whose panel lives in the never-reloaded main.php shell and
        // so must not carry one patient's conversation into another's after a
        // patient switch -- the agent binds a conversation to its patient and
        // rejects a mismatched pid. The cached bearer token is user-scoped,
        // not patient-scoped, so it is intentionally kept.
        function reset() {
            // #221: invalidate any turn currently in flight -- see turnSeq
            // above. Any `myTurn` captured by an earlier sendMessage() call
            // no longer equals turnSeq after this, so that turn's pending
            // continuations (and any late frames its still-open stream goes
            // on to deliver) become no-ops.
            turnSeq++;
            // #208: a reset() during an in-flight send (e.g. a patient
            // switch mid-wait) must stop the fallback timer too, not just
            // rely on the textContent clear below to visually drop the
            // indicator's DOM node.
            if (activeThinking) {
                activeThinking.stop();
            }
            // #213: same reasoning as activeThinking above -- clear a
            // mid-flight reasoning zone explicitly (not just relying on the
            // messagesEl.textContent wipe below) so a late, stale
            // reasoning_delta from an abandoned request cannot mutate a
            // detached node's text for no visible effect.
            if (activeReasoningZone) {
                activeReasoningZone.stop();
            }
            // #221 send-lock: reset() supersedes any in-flight turn (whose own
            // exit paths are now skipped by the stale() guard), so it must
            // re-enable the input itself -- otherwise a patient switch mid-send
            // would leave the new patient's panel permanently unable to send.
            setSendEnabled(true);
            conversationId = null;
            options.messagesEl.textContent = '';
            // A fresh conversation is a fresh first-open: bring the about
            // explainer back so the P2.17 global launcher's never-reloaded
            // panel shows it again after a patient switch.
            if (options.aboutEl) {
                options.aboutEl.classList.remove('copilot-hidden');
            }
        }

        return { init: init, sendMessage: sendMessage, reset: reset };
    }

    // The single chat controller wired to the live DOM panel, captured so the
    // launcher toggle (copilot.js) can reset it on open via
    // window.CopilotChat.resetActiveConversation() below.
    var activeController = null;

    function initFromDom() {
        var panel = document.getElementById('copilot-chat-panel');
        var messagesEl = document.getElementById('copilot-chat-messages');
        var formEl = document.getElementById('copilot-chat-form');
        var inputEl = document.getElementById('copilot-chat-input');
        if (!panel || !messagesEl || !formEl || !inputEl || !window.CopilotContext) {
            return;
        }

        // P2.20 first-open "about" explainer -- optional elements (older
        // cached markup could lack them), so init tolerates their absence.
        var aboutEl = document.getElementById('copilot-chat-about');
        var aboutLegendEl = document.getElementById('copilot-chat-about-legend');
        if (aboutLegendEl) {
            renderAboutLegend(aboutLegendEl);
        }

        // This script's own URL gives us the module's public/ base path
        // without needing the server to thread it through CopilotContext.
        var baseUrl = CURRENT_SCRIPT_SRC.replace(/\/assets\/js\/copilot-chat\.js(\?.*)?$/, '');

        activeController = createChatController({
            messagesEl: messagesEl,
            formEl: formEl,
            inputEl: inputEl,
            aboutEl: aboutEl,
            context: window.CopilotContext,
            brokerUrl: baseUrl + '/ajax.php',
            proxyUrl: baseUrl + '/chat-proxy.php',
            feedbackUrl: baseUrl + '/feedback-proxy.php',
            authorizeUrl: baseUrl + '/oauth-authorize.php',
            fetchImpl: window.fetch.bind(window)
        });
        activeController.init();
    }

    // Reset the live panel's conversation (see createChatController.reset).
    // A no-op before the DOM panel is wired or if it has no chat form.
    function resetActiveConversation() {
        if (activeController) {
            activeController.reset();
        }
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
        createThinkingIndicator: createThinkingIndicator,
        createReasoningZone: createReasoningZone,
        createReasoningRevealer: createReasoningRevealer,
        createChatController: createChatController,
        verdictBadgeInfo: verdictBadgeInfo,
        renderVerdictBadge: renderVerdictBadge,
        renderAboutLegend: renderAboutLegend,
        renderWarningBanner: renderWarningBanner,
        renderVerification: renderVerification,
        buildFeedbackPayload: buildFeedbackPayload,
        renderFeedbackWidget: renderFeedbackWidget,
        applyFeedbackPendingState: applyFeedbackPendingState,
        applyFeedbackSuccessState: applyFeedbackSuccessState,
        applyFeedbackErrorState: applyFeedbackErrorState,
        attachFeedbackHandlers: attachFeedbackHandlers,
        resetActiveConversation: resetActiveConversation
    };
})(window, document);
