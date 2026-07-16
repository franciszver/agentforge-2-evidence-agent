/**
 * @jest-environment jsdom
 */

/**
 * Tests for the P4.4 feedback-button UI in
 * interface/modules/custom_modules/oe-module-clinical-copilot/public/assets/js/copilot-chat.js
 *
 * Covers the pure/testable surface: the response correlation-id association
 * (renderFeedbackWidget), the POST payload shape (buildFeedbackPayload), the
 * pending/success/error DOM states, and the click-orchestration guard
 * against double-submit (attachFeedbackHandlers) -- all driven with fake
 * `fetchImpl`/`ensureToken` seams, no real network/stream used, same
 * discipline as clinical-copilot-chat.test.js / clinical-copilot-verification.test.js.
 *
 * Run with: npm test -- tests/js/clinical-copilot-feedback.test.js
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

const fs = require('fs');
const path = require('path');

// Load the module once — it self-executes and attaches to window.
const src = fs.readFileSync(
    path.resolve(
        __dirname,
        '../../interface/modules/custom_modules/oe-module-clinical-copilot/public/assets/js/copilot-chat.js'
    ),
    'utf8'
);
new Function('window', 'document', src)(global.window, global.document);

const {
    buildFeedbackPayload,
    renderFeedbackWidget,
    applyFeedbackPendingState,
    applyFeedbackSuccessState,
    applyFeedbackErrorState,
    attachFeedbackHandlers
} = global.window.CopilotChat;

function makeContainer() {
    const div = document.createElement('div');
    document.body.appendChild(div);
    return div;
}

/** A deferred promise, so a test can control exactly when fetchImpl resolves. */
function deferred() {
    let resolve;
    const promise = new Promise((res) => {
        resolve = res;
    });
    return { promise, resolve };
}

/**
 * Drains every currently-queued microtask (promise .then chain), regardless
 * of chain depth -- unlike a fixed number of `await Promise.resolve()`
 * hops, a macrotask (setTimeout) is only run after the whole microtask
 * queue empties, so this is robust to attachFeedbackHandlers' internal
 * .then chain length changing.
 */
function flushPromises() {
    return new Promise((resolve) => setTimeout(resolve, 0));
}

afterEach(() => {
    document.body.innerHTML = '';
});

// ---------------------------------------------------------------------------
// buildFeedbackPayload — POST payload shape
// ---------------------------------------------------------------------------
describe('buildFeedbackPayload', () => {
    test('carries the CSRF token, bearer token, correlation id, and thumb', () => {
        const payload = buildFeedbackPayload({ csrfToken: 'csrf-1' }, 'tok-1', 'corr-1', 'up', null);

        expect(payload).toEqual({
            csrf_token_form: 'csrf-1',
            token: 'tok-1',
            correlation_id: 'corr-1',
            thumb: 'up'
        });
    });

    test('omits comment when not provided', () => {
        const payload = buildFeedbackPayload({ csrfToken: 'csrf-1' }, 'tok-1', 'corr-1', 'down', null);

        expect(payload).not.toHaveProperty('comment');
    });

    test('includes comment when provided', () => {
        const payload = buildFeedbackPayload({ csrfToken: 'csrf-1' }, 'tok-1', 'corr-1', 'down', 'Missed the A1C.');

        expect(payload.comment).toBe('Missed the A1C.');
    });
});

// ---------------------------------------------------------------------------
// renderFeedbackWidget — correlation-id association + DOM shape
// ---------------------------------------------------------------------------
describe('renderFeedbackWidget', () => {
    test('associates the widget with the response correlation id', () => {
        const container = makeContainer();
        renderFeedbackWidget(container, 'corr-42');

        const wrapper = container.querySelector('.copilot-feedback');
        expect(wrapper.getAttribute('data-correlation-id')).toBe('corr-42');
    });

    test('renders real <button> elements with accessible labels and thumb markers', () => {
        const container = makeContainer();
        const widget = renderFeedbackWidget(container, 'corr-1');

        expect(widget.upBtn.tagName).toBe('BUTTON');
        expect(widget.upBtn.type).toBe('button');
        expect(widget.upBtn.getAttribute('aria-label')).toBe('Helpful');
        expect(widget.upBtn.getAttribute('data-thumb')).toBe('up');

        expect(widget.downBtn.tagName).toBe('BUTTON');
        expect(widget.downBtn.getAttribute('aria-label')).toBe('Not helpful');
        expect(widget.downBtn.getAttribute('data-thumb')).toBe('down');
    });

    test('the comment box is hidden until revealed', () => {
        const container = makeContainer();
        const widget = renderFeedbackWidget(container, 'corr-1');

        expect(widget.commentWrap.classList.contains('copilot-hidden')).toBe(true);
    });

    test('appends to the given container without clearing prior content', () => {
        const container = makeContainer();
        container.appendChild(document.createElement('div'));
        renderFeedbackWidget(container, 'corr-1');

        expect(container.children).toHaveLength(2);
    });
});

// ---------------------------------------------------------------------------
// State mutators — pending / success / error
// ---------------------------------------------------------------------------
describe('feedback widget state', () => {
    test('pending state disables both buttons', () => {
        const container = makeContainer();
        const widget = renderFeedbackWidget(container, 'corr-1');

        applyFeedbackPendingState(widget);

        expect(widget.upBtn.disabled).toBe(true);
        expect(widget.downBtn.disabled).toBe(true);
    });

    test('success state disables both buttons, marks the chosen thumb, and shows a status message', () => {
        const container = makeContainer();
        const widget = renderFeedbackWidget(container, 'corr-1');

        applyFeedbackSuccessState(widget, 'down');

        expect(widget.upBtn.disabled).toBe(true);
        expect(widget.downBtn.disabled).toBe(true);
        expect(widget.wrapper.classList.contains('copilot-feedback-submitted')).toBe(true);
        expect(widget.downBtn.classList.contains('copilot-feedback-selected')).toBe(true);
        expect(widget.upBtn.classList.contains('copilot-feedback-selected')).toBe(false);
        expect(widget.status.textContent).toMatch(/thanks/i);
    });

    test('success state marks the up thumb when thumb is "up"', () => {
        const container = makeContainer();
        const widget = renderFeedbackWidget(container, 'corr-1');

        applyFeedbackSuccessState(widget, 'up');

        expect(widget.upBtn.classList.contains('copilot-feedback-selected')).toBe(true);
        expect(widget.downBtn.classList.contains('copilot-feedback-selected')).toBe(false);
    });

    test('error state re-enables both buttons and shows a retry message', () => {
        const container = makeContainer();
        const widget = renderFeedbackWidget(container, 'corr-1');
        applyFeedbackPendingState(widget);

        applyFeedbackErrorState(widget);

        expect(widget.upBtn.disabled).toBe(false);
        expect(widget.downBtn.disabled).toBe(false);
        expect(widget.status.textContent).toMatch(/try again/i);
    });
});

// ---------------------------------------------------------------------------
// attachFeedbackHandlers — click orchestration
// ---------------------------------------------------------------------------
describe('attachFeedbackHandlers', () => {
    function makeOptions(overrides) {
        return Object.assign(
            {
                context: { csrfToken: 'csrf-1' },
                ensureToken: () => Promise.resolve('tok-1'),
                fetchImpl: jest.fn(() => Promise.resolve({ ok: true })),
                feedbackUrl: '/feedback-proxy.php'
            },
            overrides
        );
    }

    test('clicking the up button posts the correct payload to the feedback proxy', async () => {
        const container = makeContainer();
        const widget = renderFeedbackWidget(container, 'corr-7');
        const fetchImpl = jest.fn(() => Promise.resolve({ ok: true }));
        attachFeedbackHandlers(widget, 'corr-7', makeOptions({ fetchImpl: fetchImpl }));

        widget.upBtn.dispatchEvent(new window.Event('click'));
        await flushPromises();
        expect(fetchImpl).toHaveBeenCalledTimes(1);
        const [url, init] = fetchImpl.mock.calls[0];
        expect(url).toBe('/feedback-proxy.php');
        expect(init.method).toBe('POST');
        const body = JSON.parse(init.body);
        expect(body).toEqual({
            csrf_token_form: 'csrf-1',
            token: 'tok-1',
            correlation_id: 'corr-7',
            thumb: 'up'
        });
    });

    test('a successful submit applies the success state', async () => {
        const container = makeContainer();
        const widget = renderFeedbackWidget(container, 'corr-1');
        attachFeedbackHandlers(widget, 'corr-1', makeOptions());

        await attachFeedbackHandlersClick(widget.upBtn);

        expect(widget.wrapper.classList.contains('copilot-feedback-submitted')).toBe(true);
        expect(widget.upBtn.classList.contains('copilot-feedback-selected')).toBe(true);
    });

    test('a failed submit applies the error state and allows retry', async () => {
        const container = makeContainer();
        const widget = renderFeedbackWidget(container, 'corr-1');
        const fetchImpl = jest.fn(() => Promise.resolve({ ok: false }));
        attachFeedbackHandlers(widget, 'corr-1', makeOptions({ fetchImpl: fetchImpl }));

        await attachFeedbackHandlersClick(widget.upBtn);

        expect(widget.status.textContent).toMatch(/try again/i);
        expect(widget.upBtn.disabled).toBe(false);
        expect(widget.wrapper.classList.contains('copilot-feedback-submitted')).toBe(false);
    });

    test('a rejected ensureToken (broker failure) applies the error state', async () => {
        const container = makeContainer();
        const widget = renderFeedbackWidget(container, 'corr-1');
        attachFeedbackHandlers(
            widget,
            'corr-1',
            makeOptions({ ensureToken: () => Promise.reject(new Error('token broker request failed')) })
        );

        await attachFeedbackHandlersClick(widget.upBtn);

        expect(widget.status.textContent).toMatch(/try again/i);
    });

    test('a second click while a submission is in flight does not send a second request (no double-submit)', async () => {
        const container = makeContainer();
        const widget = renderFeedbackWidget(container, 'corr-1');
        const inFlight = deferred();
        const fetchImpl = jest.fn(() => inFlight.promise);
        attachFeedbackHandlers(widget, 'corr-1', makeOptions({ fetchImpl: fetchImpl }));

        widget.upBtn.dispatchEvent(new window.Event('click'));
        widget.upBtn.dispatchEvent(new window.Event('click'));
        widget.downBtn.dispatchEvent(new window.Event('click'));
        await flushPromises();
        expect(fetchImpl).toHaveBeenCalledTimes(1);

        inFlight.resolve({ ok: true });
        await flushPromises();
    });

    test('clicking again after a successful submit does not send a second request', async () => {
        const container = makeContainer();
        const widget = renderFeedbackWidget(container, 'corr-1');
        const fetchImpl = jest.fn(() => Promise.resolve({ ok: true }));
        attachFeedbackHandlers(widget, 'corr-1', makeOptions({ fetchImpl: fetchImpl }));

        await attachFeedbackHandlersClick(widget.upBtn);
        expect(fetchImpl).toHaveBeenCalledTimes(1);

        // Buttons are disabled post-success, but even a synthetic click
        // bypassing that must not re-submit (state guard, not just the
        // disabled attribute).
        widget.upBtn.dispatchEvent(new window.Event('click'));
        await flushPromises();
        expect(fetchImpl).toHaveBeenCalledTimes(1);
    });

    test('clicking down reveals the optional comment box after a successful submit', async () => {
        const container = makeContainer();
        const widget = renderFeedbackWidget(container, 'corr-1');
        attachFeedbackHandlers(widget, 'corr-1', makeOptions());

        expect(widget.commentWrap.classList.contains('copilot-hidden')).toBe(true);

        await attachFeedbackHandlersClick(widget.downBtn);

        expect(widget.commentWrap.classList.contains('copilot-hidden')).toBe(false);
    });

    test('clicking up does not reveal the comment box', async () => {
        const container = makeContainer();
        const widget = renderFeedbackWidget(container, 'corr-1');
        attachFeedbackHandlers(widget, 'corr-1', makeOptions());

        await attachFeedbackHandlersClick(widget.upBtn);

        expect(widget.commentWrap.classList.contains('copilot-hidden')).toBe(true);
    });

    test('submitting a comment posts thumb=down with the comment text', async () => {
        const container = makeContainer();
        const widget = renderFeedbackWidget(container, 'corr-9');
        const fetchImpl = jest.fn(() => Promise.resolve({ ok: true }));
        attachFeedbackHandlers(widget, 'corr-9', makeOptions({ fetchImpl: fetchImpl }));

        widget.commentInput.value = 'Missed the recent A1C.';
        widget.commentSendBtn.dispatchEvent(new window.Event('click'));
        await flushPromises();
        expect(fetchImpl).toHaveBeenCalledTimes(1);
        const body = JSON.parse(fetchImpl.mock.calls[0][1].body);
        expect(body.thumb).toBe('down');
        expect(body.comment).toBe('Missed the recent A1C.');
    });

    test('submitting an empty comment is a no-op', async () => {
        const container = makeContainer();
        const widget = renderFeedbackWidget(container, 'corr-1');
        const fetchImpl = jest.fn(() => Promise.resolve({ ok: true }));
        attachFeedbackHandlers(widget, 'corr-1', makeOptions({ fetchImpl: fetchImpl }));

        widget.commentInput.value = '   ';
        widget.commentSendBtn.dispatchEvent(new window.Event('click'));
        await flushPromises();
        expect(fetchImpl).not.toHaveBeenCalled();
    });

    test('a successful comment submit hides the comment box', async () => {
        const container = makeContainer();
        const widget = renderFeedbackWidget(container, 'corr-1');
        attachFeedbackHandlers(widget, 'corr-1', makeOptions());
        widget.commentWrap.classList.remove('copilot-hidden');

        widget.commentInput.value = 'Detail here.';
        widget.commentSendBtn.dispatchEvent(new window.Event('click'));
        await flushPromises();
        expect(widget.commentWrap.classList.contains('copilot-hidden')).toBe(true);
    });

    /** Dispatches a click and flushes the fake (already-resolved) fetch/ensureToken chain to settle. */
    async function attachFeedbackHandlersClick(button) {
        button.dispatchEvent(new window.Event('click'));
        await flushPromises();
    }
});
