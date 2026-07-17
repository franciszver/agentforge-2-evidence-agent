/**
 * @jest-environment jsdom
 */

/**
 * Tests for interface/modules/custom_modules/oe-module-clinical-copilot/public/assets/js/copilot-chat.js
 *
 * Covers the P2.14 chat panel's pure/testable surface: SSE frame parsing
 * (partial chunks split across reads, multi-event buffers, comment/keep-alive
 * lines, malformed data), the reader-consumption loop, and DOM rendering
 * (incremental append, XSS safety -- assistant/user text is always rendered
 * via textContent, never innerHTML).
 *
 * Run with: npm test -- tests/js/clinical-copilot-chat.test.js
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

const fs = require('fs');
const path = require('path');
const { TextEncoder, TextDecoder } = require('util');

// jsdom does not provide these globals (Jest 28+); the real fetch API
// streaming path needs them to decode Uint8Array chunks into text.
global.TextEncoder = global.TextEncoder || TextEncoder;
global.TextDecoder = global.TextDecoder || TextDecoder;

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
    createSSEFrameParser,
    consumeSSEStream,
    appendMessage,
    createChatController,
    renderAboutLegend,
    createThinkingIndicator,
    createReasoningZone,
    createReasoningRevealer
} = global.window.CopilotChat;

const encoder = new TextEncoder();

function encode(str) {
    return encoder.encode(str);
}

/** A fake reader that yields the given chunks (strings) in sequence, then done. */
function fakeReader(chunks) {
    let i = 0;
    return {
        read: function () {
            if (i < chunks.length) {
                const value = encode(chunks[i]);
                i++;
                return Promise.resolve({ done: false, value: value });
            }
            return Promise.resolve({ done: true, value: undefined });
        }
    };
}

/**
 * A reader whose read() only resolves once the test explicitly pushes a
 * chunk (or finishes the stream) -- lets a test observe the thinking
 * indicator's status text BETWEEN frames rather than only before/after the
 * whole stream resolves (unlike fakeReader, which resolves every chunk on
 * an already-settled promise).
 */
function pausableReader() {
    const queue = [];
    let pendingResolve = null;

    function deliver(result) {
        if (pendingResolve) {
            const resolve = pendingResolve;
            pendingResolve = null;
            resolve(result);
        } else {
            queue.push(result);
        }
    }

    return {
        push: function (chunk) {
            deliver({ done: false, value: encode(chunk) });
        },
        finish: function () {
            deliver({ done: true, value: undefined });
        },
        read: function () {
            if (queue.length > 0) {
                return Promise.resolve(queue.shift());
            }
            return new Promise((resolve) => {
                pendingResolve = resolve;
            });
        }
    };
}

/** Flushes pending microtasks (promise chains) queued by the code under test. */
function flushMicrotasks() {
    return new Promise((resolve) => setTimeout(resolve, 0));
}

// ---------------------------------------------------------------------------
// createSSEFrameParser — pure frame parsing
// ---------------------------------------------------------------------------
describe('createSSEFrameParser', () => {
    test('parses a single complete frame', () => {
        const parser = createSSEFrameParser();
        const frames = parser.push('event: conversation\ndata: {"conversation_id":"abc"}\n\n');

        expect(frames).toHaveLength(1);
        expect(frames[0].event).toBe('conversation');
        expect(frames[0].data).toEqual({ conversation_id: 'abc' });
        expect(frames[0].error).toBeNull();
    });

    test('parses multiple frames in a single chunk', () => {
        const parser = createSSEFrameParser();
        const frames = parser.push(
            'event: conversation\ndata: {"conversation_id":"abc"}\n\n' +
            'event: answer\ndata: {"answer":"hi"}\n\n' +
            'event: done\ndata: {}\n\n'
        );

        expect(frames.map((f) => f.event)).toEqual(['conversation', 'answer', 'done']);
        expect(frames[2].data).toEqual({});
    });

    test('handles a frame split across two pushes (partial chunk)', () => {
        const parser = createSSEFrameParser();
        const first = parser.push('event: answer\ndata: {"ans');
        expect(first).toHaveLength(0);

        const second = parser.push('wer":"partial done"}\n\n');
        expect(second).toHaveLength(1);
        expect(second[0].event).toBe('answer');
        expect(second[0].data).toEqual({ answer: 'partial done' });
    });

    test('handles a split that lands mid-boundary (between the two newlines)', () => {
        const parser = createSSEFrameParser();
        const first = parser.push('event: done\ndata: {}\n');
        expect(first).toHaveLength(0);

        const second = parser.push('\n');
        expect(second).toHaveLength(1);
        expect(second[0].event).toBe('done');
    });

    test('handles many small chunks reassembling one frame', () => {
        const parser = createSSEFrameParser();
        const whole = 'event: tool_call\ndata: {"tool":"get_medications","args":{},"error":null}\n\n';
        let frames = [];
        for (let i = 0; i < whole.length; i++) {
            frames = frames.concat(parser.push(whole[i]));
        }

        expect(frames).toHaveLength(1);
        expect(frames[0].event).toBe('tool_call');
        expect(frames[0].data.tool).toBe('get_medications');
    });

    test('ignores SSE comment / keep-alive lines', () => {
        const parser = createSSEFrameParser();
        const frames = parser.push(': keep-alive\n\nevent: done\ndata: {}\n\n');

        expect(frames).toHaveLength(1);
        expect(frames[0].event).toBe('done');
    });

    test('surfaces malformed JSON in a data line as a frame-level error, not a throw', () => {
        const parser = createSSEFrameParser();
        expect(() => {
            const frames = parser.push('event: answer\ndata: {not valid json\n\n');
            expect(frames).toHaveLength(1);
            expect(frames[0].data).toBeNull();
            expect(frames[0].error).toEqual(expect.any(String));
        }).not.toThrow();
    });

    test('treats an empty data payload as an empty object', () => {
        const parser = createSSEFrameParser();
        const frames = parser.push('event: done\ndata: \n\n');

        expect(frames[0].data).toEqual({});
    });

    test('retains an unterminated trailing partial frame across pushes without emitting it', () => {
        const parser = createSSEFrameParser();
        const frames = parser.push('event: answer\ndata: {"answer":"ok"}\n\nevent: done\ndata: {}');

        // The done frame's trailing "\n\n" has not arrived yet.
        expect(frames.map((f) => f.event)).toEqual(['answer']);

        const more = parser.push('\n\n');
        expect(more.map((f) => f.event)).toEqual(['done']);
    });
});

// ---------------------------------------------------------------------------
// consumeSSEStream — reader-consumption loop (drives the parser from a
// ReadableStreamDefaultReader-shaped object; no real network/stream used)
// ---------------------------------------------------------------------------
describe('consumeSSEStream', () => {
    test('decodes and parses frames delivered across multiple reader.read() calls', async () => {
        const reader = fakeReader([
            'event: conversation\ndata: {"conversation_id":"c1"}\n\n',
            'event: tool_call\ndata: {"tool":"get_medi',
            'cations","args":{},"error":null}\n\n',
            'event: answer\ndata: {"answer":"She takes lisinopril."}\n\nevent: done\ndata: {}\n\n'
        ]);

        const received = [];
        await consumeSSEStream(reader, (frame) => received.push(frame));

        expect(received.map((f) => f.event)).toEqual(['conversation', 'tool_call', 'answer', 'done']);
        expect(received[2].data.answer).toBe('She takes lisinopril.');
    });

    test('stops when the reader reports done, even mid-buffer', async () => {
        const reader = fakeReader(['event: conversation\ndata: {"conversation_id":"c1"}\n\n']);

        const received = [];
        await consumeSSEStream(reader, (frame) => received.push(frame));

        expect(received).toHaveLength(1);
    });
});

// ---------------------------------------------------------------------------
// appendMessage — DOM rendering, XSS safety
// ---------------------------------------------------------------------------
describe('appendMessage', () => {
    afterEach(() => {
        document.body.innerHTML = '';
    });

    function makeContainer() {
        const div = document.createElement('div');
        document.body.appendChild(div);
        return div;
    }

    test('appends a user bubble with the expected classes and text', () => {
        const container = makeContainer();
        const el = appendMessage(container, 'user', 'What meds is she on?');

        expect(container.children).toHaveLength(1);
        expect(el.className).toContain('copilot-chat-message-user');
        expect(el.textContent).toBe('What meds is she on?');
    });

    test('appends an assistant bubble with the expected classes and text', () => {
        const container = makeContainer();
        const el = appendMessage(container, 'assistant', 'She takes lisinopril.');

        expect(el.className).toContain('copilot-chat-message-assistant');
        expect(el.textContent).toBe('She takes lisinopril.');
    });

    test('multiple calls append incrementally without clearing prior messages', () => {
        const container = makeContainer();
        appendMessage(container, 'user', 'first');
        appendMessage(container, 'assistant', 'second');
        appendMessage(container, 'user', 'third');

        expect(container.children).toHaveLength(3);
        expect(container.children[0].textContent).toBe('first');
        expect(container.children[2].textContent).toBe('third');
    });

    test('renders a <script> payload as inert text, never as a DOM element (XSS safety)', () => {
        const container = makeContainer();
        const payload = '<script>window.__pwned = true;</script>';
        const el = appendMessage(container, 'assistant', payload);

        expect(el.textContent).toBe(payload);
        expect(container.querySelector('script')).toBeNull();
        expect(window.__pwned).toBeUndefined();
    });

    test('renders an <img onerror> payload as inert text, never as a DOM element (XSS safety)', () => {
        const container = makeContainer();
        const payload = '<img src=x onerror="window.__pwned = true">';
        const el = appendMessage(container, 'assistant', payload);

        expect(el.textContent).toBe(payload);
        expect(container.querySelector('img')).toBeNull();
        expect(window.__pwned).toBeUndefined();
    });

    test('never sets innerHTML on the container with raw message text', () => {
        const container = makeContainer();
        appendMessage(container, 'assistant', '<b>bold</b>');

        // If innerHTML had been used, this would be a real <b> element.
        expect(container.querySelector('b')).toBeNull();
    });
});

// ---------------------------------------------------------------------------
// createChatController — consent-required redirect (#124 Phase 3)
// ---------------------------------------------------------------------------
describe('createChatController consent-required handling', () => {
    const AUTHORIZE_URL = 'https://host/base/oauth-authorize.php';

    afterEach(() => {
        document.body.innerHTML = '';
    });

    function makeController(fetchImpl, redirectImpl) {
        const messagesEl = document.createElement('div');
        document.body.appendChild(messagesEl);

        return {
            messagesEl: messagesEl,
            controller: createChatController({
                messagesEl: messagesEl,
                formEl: document.createElement('form'),
                inputEl: document.createElement('textarea'),
                context: { csrfToken: 'csrf' },
                brokerUrl: 'https://host/base/ajax.php',
                proxyUrl: 'https://host/base/chat-proxy.php',
                feedbackUrl: 'https://host/base/feedback-proxy.php',
                authorizeUrl: AUTHORIZE_URL,
                fetchImpl: fetchImpl,
                redirectImpl: redirectImpl
            })
        };
    }

    test('a consent_required broker response redirects to the authorize entry and never calls the proxy', async () => {
        const redirectImpl = jest.fn();
        const fetchImpl = jest.fn().mockResolvedValue({
            ok: true,
            json: () => Promise.resolve({ consent_required: true })
        });

        const { messagesEl, controller } = makeController(fetchImpl, redirectImpl);
        controller.sendMessage('What meds is she on?');

        // Flush the microtask/promise chain triggered by ensureToken.
        await new Promise((resolve) => setTimeout(resolve, 0));

        expect(redirectImpl).toHaveBeenCalledTimes(1);
        expect(redirectImpl).toHaveBeenCalledWith(AUTHORIZE_URL);
        // Only the broker was hit — the proxy was never called.
        expect(fetchImpl).toHaveBeenCalledTimes(1);
        expect(fetchImpl).toHaveBeenCalledWith('https://host/base/ajax.php', expect.anything());
        // The user's message is shown; no "unavailable" bubble is appended.
        expect(messagesEl.textContent).not.toContain('unavailable');
    });
});

// ---------------------------------------------------------------------------
// createChatController — no-patient handling (P2.17 global launcher)
//
// The launcher opens this panel on every page, including ones with no patient
// selected. ChatProxyController resolves the pid server-side per request and
// returns a 400 { reason: 'no_patient_in_session' } when none is bound, so the
// panel shows a specific "open a patient chart first" hint rather than the
// generic unavailable message.
// ---------------------------------------------------------------------------
describe('createChatController no-patient handling', () => {
    afterEach(() => {
        document.body.innerHTML = '';
    });

    function makeController(fetchImpl) {
        const messagesEl = document.createElement('div');
        document.body.appendChild(messagesEl);

        return {
            messagesEl: messagesEl,
            controller: createChatController({
                messagesEl: messagesEl,
                formEl: document.createElement('form'),
                inputEl: document.createElement('textarea'),
                context: { csrfToken: 'csrf' },
                brokerUrl: 'https://host/base/ajax.php',
                proxyUrl: 'https://host/base/chat-proxy.php',
                feedbackUrl: 'https://host/base/feedback-proxy.php',
                authorizeUrl: 'https://host/base/oauth-authorize.php',
                fetchImpl: fetchImpl
            })
        };
    }

    // fetchImpl that issues a token from the broker, then answers the proxy
    // with the given (status, body) pair.
    function brokerThenProxy(proxyStatus, proxyBody) {
        return jest.fn((url) => {
            if (url === 'https://host/base/ajax.php') {
                return Promise.resolve({ ok: true, json: () => Promise.resolve({ token: 'tok' }) });
            }
            return Promise.resolve({
                ok: proxyStatus >= 200 && proxyStatus < 300,
                status: proxyStatus,
                json: () => Promise.resolve(proxyBody)
            });
        });
    }

    test('a 400 no_patient_in_session proxy response shows the open-a-patient hint, not the generic error', async () => {
        const fetchImpl = brokerThenProxy(400, { error: 'No patient in session', reason: 'no_patient_in_session' });
        const { messagesEl, controller } = makeController(fetchImpl);

        await controller.sendMessage('What meds is she on?');

        expect(messagesEl.textContent).toContain('Open a patient chart first');
        expect(messagesEl.textContent).not.toContain('unavailable');
    });

    test('a 400 without the no_patient reason falls back to the generic unavailable message', async () => {
        const fetchImpl = brokerThenProxy(400, { error: 'Invalid request' });
        const { messagesEl, controller } = makeController(fetchImpl);

        await controller.sendMessage('What meds is she on?');

        expect(messagesEl.textContent).toContain('unavailable');
        expect(messagesEl.textContent).not.toContain('Open a patient chart first');
    });
});

// ---------------------------------------------------------------------------
// createChatController — conversation reset (P2.17 global launcher)
//
// The launcher's panel lives in the never-reloaded main.php shell, so after a
// patient switch it must NOT carry the prior patient's conversation id into
// the next request (the agent binds a conversation to its patient and rejects
// a mismatched pid). reset() drops the cached id and clears the transcript so
// the next send opens a fresh conversation bound to the current patient.
// ---------------------------------------------------------------------------
describe('createChatController conversation reset', () => {
    afterEach(() => {
        document.body.innerHTML = '';
    });

    function streamResp(frames) {
        return { ok: true, status: 200, body: { getReader: () => fakeReader(frames) } };
    }

    test('reset() drops the conversation id (and clears the transcript) so the next send starts fresh', async () => {
        const proxyBodies = [];
        const fetchImpl = jest.fn((url, opts) => {
            if (url.endsWith('/ajax.php')) {
                return Promise.resolve({ ok: true, json: () => Promise.resolve({ token: 'tok' }) });
            }
            proxyBodies.push(JSON.parse(opts.body));
            return Promise.resolve(streamResp([
                'event: conversation\ndata: {"conversation_id":"A"}\n\n',
                'event: answer\ndata: {"answer":"hi"}\n\n'
            ]));
        });

        const messagesEl = document.createElement('div');
        document.body.appendChild(messagesEl);
        const controller = createChatController({
            messagesEl: messagesEl,
            formEl: document.createElement('form'),
            inputEl: document.createElement('textarea'),
            context: { csrfToken: 'csrf' },
            brokerUrl: 'https://host/base/ajax.php',
            proxyUrl: 'https://host/base/chat-proxy.php',
            feedbackUrl: 'https://host/base/feedback-proxy.php',
            authorizeUrl: 'https://host/base/oauth-authorize.php',
            fetchImpl: fetchImpl
        });

        // First send has no prior id; the answer frame sets it to 'A'.
        await controller.sendMessage('q1');
        expect(proxyBodies[0].conversation_id).toBeNull();

        // Without a reset the id is carried forward (this is the cross-patient
        // leak the launcher must avoid).
        await controller.sendMessage('q2');
        expect(proxyBodies[1].conversation_id).toBe('A');

        // reset() clears both the id and the visible transcript.
        controller.reset();
        expect(messagesEl.textContent).toBe('');

        await controller.sendMessage('q3');
        expect(proxyBodies[2].conversation_id).toBeNull();
    });
});

// ---------------------------------------------------------------------------
// createChatController — self-heal after an error frame
//
// If the launcher panel is left OPEN across a patient switch, no open-time
// reset fires, and the stale conversation id + new session pid is hard-
// rejected by the agent's pid-binding check as an `error` frame. Clearing the
// conversation id in that branch lets the NEXT send start fresh and recover,
// instead of the still-open panel repeating the identical rejection forever.
// ---------------------------------------------------------------------------
describe('createChatController error self-heal', () => {
    afterEach(() => {
        document.body.innerHTML = '';
    });

    function streamResp(frames) {
        return { ok: true, status: 200, body: { getReader: () => fakeReader(frames) } };
    }

    function makeController(fetchImpl) {
        const messagesEl = document.createElement('div');
        document.body.appendChild(messagesEl);
        return {
            messagesEl: messagesEl,
            controller: createChatController({
                messagesEl: messagesEl,
                formEl: document.createElement('form'),
                inputEl: document.createElement('textarea'),
                context: { csrfToken: 'csrf' },
                brokerUrl: 'https://host/base/ajax.php',
                proxyUrl: 'https://host/base/chat-proxy.php',
                feedbackUrl: 'https://host/base/feedback-proxy.php',
                authorizeUrl: 'https://host/base/oauth-authorize.php',
                fetchImpl: fetchImpl
            })
        };
    }

    test('an error frame clears the conversation id so the next send starts a fresh conversation', async () => {
        const proxyBodies = [];
        let call = 0;
        const fetchImpl = jest.fn((url, opts) => {
            if (url.endsWith('/ajax.php')) {
                return Promise.resolve({ ok: true, json: () => Promise.resolve({ token: 'tok' }) });
            }
            proxyBodies.push(JSON.parse(opts.body));
            call += 1;
            if (call === 1) {
                // First send succeeds and establishes conversation 'A'.
                return Promise.resolve(streamResp([
                    'event: conversation\ndata: {"conversation_id":"A"}\n\n',
                    'event: answer\ndata: {"answer":"hi"}\n\n'
                ]));
            }
            if (call === 2) {
                // Second send (panel left open across a patient switch) is
                // rejected by the agent -> an error frame, no answer.
                return Promise.resolve(streamResp([
                    'event: error\ndata: {"status":409}\n\n'
                ]));
            }
            // Third send would wedge (repeat 'A') without the self-heal clear.
            return Promise.resolve(streamResp([
                'event: conversation\ndata: {"conversation_id":"B"}\n\n',
                'event: answer\ndata: {"answer":"ok"}\n\n'
            ]));
        });

        const { messagesEl, controller } = makeController(fetchImpl);

        await controller.sendMessage('q1');
        expect(proxyBodies[0].conversation_id).toBeNull();

        // The rejected send still reuses 'A' (this is the request the agent
        // rejects); the error branch then clears the id.
        await controller.sendMessage('q2');
        expect(proxyBodies[1].conversation_id).toBe('A');
        expect(messagesEl.textContent).toContain('unavailable');

        // Self-heal: the next send starts fresh (null), not wedged on 'A'.
        await controller.sendMessage('q3');
        expect(proxyBodies[2].conversation_id).toBeNull();
    });
});

// ---------------------------------------------------------------------------
// renderAboutLegend — P2.20 first-open explainer legend
//
// Reuses VERDICT_BADGES/renderVerdictBadge (the same vocabulary a real
// answer's badge is built from) so the legend copy can never diverge.
// ---------------------------------------------------------------------------
describe('renderAboutLegend', () => {
    afterEach(() => {
        document.body.innerHTML = '';
    });

    test('renders one row per verdict, each with a badge and a one-line meaning', () => {
        const list = document.createElement('ul');
        document.body.appendChild(list);

        renderAboutLegend(list);

        expect(list.children).toHaveLength(3);

        const rowText = Array.from(list.children).map((row) => row.textContent);
        expect(rowText[0]).toContain('Verified');
        expect(rowText[1]).toContain('Partially verified');
        expect(rowText[2]).toContain('Blocked');

        // Each row's badge is the exact markup renderVerdictBadge produces
        // for a real answer -- no divergent legend-only copy/classes.
        list.children[0].querySelectorAll('.copilot-verdict-badge').forEach((badge) => {
            expect(badge.className).toContain('copilot-verdict-verified');
        });
    });
});

// ---------------------------------------------------------------------------
// createChatController — P2.20 first-open "about" explainer gives way
//
// The about block (rendered by CopilotPanelController, wired via
// options.aboutEl) is visible before any message is sent. It must hide on
// the first send and stay hidden through the rest of the conversation, but
// come back after reset() (a fresh conversation is a fresh first-open).
// ---------------------------------------------------------------------------
describe('createChatController about-state give-way', () => {
    afterEach(() => {
        document.body.innerHTML = '';
    });

    function streamResp(frames) {
        return { ok: true, status: 200, body: { getReader: () => fakeReader(frames) } };
    }

    function makeController() {
        const messagesEl = document.createElement('div');
        const aboutEl = document.createElement('div');
        document.body.appendChild(aboutEl);
        document.body.appendChild(messagesEl);

        const fetchImpl = jest.fn((url) => {
            if (url.endsWith('/ajax.php')) {
                return Promise.resolve({ ok: true, json: () => Promise.resolve({ token: 'tok' }) });
            }
            return Promise.resolve(streamResp([
                'event: conversation\ndata: {"conversation_id":"A"}\n\n',
                'event: answer\ndata: {"answer":"hi"}\n\n'
            ]));
        });

        return {
            aboutEl: aboutEl,
            messagesEl: messagesEl,
            controller: createChatController({
                messagesEl: messagesEl,
                aboutEl: aboutEl,
                formEl: document.createElement('form'),
                inputEl: document.createElement('textarea'),
                context: { csrfToken: 'csrf' },
                brokerUrl: 'https://host/base/ajax.php',
                proxyUrl: 'https://host/base/chat-proxy.php',
                feedbackUrl: 'https://host/base/feedback-proxy.php',
                authorizeUrl: 'https://host/base/oauth-authorize.php',
                fetchImpl: fetchImpl
            })
        };
    }

    test('is visible before the first send, hidden after it, and stays hidden on a later send', async () => {
        const { aboutEl, controller } = makeController();

        expect(aboutEl.classList.contains('copilot-hidden')).toBe(false);

        await controller.sendMessage('q1');
        expect(aboutEl.classList.contains('copilot-hidden')).toBe(true);

        await controller.sendMessage('q2');
        expect(aboutEl.classList.contains('copilot-hidden')).toBe(true);
    });

    test('reset() brings the about block back for a fresh conversation', async () => {
        const { aboutEl, controller } = makeController();

        await controller.sendMessage('q1');
        expect(aboutEl.classList.contains('copilot-hidden')).toBe(true);

        controller.reset();
        expect(aboutEl.classList.contains('copilot-hidden')).toBe(false);
    });

    test('sendMessage does not throw when no aboutEl is wired (e.g. older markup)', async () => {
        const messagesEl = document.createElement('div');
        document.body.appendChild(messagesEl);
        const fetchImpl = jest.fn((url) => {
            if (url.endsWith('/ajax.php')) {
                return Promise.resolve({ ok: true, json: () => Promise.resolve({ token: 'tok' }) });
            }
            return Promise.resolve(streamResp([
                'event: conversation\ndata: {"conversation_id":"A"}\n\n',
                'event: answer\ndata: {"answer":"hi"}\n\n'
            ]));
        });
        const controller = createChatController({
            messagesEl: messagesEl,
            formEl: document.createElement('form'),
            inputEl: document.createElement('textarea'),
            context: { csrfToken: 'csrf' },
            brokerUrl: 'https://host/base/ajax.php',
            proxyUrl: 'https://host/base/chat-proxy.php',
            feedbackUrl: 'https://host/base/feedback-proxy.php',
            authorizeUrl: 'https://host/base/oauth-authorize.php',
            fetchImpl: fetchImpl
        });

        await expect(controller.sendMessage('q1')).resolves.not.toThrow();
        expect(() => controller.reset()).not.toThrow();
    });
});

// ---------------------------------------------------------------------------
// createThinkingIndicator — #208 staged progress indicator's DOM primitive.
//
// A spinner + status line appended to a container, with a setStage(key)
// that swaps the status text and a remove() that detaches it. Every status
// string is static copy (no interpolation of response/record data), and an
// unrecognized stage key falls back to a generic label rather than showing
// nothing or stale text -- see the "Graceful fallback" acceptance criterion
// on issue #208.
// ---------------------------------------------------------------------------
describe('createThinkingIndicator', () => {
    afterEach(() => {
        document.body.innerHTML = '';
    });

    function makeContainer() {
        const div = document.createElement('div');
        document.body.appendChild(div);
        return div;
    }

    test('appends a spinner + status element and sets each staged status text, all static/PHI-free', () => {
        const container = makeContainer();
        const indicator = createThinkingIndicator(container);

        const el = container.querySelector('.copilot-thinking');
        expect(el).not.toBeNull();

        indicator.setStage('consulting');
        expect(el.textContent).toContain('Consulting the chart');

        indicator.setStage('reasoning');
        expect(el.textContent).toContain('Reasoning locally');
        expect(el.textContent).toContain('Qwen3-4B');

        indicator.setStage('verifying');
        expect(el.textContent).toContain('Verifying claims against the record');

        // No stage text ever carries a patient name, medication, or other
        // record value -- every assertion above is a fixed substring of
        // static copy, never an interpolated value from a response.
    });

    test('falls back to a generic "Thinking" label for an unrecognized/indeterminate stage', () => {
        const container = makeContainer();
        const indicator = createThinkingIndicator(container);

        indicator.setStage('some-stage-that-does-not-exist');

        expect(container.querySelector('.copilot-thinking').textContent).toContain('Thinking');
    });

    test('remove() detaches the indicator from its container', () => {
        const container = makeContainer();
        const indicator = createThinkingIndicator(container);

        expect(container.querySelector('.copilot-thinking')).not.toBeNull();
        indicator.remove();
        expect(container.querySelector('.copilot-thinking')).toBeNull();

        // Idempotent: a second remove() (e.g. both an error path and a
        // reset() racing to clean up) must not throw.
        expect(() => indicator.remove()).not.toThrow();
    });
});

// ---------------------------------------------------------------------------
// createChatController — #208 staged progress indicator wired into sendMessage
//
// The ~18-20s local-model wait was previously silent dead-air (nothing
// rendered between the user's message and the answer). A thinking indicator
// now appears immediately on send and advances its status as the real SSE
// frames arrive (tool_call -> reasoning; verification -> verifying), then is
// removed exactly when the answer renders. Cleared the same way on the
// hadError branch and on reset().
// ---------------------------------------------------------------------------
describe('createChatController staged thinking indicator (#208)', () => {
    afterEach(() => {
        document.body.innerHTML = '';
    });

    function streamResp(reader) {
        return { ok: true, status: 200, body: { getReader: () => reader } };
    }

    function makeController(fetchImpl) {
        const messagesEl = document.createElement('div');
        document.body.appendChild(messagesEl);
        return {
            messagesEl: messagesEl,
            controller: createChatController({
                messagesEl: messagesEl,
                formEl: document.createElement('form'),
                inputEl: document.createElement('textarea'),
                context: { csrfToken: 'csrf' },
                brokerUrl: 'https://host/base/ajax.php',
                proxyUrl: 'https://host/base/chat-proxy.php',
                feedbackUrl: 'https://host/base/feedback-proxy.php',
                authorizeUrl: 'https://host/base/oauth-authorize.php',
                fetchImpl: fetchImpl
            })
        };
    }

    function brokerThenReader(reader) {
        return jest.fn((url) => {
            if (url.endsWith('/ajax.php')) {
                return Promise.resolve({ ok: true, json: () => Promise.resolve({ token: 'tok' }) });
            }
            return Promise.resolve(streamResp(reader));
        });
    }

    test('shows the indicator immediately on send, before any network response arrives', () => {
        // A fetchImpl that never resolves -- proves the indicator appears
        // synchronously on send, not once a frame is first parsed (i.e. no
        // more silent dead-air while waiting on the token broker either).
        const fetchImpl = jest.fn(() => new Promise(() => {}));
        const { messagesEl, controller } = makeController(fetchImpl);

        controller.sendMessage('What meds is she on?');

        const indicator = messagesEl.querySelector('.copilot-thinking');
        expect(indicator).not.toBeNull();
        expect(indicator.textContent).toContain('Consulting the chart');
    });

    test('advances consulting -> reasoning -> verifying as tool_call/verification frames arrive, then clears on the answer', async () => {
        const reader = pausableReader();
        const fetchImpl = brokerThenReader(reader);
        const { messagesEl, controller } = makeController(fetchImpl);

        const sendPromise = controller.sendMessage('What meds is she on?');
        await flushMicrotasks();

        let indicator = messagesEl.querySelector('.copilot-thinking');
        expect(indicator).not.toBeNull();
        expect(indicator.textContent).toContain('Consulting the chart');

        reader.push('event: conversation\ndata: {"conversation_id":"c1"}\n\n');
        reader.push('event: tool_call\ndata: {"tool":"get_medications","args":{},"error":null}\n\n');
        await flushMicrotasks();
        expect(indicator.textContent).toContain('Reasoning locally');

        reader.push('event: verification\ndata: {"verdict":"verified","segments":[],"warnings":{}}\n\n');
        await flushMicrotasks();
        expect(indicator.textContent).toContain('Verifying claims against the record');

        reader.push('event: answer\ndata: {"answer":"She takes lisinopril."}\n\n');
        reader.finish();
        await sendPromise;

        expect(messagesEl.querySelector('.copilot-thinking')).toBeNull();
        expect(messagesEl.textContent).toContain('She takes lisinopril.');
    });

    test('falls back to the generic "thinking" state when no tool_call frame ever arrives to signal a specific stage', async () => {
        // The planner can answer with zero tool calls -- no frame ever
        // signals the consulting->reasoning boundary in that case. The
        // indicator still shows a status (never blank/stale) and still
        // clears cleanly once the answer renders.
        const reader = pausableReader();
        const fetchImpl = brokerThenReader(reader);
        const { messagesEl, controller } = makeController(fetchImpl);

        const sendPromise = controller.sendMessage('What is the diagnosis?');
        await flushMicrotasks();

        const indicator = messagesEl.querySelector('.copilot-thinking');
        expect(indicator.textContent.trim().length).toBeGreaterThan(0);

        reader.push('event: conversation\ndata: {"conversation_id":"c1"}\n\n');
        reader.push('event: answer\ndata: {"answer":"Hypertension."}\n\n');
        reader.finish();
        await sendPromise;

        expect(messagesEl.querySelector('.copilot-thinking')).toBeNull();
    });

    test('clears the indicator on an error frame (hadError branch)', async () => {
        const reader = pausableReader();
        const fetchImpl = brokerThenReader(reader);
        const { messagesEl, controller } = makeController(fetchImpl);

        const sendPromise = controller.sendMessage('What meds is she on?');
        await flushMicrotasks();
        expect(messagesEl.querySelector('.copilot-thinking')).not.toBeNull();

        reader.push('event: error\ndata: {"status":409}\n\n');
        reader.finish();
        await sendPromise;

        expect(messagesEl.querySelector('.copilot-thinking')).toBeNull();
        expect(messagesEl.textContent).toContain('unavailable');
    });

    test('clears an in-flight indicator on reset()', () => {
        const reader = pausableReader();
        const fetchImpl = brokerThenReader(reader);
        const { messagesEl, controller } = makeController(fetchImpl);

        controller.sendMessage('What meds is she on?');
        expect(messagesEl.querySelector('.copilot-thinking')).not.toBeNull();

        controller.reset();

        expect(messagesEl.querySelector('.copilot-thinking')).toBeNull();
        expect(messagesEl.textContent).toBe('');
    });
});

// ---------------------------------------------------------------------------
// createChatController — #208 reasoning fallback timer (direct coverage)
//
// The dev stack batches the whole SSE stream at the end of the wait (see the
// createThinkingIndicator docstring), so the 1.5s fallback timer -- not frame
// arrival -- is what actually delivers the "Reasoning locally..." stage
// through the long gap. It is the load-bearing mechanism and gets direct
// coverage here via jest.useFakeTimers(): one test proves the timer fires and
// flips the stage on its own; one proves stopThinking()'s clearTimeout
// prevents a late mutation after an exit (removing that clearTimeout must turn
// the second test red).
// ---------------------------------------------------------------------------
describe('createChatController reasoning fallback timer (#208)', () => {
    // THINKING_REASONING_FALLBACK_MS in copilot-chat.js -- the module keeps it
    // private, so the timing is mirrored here (kept in sync with the source).
    const FALLBACK_MS = 1500;

    afterEach(() => {
        jest.useRealTimers();
        document.body.innerHTML = '';
    });

    function makeController(fetchImpl) {
        const messagesEl = document.createElement('div');
        document.body.appendChild(messagesEl);
        return {
            messagesEl: messagesEl,
            controller: createChatController({
                messagesEl: messagesEl,
                formEl: document.createElement('form'),
                inputEl: document.createElement('textarea'),
                context: { csrfToken: 'csrf' },
                brokerUrl: 'https://host/base/ajax.php',
                proxyUrl: 'https://host/base/chat-proxy.php',
                feedbackUrl: 'https://host/base/feedback-proxy.php',
                authorizeUrl: 'https://host/base/oauth-authorize.php',
                fetchImpl: fetchImpl
            })
        };
    }

    test('the fallback timer alone advances consulting -> reasoning (no tool_call frame needed)', () => {
        jest.useFakeTimers();
        // The request never resolves -- so the ONLY thing that can advance the
        // stage is the timer, proving it is wired and fires.
        const fetchImpl = jest.fn(() => new Promise(() => {}));
        const { messagesEl, controller } = makeController(fetchImpl);

        controller.sendMessage('What meds is she on?');

        const indicator = messagesEl.querySelector('.copilot-thinking');
        expect(indicator.textContent).toContain('Consulting the chart');

        jest.advanceTimersByTime(FALLBACK_MS);

        expect(indicator.textContent).toContain('Reasoning locally');
    });

    test('reset() mid-flight clears the timer so it cannot mutate the stage afterwards', () => {
        jest.useFakeTimers();
        const fetchImpl = jest.fn(() => new Promise(() => {}));
        const { messagesEl, controller } = makeController(fetchImpl);

        controller.sendMessage('What meds is she on?');

        // Hold a reference to the status node BEFORE reset detaches it -- the
        // fallback timer's callback closes over this same element, so if the
        // timer is not cleared it will still flip this node's text even after
        // it has been removed from the DOM.
        const status = messagesEl.querySelector('.copilot-thinking-status');
        expect(status.textContent).toContain('Consulting the chart');

        // Exit mid-flight, BEFORE the fallback fires.
        controller.reset();
        expect(messagesEl.querySelector('.copilot-thinking')).toBeNull();

        jest.advanceTimersByTime(FALLBACK_MS);

        // With stopThinking()'s clearTimeout in place the callback never runs,
        // so the detached node's text is unchanged. Remove that clearTimeout
        // and the timer fires setStage('reasoning') on this node -> red.
        expect(status.textContent).not.toContain('Reasoning locally');
        expect(status.textContent).toContain('Consulting the chart');
    });
});

// ---------------------------------------------------------------------------
// createReasoningZone (#213) -- the live token-by-token "thinking" surface.
//
// A separate, clearly-labeled zone (NEVER the answer bubble) that the
// reasoning_delta SSE frame's text streams into progressively. This is the
// owner's non-negotiable UX rule: unverified/provisional model text must
// never occupy the authoritative answer slot -- it gets its own zone.
// ---------------------------------------------------------------------------
describe('createReasoningZone', () => {
    afterEach(() => {
        document.body.innerHTML = '';
    });

    function makeContainer() {
        const div = document.createElement('div');
        document.body.appendChild(div);
        return div;
    }

    test('appends a labeled zone and streams appended text progressively', () => {
        const container = makeContainer();
        const zone = createReasoningZone(container);

        const el = container.querySelector('.copilot-reasoning');
        expect(el).not.toBeNull();
        expect(el.textContent).toContain('Reasoning locally');
        expect(el.textContent).toContain('Qwen3-4B');

        zone.append('Let me check ');
        zone.append('the medication list.');

        expect(el.textContent).toContain('Let me check the medication list.');
    });

    test('renders a <script> payload as inert text, never as a DOM element (XSS safety)', () => {
        const container = makeContainer();
        const zone = createReasoningZone(container);
        const payload = '<script>window.__pwned = true;</script>';

        zone.append(payload);

        expect(container.querySelector('script')).toBeNull();
        expect(window.__pwned).toBeUndefined();
        expect(container.textContent).toContain(payload);
    });

    test('remove() detaches the zone from its container and is idempotent', () => {
        const container = makeContainer();
        const zone = createReasoningZone(container);

        expect(container.querySelector('.copilot-reasoning')).not.toBeNull();
        zone.remove();
        expect(container.querySelector('.copilot-reasoning')).toBeNull();

        expect(() => zone.remove()).not.toThrow();
    });

    test('finalize() marks the zone as no longer actively streaming', () => {
        const container = makeContainer();
        const zone = createReasoningZone(container);
        const el = container.querySelector('.copilot-reasoning');

        expect(el.className).toContain('copilot-reasoning-active');
        zone.finalize();
        expect(el.className).not.toContain('copilot-reasoning-active');
    });
});

// ---------------------------------------------------------------------------
// createChatController — reasoning_delta streaming (#213)
//
// reasoning_delta frames append into a dedicated thinking zone, distinct
// from the answer bubble. The #208 generic spinner hands off to this live
// zone the instant reasoning tokens start arriving (no redundant "Reasoning
// locally..." spinner shown alongside a real typing stream). The answer
// bubble renders ONLY from the `answer` frame -- this is the hard safety
// regression guard: draft reasoning text must never land in the answer slot.
// ---------------------------------------------------------------------------
describe('createChatController reasoning_delta streaming (#213)', () => {
    // #219 paces the reveal instead of appending immediately, so every test
    // below that expects streamed text to be visible needs fake timers
    // advanced past the drain (a few seconds is always enough headroom for
    // these short strings, well under real per-test timeouts).
    beforeEach(() => {
        jest.useFakeTimers();
    });

    afterEach(() => {
        jest.useRealTimers();
        document.body.innerHTML = '';
    });

    function streamResp(reader) {
        return { ok: true, status: 200, body: { getReader: () => reader } };
    }

    function makeController(fetchImpl) {
        const messagesEl = document.createElement('div');
        document.body.appendChild(messagesEl);
        return {
            messagesEl: messagesEl,
            controller: createChatController({
                messagesEl: messagesEl,
                formEl: document.createElement('form'),
                inputEl: document.createElement('textarea'),
                context: { csrfToken: 'csrf' },
                brokerUrl: 'https://host/base/ajax.php',
                proxyUrl: 'https://host/base/chat-proxy.php',
                feedbackUrl: 'https://host/base/feedback-proxy.php',
                authorizeUrl: 'https://host/base/oauth-authorize.php',
                fetchImpl: fetchImpl
            })
        };
    }

    function brokerThenReader(reader) {
        return jest.fn((url) => {
            if (url.endsWith('/ajax.php')) {
                return Promise.resolve({ ok: true, json: () => Promise.resolve({ token: 'tok' }) });
            }
            return Promise.resolve(streamResp(reader));
        });
    }

    // Replaces the old real-timer flushMicrotasks() helper under fake
    // timers: 0ms still lets the pending reader/consumeSSEStream promise
    // chain settle without advancing the pacing timer.
    function flushAsync(ms) {
        return jest.advanceTimersByTimeAsync(ms || 0);
    }

    test('reasoning_delta frames append into the thinking zone token-by-token, never into the answer bubble', async () => {
        const reader = pausableReader();
        const fetchImpl = brokerThenReader(reader);
        const { messagesEl, controller } = makeController(fetchImpl);

        const sendPromise = controller.sendMessage('What meds is she on?');
        await flushAsync();

        reader.push('event: conversation\ndata: {"conversation_id":"c1"}\n\n');
        reader.push('event: reasoning_delta\ndata: {"text":"Let me check "}\n\n');
        // Give the paced drain enough time to reveal this short delta.
        await flushAsync(2000);

        let zone = messagesEl.querySelector('.copilot-reasoning');
        expect(zone).not.toBeNull();
        expect(zone.textContent).toContain('Let me check');

        reader.push('event: reasoning_delta\ndata: {"text":"the medication list..."}\n\n');
        await flushAsync(2000);
        expect(zone.textContent).toContain('Let me check the medication list...');

        // The answer has not arrived yet -- no assistant bubble carrying the
        // final text should exist, and the reasoning text must never be
        // treated as the answer.
        expect(messagesEl.querySelector('.copilot-chat-message-assistant')).toBeNull();

        reader.push('event: answer\ndata: {"answer":"She takes lisinopril."}\n\n');
        reader.finish();
        await flushAsync(2000);
        await sendPromise;

        const assistantBubble = messagesEl.querySelector('.copilot-chat-message-assistant');
        expect(assistantBubble).not.toBeNull();
        expect(assistantBubble.textContent).toBe('She takes lisinopril.');
        // The verified answer bubble never contains the draft reasoning text.
        expect(assistantBubble.textContent).not.toContain('Let me check');
        expect(assistantBubble.textContent).not.toContain('medication list');
    });

    test('hands off from the #208 spinner to the reasoning zone -- no redundant spinner once reasoning starts', async () => {
        const reader = pausableReader();
        const fetchImpl = brokerThenReader(reader);
        const { messagesEl, controller } = makeController(fetchImpl);

        const sendPromise = controller.sendMessage('What meds is she on?');
        await flushAsync();

        expect(messagesEl.querySelector('.copilot-thinking')).not.toBeNull();

        reader.push('event: reasoning_delta\ndata: {"text":"Let me check."}\n\n');
        await flushAsync();

        // The generic spinner is gone -- replaced by the live zone, not
        // shown redundantly alongside it. The zone itself (and the handoff)
        // happens synchronously on frame arrival -- only its content reveal
        // is paced.
        expect(messagesEl.querySelector('.copilot-thinking')).toBeNull();
        expect(messagesEl.querySelector('.copilot-reasoning')).not.toBeNull();

        reader.push('event: answer\ndata: {"answer":"ok"}\n\n');
        reader.finish();
        await flushAsync(2000);
        await sendPromise;
    });

    test('the reasoning zone stays visible above the answer once the answer renders (minimal post-answer treatment)', async () => {
        const reader = pausableReader();
        const fetchImpl = brokerThenReader(reader);
        const { messagesEl, controller } = makeController(fetchImpl);

        const sendPromise = controller.sendMessage('What meds is she on?');
        await flushAsync();

        reader.push('event: reasoning_delta\ndata: {"text":"Reasoning text."}\n\n');
        await flushAsync();

        reader.push('event: answer\ndata: {"answer":"Final answer."}\n\n');
        reader.finish();
        await flushAsync(2000);
        await sendPromise;

        const zone = messagesEl.querySelector('.copilot-reasoning');
        expect(zone).not.toBeNull();
        expect(zone.textContent).toContain('Reasoning text.');
        expect(zone.className).not.toContain('copilot-reasoning-active');
    });

    test('clears an in-flight reasoning zone on an error frame', async () => {
        const reader = pausableReader();
        const fetchImpl = brokerThenReader(reader);
        const { messagesEl, controller } = makeController(fetchImpl);

        const sendPromise = controller.sendMessage('What meds is she on?');
        await flushAsync();

        reader.push('event: reasoning_delta\ndata: {"text":"Reasoning text."}\n\n');
        await flushAsync();
        expect(messagesEl.querySelector('.copilot-reasoning')).not.toBeNull();

        reader.push('event: error\ndata: {"status":409}\n\n');
        reader.finish();
        await flushAsync();
        await sendPromise;

        expect(messagesEl.querySelector('.copilot-reasoning')).toBeNull();
        expect(messagesEl.textContent).toContain('unavailable');
    });

    test('reset() mid-flight clears an in-flight reasoning zone', async () => {
        const reader = pausableReader();
        const fetchImpl = brokerThenReader(reader);
        const { messagesEl, controller } = makeController(fetchImpl);

        controller.sendMessage('What meds is she on?');
        await flushAsync();

        reader.push('event: reasoning_delta\ndata: {"text":"Reasoning text."}\n\n');
        await flushAsync();
        expect(messagesEl.querySelector('.copilot-reasoning')).not.toBeNull();

        controller.reset();

        expect(messagesEl.querySelector('.copilot-reasoning')).toBeNull();
        expect(messagesEl.textContent).toBe('');
    });

    test('no reasoning_delta frames -- no reasoning zone is ever created (planner answered with zero reasoning streamed)', async () => {
        const reader = pausableReader();
        const fetchImpl = brokerThenReader(reader);
        const { messagesEl, controller } = makeController(fetchImpl);

        const sendPromise = controller.sendMessage('What meds is she on?');
        await flushAsync();

        reader.push('event: answer\ndata: {"answer":"ok"}\n\n');
        reader.finish();
        await flushAsync();
        await sendPromise;

        expect(messagesEl.querySelector('.copilot-reasoning')).toBeNull();
    });

    test('a stream ending with reasoning but no truthy answer and no error clears the zone (no orphaned blinking cursor)', async () => {
        // Edge case: the answer frame carries an empty string (falsy), or the
        // stream completes cleanly after reasoning with no answer/error at
        // all. The zone must not be left in the DOM still "actively
        // streaming" (permanent cursor) with activeReasoningZone dangling.
        const reader = pausableReader();
        const fetchImpl = brokerThenReader(reader);
        const { messagesEl, controller } = makeController(fetchImpl);

        const sendPromise = controller.sendMessage('What meds is she on?');
        await flushAsync();

        reader.push('event: reasoning_delta\ndata: {"text":"thinking..."}\n\n');
        await flushAsync();
        expect(messagesEl.querySelector('.copilot-reasoning')).not.toBeNull();

        // Empty-string answer (falsy) then clean end -- neither the answer
        // branch nor the error branch fires.
        reader.push('event: answer\ndata: {"answer":""}\n\n');
        reader.finish();
        await flushAsync();
        await sendPromise;

        expect(messagesEl.querySelector('.copilot-reasoning')).toBeNull();

        // And a subsequent reset() must not throw on a dangling handle.
        expect(() => controller.reset()).not.toThrow();
    });
});

// ---------------------------------------------------------------------------
// #219 paced reveal -- the reasoning "thinking zone" reveals its already-
// received reasoning_delta text at a steady, readable rate instead of
// appending it immediately, so the typewriter stays smoothly visible whether
// Ollama delivers tokens spread out, in a sub-20ms burst, or as a single EOF
// delta. Design: buffer incoming text and drain it via a fixed-rate timer
// (REVEAL_TICK_MS / REVEAL_CHARS_PER_TICK below, mirroring the module's
// private constants -- ~40 chars/sec, within the 30-45 cps target).
// REVEAL_CAP_MS bounds how long a pending drain may block the verified
// `answer` frame from rendering: createChatController awaits
// revealer.waitForDrain() (which force-flushes and resolves at the cap)
// before appending the answer bubble, so the answer is never blocked by
// more than the cap. `prefers-reduced-motion` skips pacing entirely.
// ---------------------------------------------------------------------------
describe('createReasoningRevealer (#219 paced reveal)', () => {
    // Mirrors copilot-chat.js's private REVEAL_* constants -- kept in sync
    // with the source, same convention as FALLBACK_MS above (#208).
    const REVEAL_TICK_MS = 50;
    const REVEAL_CHARS_PER_TICK = 2;

    afterEach(() => {
        jest.useRealTimers();
        document.body.innerHTML = '';
        delete window.matchMedia;
    });

    function makeZone() {
        const container = document.createElement('div');
        document.body.appendChild(container);
        const zone = createReasoningZone(container);
        const textEl = container.querySelector('.copilot-reasoning-text');
        return { zone: zone, textEl: textEl };
    }

    // (a) A single large delta (the "one EOF burst" scenario) is still
    // revealed incrementally as timers advance -- not all at once.
    test('reveals a single large delta incrementally as timers advance, not all at once', () => {
        jest.useFakeTimers();
        const { zone, textEl } = makeZone();
        const revealer = createReasoningRevealer(zone);
        const text = 'a'.repeat(60);

        revealer.push(text);

        // Nothing revealed synchronously -- only once timers advance.
        expect(textEl.textContent.length).toBe(0);

        jest.advanceTimersByTime(REVEAL_TICK_MS);
        const afterOneTick = textEl.textContent.length;
        expect(afterOneTick).toBeGreaterThan(0);
        expect(afterOneTick).toBeLessThan(text.length);
        expect(afterOneTick).toBe(REVEAL_CHARS_PER_TICK);

        jest.advanceTimersByTime(REVEAL_TICK_MS * 4);
        const afterFiveTicks = textEl.textContent.length;
        expect(afterFiveTicks).toBeGreaterThan(afterOneTick);
        expect(afterFiveTicks).toBeLessThan(text.length);

        jest.advanceTimersByTime(5000);
        expect(textEl.textContent).toBe(text);
    });

    // (b) prefers-reduced-motion: skip pacing entirely -> instant reveal,
    // no pacing timer scheduled.
    test('prefers-reduced-motion reveals instantly and schedules no pacing timer', () => {
        jest.useFakeTimers();
        window.matchMedia = jest.fn().mockReturnValue({ matches: true });
        // Spy specifically on setInterval -- the zone's own auto-scroll
        // uses requestAnimationFrame (also a fake-timer-visible timer), so
        // asserting on jest.getTimerCount() would be a false positive
        // unrelated to whether the *pacing* timer was scheduled.
        const setIntervalSpy = jest.spyOn(window, 'setInterval');

        const { zone, textEl } = makeZone();
        const revealer = createReasoningRevealer(zone);

        revealer.push('Reasoning text.');

        expect(textEl.textContent).toBe('Reasoning text.');
        // The reveal never went through the paced drain path -- no interval
        // was scheduled at all.
        expect(setIntervalSpy).not.toHaveBeenCalled();
    });

    // (c) stop() clears the pacing timer; a late tick must not mutate the
    // zone. Mutation-tested: this assertion fails if revealer.stop() drops
    // its clearInterval call.
    test('stop() clears the pacing timer so a late tick cannot mutate the zone (mutation-tested)', () => {
        jest.useFakeTimers();
        const { zone, textEl } = makeZone();
        const revealer = createReasoningRevealer(zone);

        revealer.push('a'.repeat(60));
        jest.advanceTimersByTime(REVEAL_TICK_MS);
        const revealedBeforeStop = textEl.textContent.length;
        expect(revealedBeforeStop).toBeGreaterThan(0);
        expect(revealedBeforeStop).toBeLessThan(60);

        revealer.stop();
        jest.advanceTimersByTime(10000);

        // If stop() stopped clearing its interval, this would now equal the
        // full 60 chars -- proving the assertion actually exercises the
        // cleanup rather than passing vacuously.
        expect(textEl.textContent.length).toBe(revealedBeforeStop);
    });
});

// ---------------------------------------------------------------------------
// createChatController — #219 paced reveal wired into sendMessage: cleanup
// on reset()/error, and the answer-frame cap.
// ---------------------------------------------------------------------------
describe('createChatController reasoning pacing (#219)', () => {
    const REVEAL_TICK_MS = 50;
    const REVEAL_CAP_MS = 1500;

    afterEach(() => {
        jest.useRealTimers();
        document.body.innerHTML = '';
    });

    function streamResp(reader) {
        return { ok: true, status: 200, body: { getReader: () => reader } };
    }

    function makeController(fetchImpl) {
        const messagesEl = document.createElement('div');
        document.body.appendChild(messagesEl);
        return {
            messagesEl: messagesEl,
            controller: createChatController({
                messagesEl: messagesEl,
                formEl: document.createElement('form'),
                inputEl: document.createElement('textarea'),
                context: { csrfToken: 'csrf' },
                brokerUrl: 'https://host/base/ajax.php',
                proxyUrl: 'https://host/base/chat-proxy.php',
                feedbackUrl: 'https://host/base/feedback-proxy.php',
                authorizeUrl: 'https://host/base/oauth-authorize.php',
                fetchImpl: fetchImpl
            })
        };
    }

    function brokerThenReader(reader) {
        return jest.fn((url) => {
            if (url.endsWith('/ajax.php')) {
                return Promise.resolve({ ok: true, json: () => Promise.resolve({ token: 'tok' }) });
            }
            return Promise.resolve(streamResp(reader));
        });
    }

    // (c) cleared on reset(), mutation-tested at the controller level: hold a
    // reference to the revealed text node BEFORE reset() (reset detaches the
    // zone from messagesEl, so mutations on the detached node would
    // otherwise be invisible to a messagesEl-only assertion).
    test('reset() mid-reveal clears the pacing timer -- advancing timers afterward causes no further zone mutation', async () => {
        jest.useFakeTimers();
        const reader = pausableReader();
        const fetchImpl = brokerThenReader(reader);
        const { messagesEl, controller } = makeController(fetchImpl);

        controller.sendMessage('What meds is she on?');
        await jest.advanceTimersByTimeAsync(0);

        reader.push('event: reasoning_delta\ndata: {"text":"' + 'a'.repeat(60) + '"}\n\n');
        await jest.advanceTimersByTimeAsync(REVEAL_TICK_MS);

        const textEl = messagesEl.querySelector('.copilot-reasoning-text');
        expect(textEl).not.toBeNull();
        const revealedBeforeReset = textEl.textContent.length;
        expect(revealedBeforeReset).toBeGreaterThan(0);
        expect(revealedBeforeReset).toBeLessThan(60);

        controller.reset();
        expect(messagesEl.querySelector('.copilot-reasoning')).toBeNull();

        await jest.advanceTimersByTimeAsync(10000);

        // Mutation guard: without the pacing timer's clear, this node (held
        // from before reset(), now detached) would keep growing.
        expect(textEl.textContent.length).toBe(revealedBeforeReset);
    });

    // (c) cleared on the error path, mutation-tested the same way.
    test('an error frame clears the pacing timer -- advancing timers afterward causes no further zone mutation', async () => {
        jest.useFakeTimers();
        const reader = pausableReader();
        const fetchImpl = brokerThenReader(reader);
        const { messagesEl, controller } = makeController(fetchImpl);

        const sendPromise = controller.sendMessage('What meds is she on?');
        await jest.advanceTimersByTimeAsync(0);

        reader.push('event: reasoning_delta\ndata: {"text":"' + 'a'.repeat(60) + '"}\n\n');
        await jest.advanceTimersByTimeAsync(REVEAL_TICK_MS);

        const textEl = messagesEl.querySelector('.copilot-reasoning-text');
        expect(textEl).not.toBeNull();
        const revealedBeforeError = textEl.textContent.length;
        expect(revealedBeforeError).toBeGreaterThan(0);
        expect(revealedBeforeError).toBeLessThan(60);

        reader.push('event: error\ndata: {"status":409}\n\n');
        reader.finish();
        await jest.advanceTimersByTimeAsync(0);
        await sendPromise;

        expect(messagesEl.querySelector('.copilot-reasoning')).toBeNull();

        await jest.advanceTimersByTimeAsync(10000);

        expect(textEl.textContent.length).toBe(revealedBeforeError);
    });

    // (d) the answer frame must not be blocked on the full drain -- it
    // renders within the documented REVEAL_CAP_MS cap.
    test('when the answer frame arrives mid-reveal, the answer renders within the documented cap, not blocked on the full drain', async () => {
        jest.useFakeTimers();
        const reader = pausableReader();
        const fetchImpl = brokerThenReader(reader);
        const { messagesEl, controller } = makeController(fetchImpl);

        const sendPromise = controller.sendMessage('What meds is she on?');
        await jest.advanceTimersByTimeAsync(0);

        // At ~40 chars/sec this buffer alone would take ~10s to fully drain
        // -- far past the 1.5s cap, so this proves the cap (not the drain)
        // is what gates the answer render.
        reader.push('event: reasoning_delta\ndata: {"text":"' + 'a'.repeat(400) + '"}\n\n');
        reader.push('event: answer\ndata: {"answer":"Final answer."}\n\n');
        reader.finish();

        await jest.advanceTimersByTimeAsync(REVEAL_CAP_MS);
        await sendPromise;

        const assistantBubble = messagesEl.querySelector('.copilot-chat-message-assistant');
        expect(assistantBubble).not.toBeNull();
        expect(assistantBubble.textContent).toBe('Final answer.');
    });

    // No reasoning ever streamed -> no revealer created -> the answer is
    // never gated on anything reveal-related.
    test('an answer with no prior reasoning_delta renders immediately, with no pacing wait', async () => {
        jest.useFakeTimers();
        const reader = pausableReader();
        const fetchImpl = brokerThenReader(reader);
        const { messagesEl, controller } = makeController(fetchImpl);

        const sendPromise = controller.sendMessage('What is the diagnosis?');
        await jest.advanceTimersByTimeAsync(0);

        reader.push('event: answer\ndata: {"answer":"Hypertension."}\n\n');
        reader.finish();
        await jest.advanceTimersByTimeAsync(0);
        await sendPromise;

        expect(messagesEl.querySelector('.copilot-chat-message-assistant').textContent).toBe('Hypertension.');
    });
});
