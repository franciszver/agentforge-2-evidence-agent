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
    appendMessage
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
