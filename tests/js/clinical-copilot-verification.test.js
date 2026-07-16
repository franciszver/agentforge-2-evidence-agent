/**
 * @jest-environment jsdom
 */

/**
 * Tests for the P3.8 verification-layer UI in
 * interface/modules/custom_modules/oe-module-clinical-copilot/public/assets/js/copilot-chat.js
 *
 * Covers the pure render logic that surfaces the verification frame's payload:
 * verdict->badge mapping (all three verdicts), claim->citation chips (tap
 * reveals the underlying record), warnings->warning banner, the empty/degenerate
 * (pending) verification result, and XSS-inertness -- all verification text is
 * rendered via textContent/createTextNode, never innerHTML, because it derives
 * from the record/LLM (indirect-injection threat model, same as P2.14's
 * assistant text).
 *
 * Run with: npm test -- tests/js/clinical-copilot-verification.test.js
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
    verdictBadgeInfo,
    renderVerdictBadge,
    renderWarningBanner,
    renderVerification
} = global.window.CopilotChat;

function makeContainer() {
    const div = document.createElement('div');
    document.body.appendChild(div);
    return div;
}

afterEach(() => {
    document.body.innerHTML = '';
    delete window.__pwned;
});

// ---------------------------------------------------------------------------
// verdictBadgeInfo / renderVerdictBadge — verdict -> badge mapping
// ---------------------------------------------------------------------------
describe('verdict badge', () => {
    test('maps all three verdicts to distinct label + class', () => {
        const verified = verdictBadgeInfo('verified');
        const partial = verdictBadgeInfo('partially_verified');
        const blocked = verdictBadgeInfo('blocked');

        expect(verified.label).toBe('Verified');
        expect(partial.label).toBe('Partially verified');
        expect(blocked.label).toBe('Blocked');

        const classes = [verified.className, partial.className, blocked.className];
        expect(new Set(classes).size).toBe(3); // each verdict has its own class
    });

    test('returns null for an unknown / missing verdict (no crash)', () => {
        expect(verdictBadgeInfo('bogus')).toBeNull();
        expect(verdictBadgeInfo(null)).toBeNull();
        expect(verdictBadgeInfo(undefined)).toBeNull();
    });

    test('renderVerdictBadge is not color-only: carries a text label', () => {
        const badge = renderVerdictBadge('verified');
        expect(badge.className).toContain('copilot-verdict-verified');
        // The accessible signal is the text, not just the color.
        expect(badge.textContent).toContain('Verified');
    });

    test('renderVerdictBadge returns null for an unknown verdict', () => {
        expect(renderVerdictBadge('nope')).toBeNull();
    });
});

// ---------------------------------------------------------------------------
// renderWarningBanner — warnings -> banner
// ---------------------------------------------------------------------------
describe('warning banner', () => {
    test('renders an allergy conflict prominently', () => {
        const banner = renderWarningBanner({
            allergy_conflicts: [{ medication_name: 'Ibuprofen', allergy_substance: 'NSAID' }],
            blocking_interactions: [],
            warning_interactions: []
        });

        expect(banner).not.toBeNull();
        expect(banner.getAttribute('role')).toBe('alert');
        expect(banner.textContent).toContain('Ibuprofen');
        expect(banner.textContent).toContain('NSAID');
    });

    test('renders a blocking interaction', () => {
        const banner = renderWarningBanner({
            allergy_conflicts: [],
            blocking_interactions: [
                { drug_a: 'warfarin', drug_b: 'aspirin', severity: 'major', description: 'Bleeding risk.' }
            ],
            warning_interactions: []
        });

        expect(banner.textContent).toContain('warfarin');
        expect(banner.textContent).toContain('aspirin');
        expect(banner.textContent).toContain('Bleeding risk.');
    });

    test('returns null when there are no allergy conflicts or blocking interactions', () => {
        expect(
            renderWarningBanner({ allergy_conflicts: [], blocking_interactions: [], warning_interactions: [] })
        ).toBeNull();
        expect(renderWarningBanner(undefined)).toBeNull();
    });
});

// ---------------------------------------------------------------------------
// renderVerification — full block: badge + banner + claims/chips + notices
// ---------------------------------------------------------------------------
describe('renderVerification', () => {
    const verifiedData = {
        verdict: 'verified',
        segments: [
            {
                type: 'claim',
                text: 'She takes lisinopril 10 mg daily.',
                citations: [
                    { tool_call_id: 'call-1', record_id: 'med-42', field: 'dose', value: '10 mg' }
                ]
            }
        ],
        warnings: { allergy_conflicts: [], blocking_interactions: [], warning_interactions: [] }
    };

    test('renders the verdict badge for the response', () => {
        const container = makeContainer();
        renderVerification(container, verifiedData);

        const badge = container.querySelector('.copilot-verdict-badge');
        expect(badge).not.toBeNull();
        expect(badge.textContent).toContain('Verified');
    });

    test('renders a citation chip per claim citation; the record is hidden until the chip is tapped', () => {
        const container = makeContainer();
        renderVerification(container, verifiedData);

        const chip = container.querySelector('.copilot-citation-chip');
        const record = container.querySelector('.copilot-citation-record');
        expect(chip).not.toBeNull();
        expect(record).not.toBeNull();

        // Hidden until tapped (no hover dependence — an explicit tap toggles it).
        expect(record.classList.contains('copilot-hidden')).toBe(true);
        expect(chip.getAttribute('aria-expanded')).toBe('false');

        chip.dispatchEvent(new window.Event('click'));

        expect(record.classList.contains('copilot-hidden')).toBe(false);
        expect(chip.getAttribute('aria-expanded')).toBe('true');
        // The revealed record surfaces the underlying source_ref detail.
        expect(record.textContent).toContain('10 mg');
        expect(record.textContent).toContain('med-42');
    });

    test('renders a notice segment for a stripped claim', () => {
        const container = makeContainer();
        renderVerification(container, {
            verdict: 'blocked',
            segments: [{ type: 'notice', text: 'Not found in record.' }],
            warnings: { allergy_conflicts: [], blocking_interactions: [], warning_interactions: [] }
        });

        const notice = container.querySelector('.copilot-notice');
        expect(notice).not.toBeNull();
        expect(notice.textContent).toBe('Not found in record.');
    });

    test('renders the warning banner when the response carries a conflict', () => {
        const container = makeContainer();
        renderVerification(container, {
            verdict: 'blocked',
            segments: [],
            warnings: {
                allergy_conflicts: [{ medication_name: 'Ibuprofen', allergy_substance: 'NSAID' }],
                blocking_interactions: [],
                warning_interactions: []
            }
        });

        expect(container.querySelector('.copilot-warning-banner')).not.toBeNull();
        expect(container.querySelector('.copilot-verdict-blocked')).not.toBeNull();
    });

    test('renders nothing for the pending/degenerate result (verdict null) without crashing', () => {
        const container = makeContainer();
        const result = renderVerification(container, {
            verdict: null,
            segments: [],
            warnings: { allergy_conflicts: [], blocking_interactions: [], warning_interactions: [] }
        });

        expect(result).toBeNull();
        expect(container.children).toHaveLength(0);
    });

    test('does not crash on a null/empty payload', () => {
        const container = makeContainer();
        expect(() => renderVerification(container, null)).not.toThrow();
        expect(() => renderVerification(container, {})).not.toThrow();
        expect(container.children).toHaveLength(0);
    });
});

// ---------------------------------------------------------------------------
// XSS safety — every verification text field rendered inert
// ---------------------------------------------------------------------------
describe('XSS safety of verification rendering', () => {
    test('a <script> payload in claim text / citation value renders as inert text', () => {
        const container = makeContainer();
        renderVerification(container, {
            verdict: 'verified',
            segments: [
                {
                    type: 'claim',
                    text: '<script>window.__pwned = true;</script>',
                    citations: [
                        {
                            tool_call_id: 'call-1',
                            record_id: '<img src=x onerror="window.__pwned = true">',
                            field: 'dose',
                            value: '<script>window.__pwned = true;</script>'
                        }
                    ]
                }
            ],
            warnings: { allergy_conflicts: [], blocking_interactions: [], warning_interactions: [] }
        });

        // Reveal the record so its (would-be) payload is in the DOM too.
        container.querySelector('.copilot-citation-chip').dispatchEvent(new window.Event('click'));

        expect(container.querySelector('script')).toBeNull();
        expect(container.querySelector('img')).toBeNull();
        expect(window.__pwned).toBeUndefined();
        expect(container.querySelector('.copilot-claim-text').textContent).toBe(
            '<script>window.__pwned = true;</script>'
        );
    });

    test('a <script> payload in a warning banner renders as inert text', () => {
        const banner = renderWarningBanner({
            allergy_conflicts: [
                { medication_name: '<script>window.__pwned = true;</script>', allergy_substance: 'NSAID' }
            ],
            blocking_interactions: [],
            warning_interactions: []
        });
        document.body.appendChild(banner);

        expect(document.querySelector('script')).toBeNull();
        expect(window.__pwned).toBeUndefined();
        expect(banner.textContent).toContain('<script>window.__pwned = true;</script>');
    });
});
