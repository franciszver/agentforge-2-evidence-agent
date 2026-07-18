/**
 * @jest-environment jsdom
 */

/**
 * Tests for the P3.7 citation-overlay ("click-to-source") logic in
 * interface/modules/custom_modules/oe-module-clinical-copilot/public/assets/js/copilot-chat.js
 *
 * Capability decision (see services/copilot-agent/app/documents.py's module
 * docstring for the full write-up): a qwen2.5vl:7b bbox-grounding probe
 * against the committed lab fixture was accurate on a clean render but
 * drifted onto the wrong table cell once scan-realistic noise/rotation were
 * applied -- too unreliable to draw a pixel box from on a real scanned
 * document. This UI renders the HONEST page-level fallback instead: a
 * "View source page" link that opens the real source PDF at the cited page
 * (the standard `#page=N` browser/PDF.js anchor), plus the citation's own
 * literal quote text shown alongside it -- never a fabricated box.
 *
 * Covers: page-number parsing from `page_or_section`, the source-document
 * URL builder, and the document-citation chip (click reveals the quote/page
 * + the view-source link only for source types that actually have a stored
 * PDF -- guideline_chunk citations have no per-patient ingested document to
 * open). XSS-inertness matches the existing SourceRef citation chip's
 * contract (textContent only; a malicious source_id cannot inject a
 * javascript: URL into the link's href).
 *
 * Run with: npm test -- tests/js/clinical-copilot-citation-overlay.test.js
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(
    path.resolve(
        __dirname,
        '../../interface/modules/custom_modules/oe-module-clinical-copilot/public/assets/js/copilot-chat.js'
    ),
    'utf8'
);
new Function('window', 'document', src)(global.window, global.document);

const {
    parseCitedPageNumber,
    buildSourceDocUrl,
    buildDocumentCitationChip,
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
// parseCitedPageNumber -- pure parser
// ---------------------------------------------------------------------------
describe('parseCitedPageNumber', () => {
    test('parses "page N" (the Citation.page_or_section format)', () => {
        expect(parseCitedPageNumber('page 2')).toBe(2);
        expect(parseCitedPageNumber('page 12')).toBe(12);
    });

    test('is case-insensitive and tolerates surrounding whitespace', () => {
        expect(parseCitedPageNumber('Page 3')).toBe(3);
        expect(parseCitedPageNumber('  page 4  ')).toBe(4);
    });

    test('returns null for a non-page section label (e.g. a guideline chunk)', () => {
        expect(parseCitedPageNumber('Section: Medications')).toBeNull();
    });

    test('returns null for null/undefined/non-string input', () => {
        expect(parseCitedPageNumber(null)).toBeNull();
        expect(parseCitedPageNumber(undefined)).toBeNull();
        expect(parseCitedPageNumber(42)).toBeNull();
    });
});

// ---------------------------------------------------------------------------
// buildSourceDocUrl -- pure URL builder
// ---------------------------------------------------------------------------
describe('buildSourceDocUrl', () => {
    test('builds a query-string URL against the given base', () => {
        expect(buildSourceDocUrl('/source-doc-proxy.php', 'abc123')).toBe(
            '/source-doc-proxy.php?source_id=abc123'
        );
    });

    test('percent-encodes the source_id so it cannot smuggle extra query params or a scheme change', () => {
        const url = buildSourceDocUrl('/source-doc-proxy.php', 'javascript:alert(1)//&evil=1');
        expect(url.startsWith('/source-doc-proxy.php?source_id=')).toBe(true);
        expect(url).not.toContain('javascript:alert');
        expect(url).not.toContain('&evil=1');
    });
});

// ---------------------------------------------------------------------------
// buildDocumentCitationChip -- click-to-source chip
// ---------------------------------------------------------------------------
describe('buildDocumentCitationChip', () => {
    const labCitation = {
        source_type: 'lab_pdf',
        source_id: 'a1b2c3d4',
        page_or_section: 'page 2',
        field_or_chunk_id: 'Glucose',
        quote_or_value: 'Glucose: 105 mg/dL'
    };

    test('the chip is collapsed until clicked, then reveals the quote/page', () => {
        const { chip, record } = buildDocumentCitationChip(labCitation, '/source-doc-proxy.php');

        expect(chip.textContent).toBe('Glucose');
        expect(record.classList.contains('copilot-hidden')).toBe(true);
        expect(chip.getAttribute('aria-expanded')).toBe('false');

        chip.dispatchEvent(new window.Event('click'));

        expect(record.classList.contains('copilot-hidden')).toBe(false);
        expect(chip.getAttribute('aria-expanded')).toBe('true');
        expect(record.textContent).toContain('Glucose: 105 mg/dL');
        expect(record.textContent).toContain('page 2');
    });

    test('for a viewable source (lab_pdf/intake_form), the record carries a real link to the source page', () => {
        const { record } = buildDocumentCitationChip(labCitation, '/source-doc-proxy.php');

        const link = record.querySelector('.copilot-citation-source-link');
        expect(link).not.toBeNull();
        expect(link.tagName).toBe('A');
        expect(link.getAttribute('href')).toBe('/source-doc-proxy.php?source_id=a1b2c3d4#page=2');
        expect(link.getAttribute('target')).toBe('_blank');
        expect(link.getAttribute('rel')).toContain('noopener');
    });

    test('a guideline_chunk citation gets no view-source link -- there is no per-patient PDF backing it', () => {
        const { record } = buildDocumentCitationChip(
            {
                source_type: 'guideline_chunk',
                source_id: 'diabetes-guideline#a1c-targets',
                page_or_section: 'Section: A1c targets',
                field_or_chunk_id: 'diabetes-guideline#a1c-targets',
                quote_or_value: 'Target A1c <7% for most adults.'
            },
            '/source-doc-proxy.php'
        );

        expect(record.querySelector('.copilot-citation-source-link')).toBeNull();
        expect(record.textContent).toContain('Target A1c <7% for most adults.');
    });

    test('never draws or claims a bounding box -- only a page-level link/quote (no-fabrication contract)', () => {
        const { chip, record } = buildDocumentCitationChip(labCitation, '/source-doc-proxy.php');
        chip.dispatchEvent(new window.Event('click'));

        expect(record.querySelector('canvas')).toBeNull();
        expect(record.querySelector('[data-bbox]')).toBeNull();
    });

    test('with no sourceDocBaseUrl provided, renders the record without a view-source link (no crash)', () => {
        const { record } = buildDocumentCitationChip(labCitation, undefined);
        expect(record.querySelector('.copilot-citation-source-link')).toBeNull();
    });

    // -----------------------------------------------------------------------
    // XSS safety -- same contract as the existing SourceRef citation chip.
    // -----------------------------------------------------------------------
    test('a <script>/javascript: payload in citation fields renders inert, and cannot hijack the link href', () => {
        const container = makeContainer();
        const { chip, record } = buildDocumentCitationChip(
            {
                source_type: 'lab_pdf',
                source_id: 'javascript:window.__pwned=true//',
                page_or_section: 'page 1',
                field_or_chunk_id: '<img src=x onerror="window.__pwned = true">',
                quote_or_value: '<script>window.__pwned = true;</script>'
            },
            '/source-doc-proxy.php'
        );
        container.appendChild(chip);
        container.appendChild(record);
        chip.dispatchEvent(new window.Event('click'));

        expect(container.querySelector('script')).toBeNull();
        expect(container.querySelector('img')).toBeNull();
        expect(window.__pwned).toBeUndefined();

        const link = record.querySelector('.copilot-citation-source-link');
        // Either no link (fine) or, if rendered, its href must not start with
        // a javascript: scheme -- encodeURIComponent must have neutralized it.
        if (link) {
            expect(link.getAttribute('href').startsWith('javascript:')).toBe(false);
        }
    });
});

// ---------------------------------------------------------------------------
// renderVerification integration -- document_citations chips render
// alongside (not instead of) SourceRef citation chips on a mixed claim.
// ---------------------------------------------------------------------------
describe('renderVerification with document_citations', () => {
    test('renders a document-citation chip for a claim carrying document_citations', () => {
        const container = makeContainer();
        renderVerification(
            container,
            {
                verdict: 'verified',
                segments: [
                    {
                        type: 'claim',
                        text: 'Glucose is 105 mg/dL.',
                        citations: [],
                        document_citations: [
                            {
                                source_type: 'lab_pdf',
                                source_id: 'a1b2c3d4',
                                page_or_section: 'page 2',
                                field_or_chunk_id: 'Glucose',
                                quote_or_value: 'Glucose: 105 mg/dL'
                            }
                        ]
                    }
                ],
                warnings: { allergy_conflicts: [], blocking_interactions: [], warning_interactions: [] }
            },
            '/source-doc-proxy.php'
        );

        const chip = container.querySelector('.copilot-document-citation-chip');
        expect(chip).not.toBeNull();
        expect(chip.textContent).toBe('Glucose');

        chip.dispatchEvent(new window.Event('click'));
        const link = container.querySelector('.copilot-citation-source-link');
        expect(link.getAttribute('href')).toBe('/source-doc-proxy.php?source_id=a1b2c3d4#page=2');
    });

    test('a claim with both SourceRef and document citations renders both chip kinds', () => {
        const container = makeContainer();
        renderVerification(
            container,
            {
                verdict: 'verified',
                segments: [
                    {
                        type: 'claim',
                        text: 'On Lisinopril; glucose 105 mg/dL.',
                        citations: [{ tool_call_id: 'call-1', record_id: 'med-42', field: 'dose', value: '10 mg' }],
                        document_citations: [
                            {
                                source_type: 'lab_pdf',
                                source_id: 'a1b2c3d4',
                                page_or_section: 'page 2',
                                field_or_chunk_id: 'Glucose',
                                quote_or_value: 'Glucose: 105 mg/dL'
                            }
                        ]
                    }
                ],
                warnings: { allergy_conflicts: [], blocking_interactions: [], warning_interactions: [] }
            },
            '/source-doc-proxy.php'
        );

        expect(container.querySelectorAll('.copilot-citation-chip').length).toBe(2);
        expect(container.querySelector('.copilot-document-citation-chip')).not.toBeNull();
    });

    test('renderVerification without a sourceDocBaseUrl arg (existing 2-arg callers) still works', () => {
        const container = makeContainer();
        expect(() =>
            renderVerification(container, {
                verdict: 'verified',
                segments: [
                    {
                        type: 'claim',
                        text: 'Glucose is 105 mg/dL.',
                        citations: [],
                        document_citations: [
                            {
                                source_type: 'lab_pdf',
                                source_id: 'a1b2c3d4',
                                page_or_section: 'page 2',
                                field_or_chunk_id: 'Glucose',
                                quote_or_value: 'Glucose: 105 mg/dL'
                            }
                        ]
                    }
                ],
                warnings: { allergy_conflicts: [], blocking_interactions: [], warning_interactions: [] }
            })
        ).not.toThrow();

        expect(container.querySelector('.copilot-document-citation-chip')).not.toBeNull();
        expect(container.querySelector('.copilot-citation-source-link')).toBeNull();
    });
});
