<?php

/**
 * Isolated tests for the PKCE (RFC 7636) verifier/challenge pair used by the
 * Clinical Co-Pilot OAuth consent flow (#124 Phase 2b).
 *
 * The security property under test: the S256 code_challenge is the URL-safe,
 * unpadded base64 of SHA-256(code_verifier) -- exactly what OpenEMR's
 * CustomAuthCodeGrant recomputes at token time. If this drifts, the token
 * exchange silently fails. A published RFC 7636 test vector locks the maths.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Tests\Isolated\Modules\ClinicalCopilot;

use OpenEMR\Core\ModulesClassLoader;
use OpenEMR\Modules\ClinicalCopilot\Auth\PkcePair;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

class PkcePairTest extends TestCase
{
    protected function setUp(): void
    {
        $projectDir = dirname(__DIR__, 5);
        $classLoader = new ModulesClassLoader($projectDir);
        $classLoader->registerNamespaceIfNotExists(
            'OpenEMR\\Modules\\ClinicalCopilot\\',
            $projectDir . '/interface/modules/custom_modules/oe-module-clinical-copilot/src'
        );
    }

    #[Test]
    public function challengeMatchesRfc7636TestVector(): void
    {
        // RFC 7636 Appendix B: verifier -> S256 challenge.
        $verifier = 'dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk';
        $expectedChallenge = 'E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM';

        $this->assertSame($expectedChallenge, PkcePair::challengeFor($verifier));
    }

    #[Test]
    public function generatedChallengeIsS256OfItsOwnVerifier(): void
    {
        $pair = PkcePair::generate();

        $this->assertSame(PkcePair::challengeFor($pair->verifier), $pair->challenge);
    }

    #[Test]
    public function generatedVerifierIsUrlSafeAndWithinRfcLength(): void
    {
        $pair = PkcePair::generate();

        // RFC 7636 section 4.1: 43-128 chars from the unreserved set.
        $this->assertGreaterThanOrEqual(43, strlen($pair->verifier));
        $this->assertLessThanOrEqual(128, strlen($pair->verifier));
        $this->assertMatchesRegularExpression('/^[A-Za-z0-9\-._~]+$/', $pair->verifier);
    }

    #[Test]
    public function challengeIsUnpaddedUrlSafeBase64(): void
    {
        $pair = PkcePair::generate();

        // No '+', '/' or '=' padding -- must be transmissible in a query string as-is.
        $this->assertMatchesRegularExpression('/^[A-Za-z0-9\-_]+$/', $pair->challenge);
        $this->assertStringNotContainsString('=', $pair->challenge);
    }

    #[Test]
    public function eachGenerateProducesADistinctVerifier(): void
    {
        $this->assertNotSame(PkcePair::generate()->verifier, PkcePair::generate()->verifier);
    }
}
