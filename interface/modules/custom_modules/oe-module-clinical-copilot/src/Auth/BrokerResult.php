<?php

/**
 * Result of an agent-token broker decision (#124 Phase 3).
 *
 * Discriminated by BrokerOutcome. The bearer is present only on the Token
 * outcome; the ConsentRequired and Error outcomes never carry one, so the
 * controller cannot accidentally leak or emit a token on a non-token path.
 *
 * @package   OpenEMR\Modules\ClinicalCopilot
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Auth;

final readonly class BrokerResult
{
    private function __construct(
        public BrokerOutcome $outcome,
        public ?string $token,
    ) {
    }

    public static function token(string $token): self
    {
        return new self(BrokerOutcome::Token, $token);
    }

    public static function consentRequired(): self
    {
        return new self(BrokerOutcome::ConsentRequired, null);
    }

    public static function error(): self
    {
        return new self(BrokerOutcome::Error, null);
    }
}
