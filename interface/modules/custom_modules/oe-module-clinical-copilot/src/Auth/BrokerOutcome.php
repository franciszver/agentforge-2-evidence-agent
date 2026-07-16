<?php

/**
 * Outcome of an agent-token broker decision (#124 Phase 3).
 *
 * A closed set of the three ways the broker can resolve: hand the panel a
 * bearer, tell it consent is required (redirect into the authorize flow), or
 * fail safe on a server misconfiguration.
 *
 * @package   OpenEMR\Modules\ClinicalCopilot
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Auth;

enum BrokerOutcome
{
    /** A bearer token is available and should be handed to the panel. */
    case Token;

    /** No usable token; the panel must send the user through the authorize flow. */
    case ConsentRequired;

    /** Fail-safe: the broker refused (e.g. encryption precondition unmet). */
    case Error;
}
