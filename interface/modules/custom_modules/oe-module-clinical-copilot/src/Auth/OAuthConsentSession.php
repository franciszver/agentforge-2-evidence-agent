<?php

/**
 * Session key + CSRF-subject constants shared by the two halves of the OAuth
 * consent flow (the authorize redirect and the callback).
 *
 * Single source of truth so the value written when the authorize URL is built
 * is the exact value the callback reads back:
 *  - STATE_SUBJECT namespaces the CsrfUtils-derived `state` to this flow;
 *  - CODE_VERIFIER_KEY is where the per-request PKCE verifier lives server-side.
 *
 * @package   OpenEMR\Modules\ClinicalCopilot
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Auth;

final class OAuthConsentSession
{
    /** CsrfUtils subject that binds the OAuth `state` parameter to the session. */
    public const STATE_SUBJECT = 'clinical_copilot_oauth2_consent';

    /** Session key holding the per-request PKCE code_verifier (never sent to the client). */
    public const CODE_VERIFIER_KEY = 'clinical_copilot_oauth_code_verifier';

    private function __construct()
    {
    }
}
