<?php

/**
 * Seam for building the SMART `launch` token from the session patient id.
 *
 * Behind an interface so AuthorizeRedirectController can be unit-tested without
 * CryptoGen or the database (the real SMARTLaunchToken::serialize() encrypts,
 * and the patient UUID is a DB lookup). The production implementation is
 * SmartLaunchTokenFactory.
 *
 * @package   OpenEMR\Modules\ClinicalCopilot
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Auth;

interface LaunchTokenFactory
{
    /**
     * Build the serialized, URL-safe SMART launch token for the given patient id.
     */
    public function create(int $pid): string;
}
