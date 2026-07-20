<?php

/**
 * Production LaunchTokenFactory: serializes a SMART launch token carrying the
 * session patient's UUID.
 *
 * Mirrors OpenEMR's own SmartLaunchController: resolve the patient UUID from the
 * pid, convert the raw UUID bytes to their string form, and serialize a
 * SMARTLaunchToken (which encrypts + base64-encodes it for URL transport).
 *
 * @package   OpenEMR\Modules\ClinicalCopilot
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Auth;

use OpenEMR\Common\Uuid\UuidRegistry;
use OpenEMR\FHIR\SMART\SMARTLaunchToken;
use OpenEMR\Services\PatientService;

final class SmartLaunchTokenFactory implements LaunchTokenFactory
{
    public function __construct(
        private readonly PatientService $patientService = new PatientService(),
    ) {
    }

    public function create(int $pid): string
    {
        $uuidBytes = $this->patientService->getUuid((string) $pid);
        if (!is_string($uuidBytes) || $uuidBytes === '') {
            throw new \RuntimeException('No patient UUID available for the SMART launch token');
        }

        $token = new SMARTLaunchToken(UuidRegistry::uuidToString($uuidBytes));
        $serialized = $token->serialize();
        if (!is_string($serialized) || $serialized === '') {
            throw new \RuntimeException('SMART launch token serialization failed');
        }

        return $serialized;
    }
}
