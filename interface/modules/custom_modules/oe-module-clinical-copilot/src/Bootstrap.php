<?php

/**
 * Clinical Co-Pilot Module Bootstrap Class
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2025 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot;

use Symfony\Component\EventDispatcher\EventDispatcherInterface;

class Bootstrap
{
    const MODULE_INSTALLATION_PATH = "/interface/modules/custom_modules/oe-module-clinical-copilot";
    const MODULE_NAME = "oe-module-clinical-copilot";

    public function __construct(
        /**
         * @var EventDispatcherInterface The object responsible for sending and subscribing to events.
         * Public so the scaffold carries the wiring for the first real event
         * subscription (P2.12) without a write-only private property.
         */
        public readonly EventDispatcherInterface $eventDispatcher
    ) {
    }

    /**
     * Subscribe to events.
     * For now, this is a no-op since we're just scaffolding the module.
     *
     * @return void
     */
    public function subscribeToEvents(): void
    {
        // Placeholder for future event subscriptions
    }
}
