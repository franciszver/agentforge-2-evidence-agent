<?php

/**
 * Standalone Page Context Resolution Test for Clinical Co-Pilot Module
 *
 * Exercises the pure "does this session have a patient in context?" decision
 * used by the P2.15 standalone /copilot route (CopilotStandaloneController):
 * the page renders the chat panel when a patient is bound, or an honest
 * "open a chart first" state when not. No database or session is needed for
 * this logic (it takes the already-resolved pid as its only input), so this
 * runs isolated -- the controller's session-reading constructor and full-page
 * render (which do need a live session) are exercised by the paired Panther
 * scenario instead, same discipline as ChatProxyRequestTest for P2.14.
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
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestWith;
use PHPUnit\Framework\TestCase;

class CopilotStandaloneControllerTest extends TestCase
{
    private const CLASS_NAME = 'OpenEMR\\Modules\\ClinicalCopilot\\Controller\\CopilotStandaloneController';

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
    #[TestWith([1, true])]
    #[TestWith([42, true])]
    #[TestWith([0, false])]
    #[TestWith([-1, false])]
    public function testHasPatientContext(int $pid, bool $expected): void
    {
        $hasPatientContext = [self::CLASS_NAME, 'hasPatientContext'];

        $this->assertSame($expected, $hasPatientContext($pid));
    }
}
