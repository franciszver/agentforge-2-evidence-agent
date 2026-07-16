<?php

/**
 * Raised when the authorization_code -> token exchange fails.
 *
 * The message is for server-side logs only; the callback controller catches
 * this and returns a generic, detail-free response to the browser.
 *
 * @package   OpenEMR\Modules\ClinicalCopilot
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Auth;

final class OAuthExchangeException extends \RuntimeException
{
}
