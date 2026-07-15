<?php

/**
 * Chat Proxy Request Exception
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Chat;

/**
 * Raised by ChatProxyRequest::fromArray() when the decoded JSON body does
 * not carry a valid chat request. The message is a fixed, non-sensitive
 * label -- never echoes request content -- so it is always safe to log or
 * to key an HTTP status decision off of.
 */
final class ChatProxyRequestException extends \RuntimeException
{
}
