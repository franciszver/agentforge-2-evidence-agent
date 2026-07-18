<?php

/**
 * Flushing Output Stream (Clinical Co-Pilot)
 *
 * A write-only PSR-7 stream used as Guzzle's `sink` for the P211
 * CurlMultiHandler chat relay (ChatProxyController::streamFromAgent).
 * CurlFactory's CURLOPT_WRITEFUNCTION calls write() once per chunk libcurl
 * delivers from the upstream agent -- this stream echoes and flushes each
 * chunk to the browser immediately instead of accumulating it for a later
 * read(), which is what makes the relay genuinely incremental rather than
 * batched (see PHP's `http://` stream-wrapper buffering problem this
 * replaces, documented in ChatProxyController).
 *
 * Write-only by design: nothing in the relay ever reads the body back, so
 * read()/getContents()/seek() are rejected rather than silently no-op'd.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <ciscodg@gmail.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Http;

use Psr\Http\Message\StreamInterface;

final class FlushingOutputStream implements StreamInterface
{
    private int $bytesWritten = 0;

    private bool $closed = false;

    public function write(string $string): int
    {
        if ($this->closed) {
            throw new \RuntimeException('Cannot write to a closed stream');
        }

        echo $string;
        flush();

        $length = strlen($string);
        $this->bytesWritten += $length;

        return $length;
    }

    public function isWritable(): bool
    {
        return !$this->closed;
    }

    public function isReadable(): bool
    {
        return false;
    }

    public function isSeekable(): bool
    {
        return false;
    }

    public function read(int $length): string
    {
        throw new \RuntimeException('Stream is write-only');
    }

    public function getContents(): string
    {
        throw new \RuntimeException('Stream is write-only');
    }

    public function seek(int $offset, int $whence = SEEK_SET): void
    {
        throw new \RuntimeException('Stream is not seekable');
    }

    public function rewind(): void
    {
        throw new \RuntimeException('Stream is not seekable');
    }

    public function tell(): int
    {
        return $this->bytesWritten;
    }

    public function eof(): bool
    {
        return $this->closed;
    }

    public function getSize(): ?int
    {
        return null;
    }

    public function close(): void
    {
        $this->closed = true;
    }

    public function detach()
    {
        $this->closed = true;

        return null;
    }

    public function getMetadata(?string $key = null): mixed
    {
        return $key === null ? [] : null;
    }

    public function __toString(): string
    {
        return '';
    }
}
