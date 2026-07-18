<?php

/**
 * Flushing Output Stream Test for Clinical Co-Pilot Module
 *
 * Exercises the PSR-7 sink used by the P211 CurlMultiHandler chat relay
 * (ChatProxyController::streamFromAgent): a stream whose write() echoes each
 * chunk to the PHP output buffer and flushes immediately, instead of
 * accumulating bytes for a later read(). This is what makes the relay
 * genuinely incremental -- Guzzle's CurlFactory calls write() once per
 * CURLOPT_WRITEFUNCTION invocation, i.e. once per chunk libcurl delivers, so
 * each write() must reach the browser before the next chunk arrives rather
 * than waiting for the whole response.
 *
 * No database or session is needed, so this runs isolated. write() emitting
 * immediately is captured via ob_start()/ob_get_contents() -- the echo lands
 * in the buffer per-write, so incremental output is directly observable;
 * flush() is a harmless no-op under CLI/PHPUnit.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <ciscodg@gmail.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Tests\Isolated\Modules\ClinicalCopilot;

use OpenEMR\Core\ModulesClassLoader;
use OpenEMR\Modules\ClinicalCopilot\Http\FlushingOutputStream;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

class FlushingOutputStreamTest extends TestCase
{
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
    public function writeReturnsTheByteLengthWritten(): void
    {
        $stream = new FlushingOutputStream();

        ob_start();
        try {
            $this->assertSame(5, $stream->write('hello'));
        } finally {
            ob_end_clean();
        }
    }

    #[Test]
    public function writeEmitsEachChunkImmediatelyRatherThanBuffering(): void
    {
        $stream = new FlushingOutputStream();

        ob_start();
        try {
            $stream->write('event: conversation');
            $firstWriteMessage = 'first write must land before the second one is issued';
            $this->assertSame('event: conversation', ob_get_contents(), $firstWriteMessage);

            $stream->write("\n\ndata: {}\n\n");
            $secondWriteMessage = 'second write appends -- neither was withheld';
            $this->assertSame("event: conversation\n\ndata: {}\n\n", ob_get_contents(), $secondWriteMessage);
        } finally {
            ob_end_clean();
        }
    }

    #[Test]
    public function isWritableAndNotReadable(): void
    {
        $stream = new FlushingOutputStream();

        $this->assertTrue($stream->isWritable());
        $this->assertFalse($stream->isReadable());
        $this->assertFalse($stream->isSeekable());
    }

    #[Test]
    public function readIsRejectedBecauseTheStreamIsWriteOnly(): void
    {
        $stream = new FlushingOutputStream();

        $this->expectException(\RuntimeException::class);
        $stream->read(8192);
    }

    #[Test]
    public function getContentsIsRejectedBecauseTheStreamIsWriteOnly(): void
    {
        $stream = new FlushingOutputStream();

        $this->expectException(\RuntimeException::class);
        $stream->getContents();
    }

    #[Test]
    public function seekIsRejectedBecauseTheStreamIsNotSeekable(): void
    {
        $stream = new FlushingOutputStream();

        $this->expectException(\RuntimeException::class);
        $stream->seek(0);
    }

    #[Test]
    public function tellTracksTheTotalBytesWrittenSoFar(): void
    {
        $stream = new FlushingOutputStream();

        ob_start();
        try {
            $stream->write('abc');
            $stream->write('de');
            $this->assertSame(5, $stream->tell());
        } finally {
            ob_end_clean();
        }
    }
}
