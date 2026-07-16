<?php

/**
 * Production authorization_code -> token exchanger.
 *
 * POSTs a form-encoded token request to the OpenEMR token endpoint using the
 * confidential client's credentials in the body (client_secret_post, matching
 * how Phase 1 registered the prod client) plus the server-stored PKCE
 * code_verifier. Mirrors the copilot-agent's password-grant call shape in
 * app/openemr_auth.py, adapted to the authorization_code grant.
 *
 * Errors are surfaced as OAuthExchangeException with a generic, secret-free
 * message; the caller logs it and shows the browser nothing.
 *
 * @package   OpenEMR\Modules\ClinicalCopilot
 * @link      https://www.open-emr.org
 * @author    Francisco de Guzman <franciszver@outlook.com>
 * @copyright Copyright (c) 2026 Francisco de Guzman
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

declare(strict_types=1);

namespace OpenEMR\Modules\ClinicalCopilot\Auth;

use GuzzleHttp\Client;

final class GuzzleAuthorizationCodeExchanger implements AuthorizationCodeExchanger
{
    private const TIMEOUT_SECONDS = 15;

    public function __construct(
        private readonly OAuthConsentConfig $config,
        private readonly bool $verifySsl = true,
        private readonly Client $client = new Client(),
    ) {
    }

    public function exchange(string $code, string $codeVerifier): OAuthTokenResponse
    {
        // Boundary: EVERY failure of the exchange -- Guzzle transport errors, a
        // non-200/malformed response, or any other throwable (incl. \Error) --
        // is funneled into OAuthExchangeException so the controller's single
        // catch handles them uniformly as a generic, log-safe fail. Generic
        // messages only; the original is chained (never embedded) so creds/URLs
        // in an underlying message cannot leak.
        try {
            $response = $this->client->post($this->config->tokenUrl, [
                'form_params' => [
                    'grant_type' => 'authorization_code',
                    'code' => $code,
                    'redirect_uri' => $this->config->redirectUri,
                    'client_id' => $this->config->clientId,
                    'client_secret' => $this->config->clientSecret,
                    'code_verifier' => $codeVerifier,
                ],
                'timeout' => self::TIMEOUT_SECONDS,
                'verify' => $this->verifySsl,
                'http_errors' => false,
            ]);

            if ($response->getStatusCode() !== 200) {
                throw new OAuthExchangeException('Token endpoint returned status ' . $response->getStatusCode());
            }

            $decoded = json_decode((string) $response->getBody(), true);
            if (!is_array($decoded)) {
                throw new OAuthExchangeException('Token endpoint returned a non-JSON body');
            }

            $refreshToken = $decoded['refresh_token'] ?? null;
            if (!is_string($refreshToken) || $refreshToken === '') {
                throw new OAuthExchangeException('Token endpoint response is missing a refresh_token');
            }

            $accessTokenRaw = $decoded['access_token'] ?? null;
            $accessToken = is_string($accessTokenRaw) && $accessTokenRaw !== '' ? $accessTokenRaw : null;

            return new OAuthTokenResponse($refreshToken, $accessToken, $this->expiryFrom($decoded));
        } catch (OAuthExchangeException $e) {
            throw $e;
        } catch (\Throwable $e) {
            throw new OAuthExchangeException('Token exchange failed', 0, $e);
        }
    }

    /**
     * @param array<array-key, mixed> $decoded
     */
    private function expiryFrom(array $decoded): ?\DateTimeImmutable
    {
        $expiresIn = $decoded['expires_in'] ?? null;
        if (!is_int($expiresIn) && !(is_string($expiresIn) && ctype_digit($expiresIn))) {
            return null;
        }

        return (new \DateTimeImmutable())->add(new \DateInterval('PT' . (int) $expiresIn . 'S'));
    }
}
