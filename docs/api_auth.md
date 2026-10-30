# Authentication Service API

The authentication service exposes these endpoints:

- POST /auth/login: exchange credentials for an access token and a refresh token.
- POST /auth/refresh: exchange a valid refresh token for a new access token. Refresh tokens expire after 7 days.
- GET /auth/validate: check whether an access token is valid. Returns 401 if the token is expired.
- POST /auth/logout: revoke the current refresh token.

Access tokens expire after 15 minutes. The /auth/refresh and /auth/validate endpoints
were omitted from the v1 spec by mistake and are being added in the v1.1 docs update.
