# Cloud Build Pipeline

The MVP pipeline is validation-only.

## Pull Request

- Install dependencies.
- Run unit tests.
- Run prior-authorization red-team eval.
- Produce eval report.

## Main Branch

- Same as pull request.
- Publish evidence artifacts if the evidence bucket is configured.

## Production

Out of scope for MVP. Production requires manual approval, product-name verification for the managed GCP agent runtime, least-privilege identity review, provider privacy review, and safety-screening validation.
