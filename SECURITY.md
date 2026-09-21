# Security

Report vulnerabilities through GitHub's private vulnerability reporting for this repository. Do
not include production keys, model inputs, customer data, or public proof-of-concept exploits.

Supported production deployments use the current `main` branch, the pinned model runtime published
with `devanshbatham/nyx`, a loopback/private SGLang backend, and an HTTPS ingress in front of the
gateway. Rotate bearer keys after suspected exposure. Never ship gateway keys in browser bundles or
mobile applications.

The gateway is a single-trust-domain service. It does not provide tenant-specific keys, per-tenant
quotas, audit storage, TLS, or a secrets manager; production infrastructure must supply those when
required.
