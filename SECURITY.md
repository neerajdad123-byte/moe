# Security policy

## Supported versions

Only the [latest release](https://github.com/neerajdad123-byte/moe/releases/latest) gets security fixes. Update
before reporting: run the installer line from the README again, or download the newest zip.

## Reporting a vulnerability

Please do not open a public issue. Report it privately through GitHub:
[Report a vulnerability](https://github.com/neerajdad123-byte/moe/security/advisories/new).

Include what an attacker can do, the steps or a small script to reproduce it, and the moex version. You will get an
answer within a few days. Once a fix is released, the report can be published with credit to you, if you want that.

## What counts

The parts of moex most exposed to others:

- `moex serve`: the HTTP API (by default only on this PC, `127.0.0.1`), its Host / Origin checks against websites
  calling it, and its handling of malformed or oversized requests;
- `moex download` and `moex install`: files written to disk and changes to the user PATH;
- the install script (`install.ps1`).

`moex serve --host 0.0.0.0` has no API key by design: anyone who can reach the port can use the model. That is
documented, not a vulnerability; report ways to get more than model access through it.
