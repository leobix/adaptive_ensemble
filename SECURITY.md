# Security policy

Do not open a public issue containing credentials, Gurobi license material, private dataset links,
or sensitive operational data. Use GitHub's private security-advisory feature for a repository
security issue.

This research repository accepts security reports for the current default branch. Historical
experiment scripts are preserved for reproduction and may use older dependencies; reports should
identify whether the issue affects the active Julia publication pipeline or only historical code.

Never commit WLS access IDs, WLS secrets, license IDs, `.env` files, private keys, or `gurobi.lic`.
The CI audit rejects common credential patterns, but automated scanning is not a substitute for
manual review.
