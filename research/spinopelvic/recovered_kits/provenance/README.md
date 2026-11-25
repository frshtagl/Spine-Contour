# Recovered artifact provenance

See the parent `ARTIFACT_MANIFEST.json` for:

- Original ZIP byte sizes and SHA-256 fingerprints
- Recovered checkpoint sizes and SHA-256 fingerprints
- Existing upstream Git LFS pointer metadata
- Model architecture summaries
- Exclusions and validation status

The manifest distinguishes the checkpoint files inside the user-supplied ZIPs from the fork's existing LFS pointer objects. The U-Net matches exactly; HRNet and S1 do not.

The actual LFS binaries were not downloaded or strict-loaded during this audit. Pointer metadata is not proof that the corresponding binary object can currently be fetched or that it loads successfully.
