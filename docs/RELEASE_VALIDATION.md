# Release validation — v1.1.0

Checked on 30 October 2026 in a separate release staging directory, using Python 3.12.4 and the dependencies in `requirements-tested.txt`.

| Check | Observed result |
|---|---|
| `python code/tests/test_covfourier.py` | 19/19 passed |
| `python code/tests/test_design.py` | 5/5 passed |
| Original manuscript numerical-claim audit | 167/167 passed |
| New-applications numerical-claim audit | 71/71 passed |
| Manuscript citation consistency | 37 listed and 37 cited; no missing or uncited keys; first-citation order matched |
| Portable NOAA downloader, first two entries | Downloaded and verified against the source SHA-256 manifest |
| Data-root discovery | Resolves to this repository without an author-machine path |
| Portable scaling runner | All nine commands checked with `--dry-run`; heavy batch not rerun |

The numerical-claim audits ran against a private copy of the completed manuscript and stored results, including the coordinate cache required by the new residual-energy check. They check agreement of quoted values with the recorded data, not independent proof of the mathematics or external validity of the climate model. The unpublished manuscript and regenerable coordinate caches are not included in the public repository.

The scientific source files are byte-identical to the completed new-assistant snapshot. Shared scientific source files also match the original assistant's final code; new design/heat/data drivers are additional files. Release packaging changes documentation, dependency records, data manifests and portable helper scripts. The complete long-running experimental suite and figure rendering were not rerun during publication; the supplied results and figures remain the source snapshot.

`MANIFEST_SHA256.json` records the tracked release file hashes. The Git commit and annotated tag identify this release independently of any subsequent work. Original and new studies used different environments; historical environment notes must not be interpreted as claiming all results were regenerated on the release-validation machine.
