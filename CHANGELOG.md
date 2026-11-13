# Changelog

## 0.2.0 (2026-11-13)

Every name in code, on the wire and on disk is `xblick` now. Only the prose name of the project stays Durchblick.
The node needs the core `xblick` 0.5 and does not work with programs of Durchblick 0.4 (`blick`) or older: run every
program of a setup at 0.5.

| What | 0.1.0 | 0.2.0 |
| --- | --- | --- |
| Repository | github.com/xblick/durchblick-tapnext | github.com/xblick/xblick-tapnext |
| Distribution | `durchblick-tapnext` | `xblick-tapnext` |
| Python package | `durchblick_tapnext` | `xblick_tapnext` |
| Commands | `durchblick-tapnext`, `durchblick-tapnext-queries` | `xblick-tapnext`, `xblick-tapnext-queries` |
| Program name in log lines, the logger and reports | `durchblick-tapnext` | `xblick-tapnext` |
| Core dependency | `blick>=0.4,<0.5`, git main of github.com/xblick/durchblick | `xblick>=0.5.0,<0.6`, github.com/xblick/xblick at 45908e6 |
| Core imports and command | `import blick`, `blick rerun`, `blick play`, `blick param` | `import xblick`, `xblick rerun`, `xblick play`, `xblick param` |
| Links to the core's docs | github.com/xblick/durchblick/blob/main/docs/... | github.com/xblick/xblick/blob/main/docs/... |
| Other nodes named in the docs | `durchblick-sam3` (link github.com/xblick/durchblick-sam3), `durchblick-nemo`, `durchblick-foundationpose`, `durchblick-fast-foundationstereo` | `xblick-sam3` (github.com/xblick/xblick-sam3), `xblick-nemo`, `xblick-foundationpose`, `xblick-fast-foundationstereo` |
| CI | secret `DURCHBLICK_READ_TOKEN` to read the private core | none |

## 0.1.0

The node as `durchblick-tapnext` (Python package `durchblick_tapnext`) on the core `blick` 0.4.
