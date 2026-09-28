# WebPlotDigitizer 5.3

Source: https://github.com/automeris-io/WebPlotDigitizer
Commit: 3a3ecb11606945d0701c8a488777e6861be70056
License: AGPL-3.0 (see LICENSE).

This local build uses the generated English development HTML and unminified JavaScript. Changes: third-party asset paths relocated; the workbench bridge loaded from front_end/wpd-bridge.js; imageManager.js propagates file/image/PDF loading failures instead of leaving pending promises. Templates and the renderer are preserved for source availability.

Third-party assets retain their licenses. tarballjs source is fixed to 64ea5eb78f7fc018a223207e67f4f863fcc5d3c5 at https://github.com/ankitrohatgi/tarballjs . No npm install is needed to run the local workbench.
