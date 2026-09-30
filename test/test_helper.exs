# The disabled-FFT case only exists in a build with NX_EIGEN_FFT_LIB=none.
exclude = if System.get_env("NX_EIGEN_FFT_LIB") == "none", do: [], else: [fft_disabled: true]

ExUnit.start(exclude: exclude)
