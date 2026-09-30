defmodule NxEigen.FFTTest do
  use ExUnit.Case, async: true

  # Nx's own FFT doctests only use handfuls of samples, which leaves the
  # interesting paths of the FFT backends untested: kissfft's radix-2/3/4/5
  # butterflies, and the Bluestein convolution the `eigen` backend falls back to
  # once the largest prime factor exceeds 64. These lengths straddle that
  # threshold, and the binary backend is the oracle.
  @radix_lengths [2, 7, 13, 37, 64, 128, 1000]
  @bluestein_lengths [101, 127, 1021]

  @tolerances %{f32: 1.0e-4, f64: 1.0e-10}

  defp signal(n), do: for(i <- 0..(n - 1), do: :math.sin(i * 0.7) + 0.3 * i)

  defp relative_error(actual, expected) do
    error =
      actual
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.subtract(expected)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    scale = expected |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    error / max(scale, 1.0e-12)
  end

  for op <- [:fft, :ifft],
      length <- @radix_lengths ++ @bluestein_lengths,
      type <- [:f32, :f64] do
    test "#{op} of #{length} #{type} samples matches the binary backend" do
      data = signal(unquote(length))

      expected =
        data
        |> Nx.tensor(type: unquote(type), backend: Nx.BinaryBackend)
        |> then(&apply(Nx, unquote(op), [&1]))

      actual =
        data
        |> Nx.tensor(type: unquote(type), backend: NxEigen.Backend)
        |> then(&apply(Nx, unquote(op), [&1]))

      assert relative_error(actual, expected) < @tolerances[unquote(type)]
    end
  end

  for length <- @radix_lengths ++ @bluestein_lengths do
    test "ifft of fft returns the original #{length} samples" do
      tensor = Nx.tensor(signal(unquote(length)), type: :f64, backend: NxEigen.Backend)

      round_tripped = tensor |> Nx.fft() |> Nx.ifft()

      assert relative_error(round_tripped, Nx.backend_transfer(tensor, Nx.BinaryBackend)) <
               1.0e-10
    end
  end

  for type <- [:f32, :f64] do
    test "unit impulse of #{type} is a flat spectrum" do
      impulse = Nx.tensor([1, 0, 0, 0], type: unquote(type), backend: NxEigen.Backend)

      expected =
        Nx.tensor([1, 1, 1, 1],
          type: Nx.Type.to_complex(unquote(type)),
          backend: Nx.BinaryBackend
        )

      assert relative_error(Nx.fft(impulse), expected) < @tolerances[unquote(type)]
    end

    test "constant #{type} input is a single DC bin" do
      constant = Nx.tensor([1, 1, 1, 1], type: unquote(type), backend: NxEigen.Backend)

      expected =
        Nx.tensor([4, 0, 0, 0],
          type: Nx.Type.to_complex(unquote(type)),
          backend: Nx.BinaryBackend
        )

      assert relative_error(Nx.fft(constant), expected) < @tolerances[unquote(type)]
    end

    test "length 1 #{type} transform is the sample" do
      sample = Nx.tensor([3.25], type: unquote(type), backend: NxEigen.Backend)

      expected =
        Nx.tensor([Complex.new(3.25, 0)],
          type: Nx.Type.to_complex(unquote(type)),
          backend: Nx.BinaryBackend
        )

      spectrum = Nx.fft(sample)
      assert relative_error(spectrum, expected) < @tolerances[unquote(type)]

      assert relative_error(Nx.ifft(spectrum), Nx.backend_transfer(sample, Nx.BinaryBackend)) <
               @tolerances[unquote(type)]
    end
  end

  test "complex impulse keeps both components" do
    impulse =
      Nx.tensor([Complex.new(1, -2), 0, 0, 0], type: :c128, backend: NxEigen.Backend)

    expected =
      Nx.tensor(
        [Complex.new(1, -2), Complex.new(1, -2), Complex.new(1, -2), Complex.new(1, -2)],
        type: :c128,
        backend: Nx.BinaryBackend
      )

    assert relative_error(Nx.fft(impulse), expected) < 1.0e-10
  end

  test "complex input matches the binary backend" do
    data = [
      Complex.new(1, 0.5),
      Complex.new(-1, 2),
      Complex.new(0.25, -0.5),
      Complex.new(3, 1),
      Complex.new(0, 0),
      Complex.new(-0.5, 0.25),
      Complex.new(2, -1)
    ]

    expected = data |> Nx.tensor(type: :c128, backend: Nx.BinaryBackend) |> Nx.fft()
    actual = data |> Nx.tensor(type: :c128, backend: NxEigen.Backend) |> Nx.fft()

    assert relative_error(actual, expected) < 1.0e-10

    assert relative_error(
             Nx.ifft(actual),
             Nx.tensor(data, type: :c128, backend: Nx.BinaryBackend)
           ) <
             1.0e-10
  end

  test "fft along each axis of a matrix matches the binary backend" do
    data = [[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 1.0], [5.0, 0.0, 1.0, 0.0]]

    for axis <- [0, -1] do
      expected = data |> Nx.tensor(type: :f64, backend: Nx.BinaryBackend) |> Nx.fft(axis: axis)
      actual = data |> Nx.tensor(type: :f64, backend: NxEigen.Backend) |> Nx.fft(axis: axis)

      assert relative_error(actual, expected) < 1.0e-10
    end
  end

  # Excluded unless the NIF was built with NX_EIGEN_FFT_LIB=none.
  # The stubs return -1 and the NIF raises; the rest of this file needs a real transform.
  @tag :fft_disabled
  test "fft reports that support is not compiled in" do
    tensor = Nx.tensor([1.0, 0.0], backend: NxEigen.Backend)

    assert_raise RuntimeError,
                 "FFT operation failed (rc=-1). Is FFT support compiled in? See NX_EIGEN_FFT_LIB / NX_EIGEN_FFT_SO in the README.",
                 fn -> Nx.fft(tensor) end
  end

  test "a length with a large prime factor completes promptly" do
    # Without Bluestein this is a 9-second O(n²) transform on the `eigen`
    # backend, so a generous ceiling still catches a regression.
    tensor = Nx.tensor(signal(65_521), type: :f64, backend: NxEigen.Backend)

    {microseconds, _} = :timer.tc(fn -> Nx.fft(tensor) end)

    assert microseconds < 1_000_000
  end
end
