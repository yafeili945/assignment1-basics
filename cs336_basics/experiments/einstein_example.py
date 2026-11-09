"""Examples of einops.einsum and einops.rearrange.

Run with:
    python einstein_examples.py
"""

import torch
from einops import einsum, rearrange


def example1_batched_matrix_multiplication() -> None:
    """Apply the same linear transformation at every sequence position."""
    batch_size = 2
    sequence_length = 4
    d_in = 3
    d_out = 5

    D = torch.randn(batch_size, sequence_length, d_in)
    A = torch.randn(d_out, d_in)

    # Basic PyTorch implementation:
    # (batch, sequence, d_in) @ (d_in, d_out)
    # -> (batch, sequence, d_out)
    y_matmul = D @ A.T

    # The dimension names document the contraction directly.
    y_einsum = einsum(
        D,
        A,
        "batch sequence d_in, d_out d_in -> batch sequence d_out",
    )

    # The ellipsis allows D to have any number of leading batch-like axes.
    y_general = einsum(
        D,
        A,
        "... d_in, d_out d_in -> ... d_out",
    )

    torch.testing.assert_close(y_matmul, y_einsum)
    torch.testing.assert_close(y_matmul, y_general)

    print("Example 1")
    print("D:", D.shape)
    print("A:", A.shape)
    print("Y:", y_einsum.shape)


def example2_broadcasted_image_dimming() -> None:
    """Create ten differently dimmed versions of every image."""
    images = torch.randn(64, 128, 128, 3)
    dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

    # Explicitly insert singleton axes so broadcasting lines up correctly.
    dim_value = rearrange(
        dim_by,
        "dim_value -> 1 dim_value 1 1 1",
    )
    images_rearranged = rearrange(
        images,
        "batch height width channel -> batch 1 height width channel",
    )
    dimmed_with_broadcast = images_rearranged * dim_value

    # Einsum expresses the same operation in one step.
    dimmed_with_einsum = einsum(
        images,
        dim_by,
        "batch height width channel, dim_value "
        "-> batch dim_value height width channel",
    )

    torch.testing.assert_close(dimmed_with_broadcast, dimmed_with_einsum)

    print("\nExample 2")
    print("images:", images.shape)
    print("dim_by:", dim_by.shape)
    print("dimmed_images:", dimmed_with_einsum.shape)


def example3_pixel_mixing() -> None:
    """Mix pixels independently within each image channel."""
    batch_size = 64
    height = 32
    width = 32
    channels = 3
    num_pixels = height * width

    channels_last = torch.randn(batch_size, height, width, channels)
    B = torch.randn(num_pixels, num_pixels)

    # Plain PyTorch implementation using view and transpose.
    channels_last_flat = channels_last.view(
        -1,
        num_pixels,
        channels,
    )
    channels_first_flat = channels_last_flat.transpose(1, 2)
    channels_first_flat_transformed = channels_first_flat @ B.T
    channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
    transformed_with_torch = channels_last_flat_transformed.view(*channels_last.shape)

    # The same operation using rearrange and einsum.
    channels_first = rearrange(
        channels_last,
        "batch height width channel -> batch channel (height width)",
    )
    channels_first_transformed = einsum(
        channels_first,
        B,
        "batch channel pixel_in, pixel_out pixel_in "
        "-> batch channel pixel_out",
    )
    transformed_with_einops = rearrange(
        channels_first_transformed,
        "batch channel (height width) -> batch height width channel",
        height=height,
        width=width,
    )

    torch.testing.assert_close(transformed_with_torch, transformed_with_einops)

    print("\nExample 3")
    print("channels_last:", channels_last.shape)
    print("B:", B.shape)
    print("channels_last_transformed:", transformed_with_einops.shape)


def optional_einx_example() -> None:
    """Run the all-in-one einx.dot version when einx is installed."""
    try:
        import einx
    except ImportError:
        print("\nOptional einx example skipped: einx is not installed.")
        return

    batch_size = 64
    height = 32
    width = 32
    channels = 3

    channels_last = torch.randn(batch_size, height, width, channels)
    B = torch.randn(height * width, height * width)

    channels_last_transformed = einx.dot(
        "batch row_in col_in channel, "
        "(row_out col_out) (row_in col_in) "
        "-> batch row_out col_out channel",
        channels_last,
        B,
        col_in=width,
        col_out=width,
    )

    print("\nOptional einx example")
    print("channels_last_transformed:", channels_last_transformed.shape)


if __name__ == "__main__":
    torch.manual_seed(0)
    example1_batched_matrix_multiplication()
    example2_broadcasted_image_dimming()
    example3_pixel_mixing()
    optional_einx_example()