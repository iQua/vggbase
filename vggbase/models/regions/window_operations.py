"""
The useful functions and modules for window operations.

"""


def window_count(H, W, window_size):
    """ Compute the number of windows
    Args:
        W, W: the height and width size
        window_size (int): window size

    Returns:
        number_of_windows: the total number of windows
    """
    assert H % window_size == 0
    assert W % window_size == 0
    height_number_windows = H // window_size
    width_number_windows = W // window_size

    return height_number_windows * width_number_windows


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape

    # with view, will always share its data with the original tensor
    # B, number_of_windows_in_h, window_size, number_of_windoes_in_w, window_size, C
    h_windows = H // window_size
    w_windows = W // window_size
    x = x.view(B, h_windows, window_size, w_windows, window_size,
               C)

    window_resolution = [h_windows, w_windows]

    # obtain the windows' tensor:
    #   total_number_of_windows, window_size, window_size, C
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size, window_size, C)

    return windows, window_resolution


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
