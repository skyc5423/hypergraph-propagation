from re import I
import numpy as np
from PIL import Image, ImageDraw


def sample_max_similarity(matrix, K):
    N, M = matrix.shape
    sampled_pairs = []
    used_n = set()
    used_m = set()

    # Create a copy of the matrix to modify
    working_matrix = matrix.copy()

    for _ in range(K):
        if len(used_n) == N or len(used_m) == M:
            break  # Stop if we've used all rows or columns

        # Find the maximum value in the working matrix
        max_val = np.max(working_matrix)
        if max_val == -np.inf:
            break  # Stop if no more valid pairs

        # Find the indices of the maximum value
        n, m = np.unravel_index(
            np.argmax(working_matrix), working_matrix.shape)

        # Add the pair to our results
        sampled_pairs.append((n, m))

        # Mark this row and column as used
        used_n.add(n)
        used_m.add(m)

        # Set the entire row and column to -inf to prevent reuse
        working_matrix[n, :] = -np.inf
        working_matrix[:, m] = -np.inf

    return sampled_pairs


def cosine_similarity(x1, x2):
    x2 = x2.T
    x1_norm = x1 / np.linalg.norm(x1, axis=1, keepdims=True)
    x2_norm = x2 / np.linalg.norm(x2, axis=0, keepdims=True)
    return np.dot(x1_norm, x2_norm)


def visualize_salient_patch(image, valid_indices, save_path=None):
    img = Image.open(image)
    draw = ImageDraw.Draw(img)
    for i in valid_indices:
        y = i // 16
        x = i % 16
        draw.rectangle([x * 14, y * 14, (x + 1) * 14, (y + 1) * 14],
                       outline='red', width=2)

    img.save(save_path)
