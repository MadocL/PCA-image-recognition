from cv2 import imread, imwrite, cvtColor, resize, COLOR_RGB2GRAY, COLOR_GRAY2RGB, INTER_LINEAR
from matplotlib.pyplot import figure, scatter, quiver, xlim, ylim, axhline, axvline, grid, gca, legend, title
from numpy import array, mean as np_mean, sqrt, linalg, newaxis, min as np_min, max as np_max, uint8
from os import listdir

# ! Images must follow this naming rule: 'XXYY.jpg' where XX is the class ID and YY is the image ID within class XX.
TRAIN_IMAGES_DIRECTORY = "src/train"
TEST_IMAGES_DIRECTORY = "/src/test"
RESULTS_DIRECTORY = "pca_results"

INITIAL_IMAGE_HEIGHT = 3024
INITIAL_IMAGE_WIDTH = 4032
IMAGE_REDUCTION_SCALE = 24  # must be a common factor of height and width (may be the greatest, as is the case)
KEPT_EIGEN_VECTORS_RATE = 0.16  # arbitrary rate, fine-tuned after seeing results


def load_image(image_path):
    return resize(
        src=cvtColor(
            imread(image_path),
            COLOR_RGB2GRAY
        ),
        dsize=None,
        fx=1/IMAGE_REDUCTION_SCALE,
        fy=1/IMAGE_REDUCTION_SCALE,
        interpolation=INTER_LINEAR
    ).flatten()


def load_train_images(image_names):
    return array([load_image(f"{TRAIN_IMAGES_DIRECTORY}/{name}") for name in image_names]).T


def compute_mean_vector_and_normalize_image_vectors(image_vectors):
    mean_vector = np_mean(image_vectors, axis=1)  # vector named "psi"
    normalized_image_vectors = image_vectors - mean_vector[:, newaxis]  # matrix named "A"

    return mean_vector, normalized_image_vectors


def compute_eigen_values_and_vectors(normalized_image_vectors):
    A = normalized_image_vectors

    return linalg.eigh(A.T @ A)  # linalg.eigh() return directly eigen values and eigen vectors


def reduce_number_of_eigen_vector(eigen_values, eigen_vectors):
    eigen_value_with_vector = sorted(zip(eigen_values, eigen_vectors))

    # amount of vectors to keep with highest eigen values
    highest_eigen_values_amount = int(len(eigen_value_with_vector) * KEPT_EIGEN_VECTORS_RATE)
    reduced_eigen_vectors = eigen_value_with_vector[-highest_eigen_values_amount:]

    return array([k for k, _ in reduced_eigen_vectors]), array([v for _, v in reduced_eigen_vectors])


def compute_final_eigen_vectors(eigen_vectors, normalized_image_vectors):
    return array([normalized_image_vectors @ ev for ev in eigen_vectors])


def save_eigen_vectors(eigen_vectors):
    images_shape_after_reduction = (
        int(INITIAL_IMAGE_HEIGHT * (1/IMAGE_REDUCTION_SCALE)),
        int(INITIAL_IMAGE_WIDTH * (1/IMAGE_REDUCTION_SCALE))
    )

    for i in range(len(eigen_vectors)):
        min_value = np_min(eigen_vectors[i])
        max_value = np_max(eigen_vectors[i])

        eigen_vector_image = cvtColor(
            (
                (
                    (eigen_vectors[i] - min_value) / (max_value - min_value)
                ) * 255
            )
            .astype(uint8)
            .reshape(images_shape_after_reduction),
            COLOR_GRAY2RGB
        )

        imwrite(f"{RESULTS_DIRECTORY}/eigen_vectors_img/vector_{i}.jpg", eigen_vector_image)


def save_pca_scatter_plot_with_vectors(weights, eigen_vectors, scale_factor):

    pc1 = weights[:, 0]
    pc2 = weights[:, 1]

    fig = figure(figsize=(8, 6))
    scatter(pc1, pc2, c='b', marker='o', label='Projected images')

    pc1_range = max(pc1) - min(pc1)
    pc2_range = max(pc2) - min(pc2)

    scale = scale_factor * sqrt(pc1_range**2 + pc2_range**2) / 10

    origin = [0], [0]
    quiver(
        *origin, eigen_vectors[0, 0] * scale, eigen_vectors[1, 0] * scale,
        color=['r'], scale=1, label='PC1', angles='xy', scale_units='xy'
    )
    quiver(
        *origin, eigen_vectors[0, 1] * scale, eigen_vectors[1, 1] * scale,
        color=['g'], scale=1, label='PC2', angles='xy', scale_units='xy'
    )
    xlim([min(pc1) - 1, max(pc1) + 1])
    ylim([min(pc2) - 1, max(pc2) + 1])

    axhline(0, color='black', linewidth=0.5)
    axvline(0, color='black', linewidth=0.5)
    grid(True)

    gca().set_xlabel("1st Principal Component (PC1)")
    gca().set_ylabel("2nd Principal Component (PC2)")
    legend()
    title("Images and eigen vectors projection (PC1, PC2)")

    fig.savefig(f"{RESULTS_DIRECTORY}/pca_scatter_plot.png", format='png')


def compute_train_images_weights(eigen_vectors, normalized_image_vectors):
    return array([eigen_vectors @ d for d in normalized_image_vectors.T])


def pca_training():
    train_image_names = sorted(listdir(TRAIN_IMAGES_DIRECTORY))
    mean_vector, normalized_image_vectors = compute_mean_vector_and_normalize_image_vectors(
        image_vectors=load_train_images(train_image_names)
    )
    eigen_values, eigen_vectors = compute_eigen_values_and_vectors(normalized_image_vectors)
    eigen_vectors = compute_final_eigen_vectors(eigen_vectors, normalized_image_vectors)

    eigen_values, eigen_vectors = reduce_number_of_eigen_vector(eigen_values, eigen_vectors)
    save_eigen_vectors(eigen_vectors)

    train_images_weights = compute_train_images_weights(eigen_vectors, normalized_image_vectors)
    save_pca_scatter_plot_with_vectors(train_images_weights, eigen_vectors, scale_factor=0.2)  # arbitrary scale

    return train_image_names, mean_vector, eigen_vectors, train_images_weights


def compute_weights_distances_with_train_images_ones(train_images_weights, weights, image_names):

    weights_distances = [
        (image_name, linalg.norm(weights - tiw))  # euclidean distance
        for image_name, tiw in zip(image_names, train_images_weights)
    ]
    return sorted(weights_distances, key=lambda x: x[1])


def pca_tests(train_image_names, mean_vector, eigen_vectors, train_images_weights):
    test_image_names = sorted(listdir(TEST_IMAGES_DIRECTORY))
    correct_count = 0

    for test_image_name in test_image_names:
        image = load_image(f"{TEST_IMAGES_DIRECTORY}/{test_image_name}")
        weights = eigen_vectors @ (image - mean_vector)  # vector named "omega"

        weight_distances_with_train_images = compute_weights_distances_with_train_images_ones(
            train_images_weights, weights, train_image_names
        )

        is_prediction_correct = test_image_name[0:2] == weight_distances_with_train_images[0][0][0:2]
        if is_prediction_correct:
            correct_count = correct_count + 1

        print(
            f"Nearest image from '{test_image_name}':", weight_distances_with_train_images[0],
            f"\n{is_prediction_correct=}",
        )

    print("PCA Performance :", int(correct_count/len(test_image_names)*100), "%")
    print(f"with {IMAGE_REDUCTION_SCALE=} and {KEPT_EIGEN_VECTORS_RATE=}")


if __name__ == '__main__':

    train_image_names, mean_vector, eigen_vectors, train_images_weights = pca_training()

    pca_tests(train_image_names, mean_vector, eigen_vectors, train_images_weights)
