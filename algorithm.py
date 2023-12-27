import numpy as np
from scipy.spatial import distance_matrix


def distance(x, y, p_norm=2):
    return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)


def sample_inside_sphere(dimensionality, radius, p_norm=2):
    direction_unit_vector = (2 * np.random.rand(dimensionality) - 1)
    direction_unit_vector = direction_unit_vector / distance(direction_unit_vector, np.zeros(dimensionality), p_norm)

    return direction_unit_vector * np.random.rand() * radius


def rbf(d, gamma=1.0):
    if gamma == 0.0:
        return 0.0
    else:
        return np.exp(-(d / gamma) ** 2)


def rbf_score(point, minority_points, gamma=1.0, p_norm=2):
    result = 0.0

    for minority_point in minority_points:
        result += rbf(distance(point, minority_point, p_norm), gamma)

    return result


class NRO:
    def __init__(self, k1=1, k2=1, gamma=1.0, p_norm=2, regions='S', n_samples=500,
                 cleaning_strategy='translate and remove', selection_strategy='proportional',
                 minority_class=None, n=None):
        assert cleaning_strategy in ['translate and remove', 'translate', 'remove']
        assert selection_strategy in ['proportional', 'random']

        self.k1 = k1
        self.k2 = k2
        self.gamma = gamma
        self.p_norm = p_norm
        self.cleaning_strategy = cleaning_strategy
        self.selection_strategy = selection_strategy
        self.regions = regions
        self.n_samples = n_samples
        self.minority_class = minority_class
        self.n = n

    def fit_sample(self, X, y):
        if self.minority_class is None:
            classes = np.unique(y)
            sizes = [sum(y == c) for c in classes]
            minority_class = classes[np.argmin(sizes)]
        else:
            minority_class = self.minority_class

        minority_points = X[y == minority_class].copy()
        majority_points = X[y != minority_class].copy()
        minority_labels = y[y == minority_class].copy()
        majority_labels = y[y != minority_class].copy()

        if self.n is None:
            n = len(majority_points) - len(minority_points)
        else:
            n = self.n

        distances = distance_matrix(minority_points, majority_points)

        radii = np.zeros(len(minority_points))

        translations = np.zeros(majority_points.shape)
        kept_indices = np.full(len(majority_points), True)
        move_majority_list = []

        for i in range(len(minority_points)):
            minority_point = minority_points[i]
            sorted_distances = np.argsort(distances[i])
            sorted_distances_k = np.argsort(distances[i])[self.k1]
            radius = distances[i, sorted_distances_k] - 0.01
            n_majority_points_within_radius = self.k1
            radii[i] = radius

            for j in range(n_majority_points_within_radius):
                majority_point = majority_points[sorted_distances[j]]
                d = distances[i, sorted_distances[j]]

                while d < 1e-20:
                    majority_point += (1e-6 * np.random.rand(len(majority_point)) + 1e-6) * \
                                      np.random.choice([-1.0, 1.0], len(majority_point))
                    d = distance(minority_point, majority_point)

                translation = (radius - d) / d * (majority_point - minority_point)
                translations[sorted_distances[j]] += translation

                kept_indices[sorted_distances[j]] = False

                move_majority = majority_point + translation
                move_majority_list.append(move_majority)
                move_majoritys = np.vstack(move_majority_list)

        if self.cleaning_strategy == 'translate':
            majority_points += translations
        elif self.cleaning_strategy == 'remove':
            majority_points = majority_points[kept_indices]
            majority_labels = majority_labels[kept_indices]

        appended = []

        if self.selection_strategy == 'proportional':
            for i in range(len(minority_points)):
                minority_point = minority_points[i]
                n_synthetic_samples = int(np.round(1.0 / (radii[i] * np.sum(1.0 / radii)) * n))
                r = radii[i]

                if self.gamma is None or ('D' in self.regions and 'S' in self.regions and 'E' in self.regions):
                    for _ in range(n_synthetic_samples):
                        appended.append(minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm))
                else:
                    samples = []
                    scores = []

                    for _ in range(self.n_samples):
                        sample = minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm)
                        score = rbf_score(sample, minority_points, self.gamma, self.p_norm)

                        samples.append(sample)
                        scores.append(score)

                    seed_score = rbf_score(minority_point, minority_points, self.gamma, self.p_norm)

                    dangerous_threshold = seed_score - 0.33 * (seed_score - np.min(scores + [seed_score]))
                    safe_threshold = seed_score + 0.33 * (np.max(scores + [seed_score]) - seed_score)

                    suitable_samples = [minority_point]

                    for sample, score in zip(samples, scores):
                        if score <= dangerous_threshold:
                            case = 'D'
                        elif score >= safe_threshold:
                            case = 'S'
                        else:
                            case = 'E'

                        if case in self.regions:
                            suitable_samples.append(sample)

                    suitable_samples = np.array(suitable_samples)

                    if n_synthetic_samples <= len(suitable_samples):
                        replace = False
                    else:
                        replace = True

                    selected_samples = suitable_samples[
                        np.random.choice(len(suitable_samples), n_synthetic_samples, replace=replace)
                    ]

                    for sample in selected_samples:
                        appended.append(sample)
        elif self.selection_strategy == 'random':
            for i in np.random.choice(range(len(minority_points)), n):
                minority_point = minority_points[i]
                r = radii[i]

                appended.append(minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm))

        appended = np.array(appended)

        if len(appended) > 0:
            if self.cleaning_strategy == 'translate and remove':
                points = np.concatenate([majority_points, minority_points, appended])
                labels = np.concatenate([majority_labels, minority_labels, np.tile([minority_class], len(appended))])
                distances_move = distance_matrix(move_majoritys, points)

                indices_to_remove = []

                for i in range(len(move_majoritys)):
                    move_majority_label = majority_labels[0]
                    sorted_indices = np.argsort(distances_move[i])
                    nearest_indices = sorted_indices[1:self.k2+1]
                    nearest_labels = labels[nearest_indices]

                    if not np.all(nearest_labels == move_majority_label):
                        indices_to_remove.append(i)

                indices_in_points = np.where(np.in1d(points, move_majoritys[indices_to_remove]))[0]

                points = np.delete(points, indices_in_points, axis=0)
                labels = np.delete(labels, indices_in_points)
            else:
                points = np.concatenate([majority_points, minority_points, appended])
                labels = np.concatenate([majority_labels, minority_labels, np.tile([minority_class], len(appended))])
        else:
            points = np.concatenate([majority_points, minority_points])
            labels = np.concatenate([majority_labels, minority_labels])

        return points, labels


class MultiClassNRO:
    def __init__(self, k1=1, k2=1, cleaning_strategy='translate and remove', selection_strategy='proportional',
                 p_norm=2.0, gamma=0.1, method='sampling'):

        assert cleaning_strategy in ['translate and remove', 'translate', 'remove']
        assert selection_strategy in ['proportional', 'random']
        assert method in ['sampling', 'complete']

        self.k1 = k1
        self.k2 = k2
        self.cleaning_strategy = cleaning_strategy
        self.selection_strategy = selection_strategy
        self.p_norm = p_norm
        self.gamma = gamma
        self.method = method


    def fit_sample(self, X, y):
        classes = np.unique(y)
        sizes = np.array([sum(y == c) for c in classes])
        indices = np.argsort(sizes)[::-1]
        classes = classes[indices]
        observations = {c: X[y == c] for c in classes}
        n_max = max(sizes)

        if self.method == 'sampling':
            for i in range(1, len(classes)):
                current_class = classes[i]
                n = n_max - len(observations[current_class])

                used_observations = {}
                unused_observations = {}

                for j in range(0, i):
                    all_indices = list(range(len(observations[classes[j]])))
                    used_indices = np.random.choice(all_indices, int(n_max / i), replace=False)

                    used_observations[classes[j]] = [
                        observations[classes[j]][idx] for idx in all_indices if idx in used_indices
                    ]
                    unused_observations[classes[j]] = [
                        observations[classes[j]][idx] for idx in all_indices if idx not in used_indices
                    ]

                used_observations[current_class] = observations[current_class]
                unused_observations[current_class] = []

                for j in range(i + 1, len(classes)):
                    used_observations[classes[j]] = []
                    unused_observations[classes[j]] = observations[classes[j]]

                took_points, took_labels = MultiClassNRO._took_observations(used_observations)

                nro = NRO(k1=self.k1, k2=self.k2, cleaning_strategy=self.cleaning_strategy,
                          selection_strategy=self.selection_strategy, p_norm=self.p_norm, gamma=self.gamma,
                          minority_class=current_class, n=n)
                oversampled_points, oversampled_labels = nro.fit_sample(took_points, took_labels)

                observations = {}

                for cls in classes:
                    class_oversampled_points = oversampled_points[oversampled_labels == cls]
                    class_unused_points = unused_observations[cls]

                    if len(class_oversampled_points) == 0 and len(class_unused_points) == 0:
                        observations[cls] = np.array([])
                    elif len(class_oversampled_points) == 0:
                        observations[cls] = class_unused_points
                    elif len(class_unused_points) == 0:
                        observations[cls] = class_oversampled_points
                    else:
                        observations[cls] = np.concatenate([class_oversampled_points, class_unused_points])
        else:
            for i in range(1, len(classes)):
                current_class = classes[i]
                n = n_max - len(observations[current_class])

                took_points, took_labels = MultiClassNRO._took_observations(observations)

                nro = NRO(k1=self.k1, k2=self.k2, cleaning_strategy=self.cleaning_strategy,
                          selection_strategy=self.selection_strategy, p_norm=self.p_norm, gamma=self.gamma,
                          minority_class=current_class, n=n)

                oversampled_points, oversampled_labels = nro.fit_sample(took_points, took_labels)

                observations = {cls: oversampled_points[oversampled_labels == cls] for cls in classes}

        took_points, took_labels = MultiClassNRO._took_observations(observations)

        return took_points, took_labels

    @staticmethod
    def _took_observations(observations):
        unpacked_points = []
        unpacked_labels = []

        for cls in observations.keys():
            if len(observations[cls]) > 0:
                unpacked_points.append(observations[cls])
                unpacked_labels.append(np.tile([cls], len(observations[cls])))

        unpacked_points = np.concatenate(unpacked_points)
        unpacked_labels = np.concatenate(unpacked_labels)

        return unpacked_points, unpacked_labels
