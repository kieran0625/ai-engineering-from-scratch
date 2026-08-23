# Gradient health monitor exercise for:
#   phases/03-deep-learning-core/04-activation-functions/docs/en.md
# Derivatives follow the activation definitions in that lesson.
# The implementation uses only Python standard-library math and random.
# It records gradients before SGD updates so each epoch is diagnosable.

import math
import random


def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    value = sigmoid(x)
    return value * (1.0 - value)


def relu(x):
    return max(0.0, x)


def relu_derivative(x):
    return 1.0 if x > 0.0 else 0.0


def make_circle_data(n=200, seed=42):
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        x = rng.uniform(-2.0, 2.0)
        y = rng.uniform(-2.0, 2.0)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    else:
        yield value


def new_epoch_stats():
    return {
        "hidden": {"sum_abs": 0.0, "count": 0, "max_abs": 0.0},
        "output": {"sum_abs": 0.0, "count": 0, "max_abs": 0.0},
    }


def record_gradient_stats(stats, gradients):
    for layer, tensors in gradients.items():
        for value in flatten(tensors):
            magnitude = abs(value)
            stats[layer]["sum_abs"] += magnitude
            stats[layer]["count"] += 1
            stats[layer]["max_abs"] = max(stats[layer]["max_abs"], magnitude)


def report_gradient_stats(
    epoch,
    stats,
    vanishing_threshold=0.001,
    exploding_threshold=100.0,
):
    for layer, values in stats.items():
        mean_abs = values["sum_abs"] / values["count"]
        max_abs = values["max_abs"]
        warnings = []

        if mean_abs < vanishing_threshold:
            warnings.append("vanishing gradient")
        if max_abs > exploding_threshold:
            warnings.append("exploding gradient")

        suffix = f"  WARNING: {', '.join(warnings)}" if warnings else ""
        print(
            f"  epoch={epoch:03d} layer={layer:6s} "
            f"mean={mean_abs:.6f} max={max_abs:.6f}{suffix}"
        )


class MonitoredMLP:
    def __init__(self, hidden_size=8, learning_rate=0.1, seed=0):
        rng = random.Random(seed)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.w1 = [
            [rng.gauss(0.0, 0.5) for _ in range(2)]
            for _ in range(hidden_size)
        ]
        self.b1 = [0.0] * hidden_size
        self.w2 = [rng.gauss(0.0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x):
        self.x = x
        self.z1 = [
            sum(self.w1[i][j] * x[j] for j in range(2)) + self.b1[i]
            for i in range(self.hidden_size)
        ]
        self.h = [relu(z) for z in self.z1]
        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size))
        self.z2 += self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def gradients(self, target):
        # Loss is 0.5 * (prediction - target)^2.
        d_out = (self.out - target) * sigmoid_derivative(self.z2)
        grad_w2 = [d_out * h for h in self.h]
        grad_b2 = d_out
        grad_w1 = [[0.0, 0.0] for _ in range(self.hidden_size)]
        grad_b1 = [0.0] * self.hidden_size

        for i in range(self.hidden_size):
            d_z1 = d_out * self.w2[i] * relu_derivative(self.z1[i])
            grad_b1[i] = d_z1
            for j in range(2):
                grad_w1[i][j] = d_z1 * self.x[j]

        return {
            "hidden": [grad_w1, grad_b1],
            "output": [grad_w2, grad_b2],
        }

    def apply_gradients(self, gradients):
        grad_w1, grad_b1 = gradients["hidden"]
        grad_w2, grad_b2 = gradients["output"]

        for i in range(self.hidden_size):
            for j in range(2):
                self.w1[i][j] -= self.learning_rate * grad_w1[i][j]
            self.b1[i] -= self.learning_rate * grad_b1[i]
            self.w2[i] -= self.learning_rate * grad_w2[i]
        self.b2 -= self.learning_rate * grad_b2

    def train(self, data, epochs=5):
        for epoch in range(epochs):
            stats = new_epoch_stats()
            total_loss = 0.0
            correct = 0

            for x, target in data:
                prediction = self.forward(x)
                total_loss += 0.5 * (prediction - target) ** 2
                gradients = self.gradients(target)

                # Diagnose the original gradient before applying the update.
                record_gradient_stats(stats, gradients)
                self.apply_gradients(gradients)

                if (prediction >= 0.5) == (target >= 0.5):
                    correct += 1

            average_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100.0
            print(
                f"epoch={epoch:03d} loss={average_loss:.6f} "
                f"accuracy={accuracy:.1f}%"
            )
            report_gradient_stats(epoch, stats)


if __name__ == "__main__":
    print("Gradient health monitor: ReLU MLP on circle data")
    model = MonitoredMLP(hidden_size=8, learning_rate=0.1, seed=0)
    model.train(make_circle_data(), epochs=5)
