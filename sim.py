import logging
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple

# Set up logging configuration
logging.basicConfig(level=logging.WARNING)

class Simulation:
    """
    A datastream generator that simulates a datastream with random anomalies and concept drift.
    """

    def __init__(self, max_steps: int = 1000):
        """
        Initialize the simulation with the specified number of steps.
        """
        self.max_steps = max_steps
        self.t = 0
        self.seasonal_period = 100
        self.anomaly_prob = 0.05
        self.drift_rate = 0.001
        self.seasonal_drift_rate = 0.0001

    def __iter__(self):
        """
        Return the iterator object itself.
        """
        return self

    def __next__(self) -> Tuple[float, bool]:
        """
        Generate the next data point in the simulation. Implements the logic for an iterator.
        """
        if self.t >= self.max_steps:
            raise StopIteration

        # Generate the data point based on the regular pattern, seasonal pattern, and noise
        regular_pattern = self.t * 0.1 * (1 + self.drift_rate * self.t)
        seasonal_pattern = (
            10
            * np.sin(2 * np.pi * (self.t % self.seasonal_period) / self.seasonal_period)
            * (1 + self.seasonal_drift_rate * self.t)
        )
        noise = np.random.normal(0, 1)
        value = regular_pattern + seasonal_pattern + noise

        # Check for anomaly
        is_anomaly = False
        if random.random() < self.anomaly_prob:
            anomaly = np.random.choice([-1, 1]) * np.random.uniform(5, 20)
            value += anomaly
            is_anomaly = True

        self.t += 1
        return value, is_anomaly


class EWMAAnomalyDetector:
    def __init__(self, alpha=0.5, threshold=3):
        """
        Initialize the EWMA anomaly detector with the specified alpha and threshold.
        """
        self.alpha = alpha
        self.threshold = threshold
        self.prev_ewma = 0

    def detect(self, data_point: float) -> bool:
        """
        Detect anomalies using Exponential Weighted Moving Average (EWMA).
        """

        # Calculate the Exponential Weighted Moving Average (EWMA) with smoothing factor alpha
        ewma = self.alpha * data_point + (1 - self.alpha) * self.prev_ewma

        # Validate the EWMA value and reset to 0 if it becomes invalid
        if not np.isfinite(ewma):
            logging.warning(
                f"EWMA became unstable with data point {data_point}. Resetting EWMA to 0."
            )
            ewma = 0

        # Update the previous EWMA value
        self.prev_ewma = ewma

        # Check if the deviation between the data point and the EWMA is above a threshold
        deviation = abs(data_point - ewma)
        if deviation > self.threshold:
            return True
        return False


class EWMAAnimation:
    def __init__(self, max_data_points=250, interval=250):
        """
        Initialize the animation with the specified number of data points and interval.
        """

        # Set the maximum number of data points and the interval for updating the plot
        self.max_data_points = max_data_points
        self.interval = interval

        # Initialize the data stream and lists to store data points and anomalies
        self.data_stream = Simulation(max_steps=max_data_points)
        self.data_points = []
        self.predicted_anomalies = []

        # Initialize the anomaly detector
        self.anomaly_detector = EWMAAnomalyDetector(alpha=0.3, threshold=3)

        # Initialize the plot
        self.fig, self.ax = plt.subplots(figsize=(15, 5))

        # Set up the plot labels and title
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.ax.set_title("EWMA Anomaly Detection")

        # Set up the plot elements
        (self.line,) = self.ax.plot([], [], label="Data")
        self.predicted_anomaly_scatter = self.ax.scatter(
            [], [], color="blue", label="Predicted Anomalies"
        )
        self.ax.set_xlim(0, self.interval)
        self.ax.set_ylim(-30, 30)
        self.ax.legend()

    def update(self, frame):
        """
        Update the plot with the next data point from the data stream.
        """

        # Get the next data point and check for anomalies
        value, true_anomaly = next(self.data_stream)
        self.data_points.append(value)

        # Update the EWMA anomaly detection model
        is_predicted_anomaly = self.anomaly_detector.detect(value)
        self.predicted_anomalies.append(is_predicted_anomaly)

        # Cross-validate anomalies (true vs predicted)
        if true_anomaly and not is_predicted_anomaly:
            logging.warning(
                f"True anomaly at t={len(self.data_points)} was missed by the detector."
            )
        elif is_predicted_anomaly and not true_anomaly:
            logging.warning(f"False positive detected at t={len(self.data_points)}.")

        # Update the line plot of data points
        self.line.set_data(range(len(self.data_points)), self.data_points)

        # Update the scatter plot of predicted anomalies
        predicted_anomaly_indices = [
            i for i, x in enumerate(self.predicted_anomalies) if x
        ]
        predicted_anomaly_values = [
            self.data_points[i] for i in predicted_anomaly_indices
        ]

        # Efficiently update the scatter plot by setting new offsets
        new_offsets = np.c_[predicted_anomaly_indices, predicted_anomaly_values]
        self.predicted_anomaly_scatter.set_offsets(new_offsets)

        # Adjust the x axis limits to keep the last `interval` data points in view
        if len(self.data_points) > self.interval:
            self.ax.set_xlim(
                len(self.data_points) - self.interval, len(self.data_points)
            )

        # Adjust the y-axis limits only if the new data crosses a threshold
        y_min, y_max = self.ax.get_ylim()
        if value > y_max - 5 or value < y_min + 5:
            self.ax.set_ylim(min(self.data_points) - 5, max(self.data_points) + 5)

        # Return the updated plot elements
        return self.line, self.predicted_anomaly_scatter

    def run(self):
        """
        Run the animation.
        """

        # Create the animation
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=range(self.max_data_points),
            interval=self.interval,
            blit=False,
            repeat=False,
        )

        # Display the animation
        plt.show()


if __name__ == "__main__":
    ewma_animation = EWMAAnimation(max_data_points=500, interval=250)

    # Run the animation
    try:
        ewma_animation.run()
    except KeyboardInterrupt:
        print()
        print("Exiting...")
