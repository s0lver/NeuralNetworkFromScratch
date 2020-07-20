package com.s0lver.neuralnetwork;

import java.util.Random;

/**
 * Static util functions
 */
public class Utils {
    /**
     * For reproducibility purposes
     */
    static Random random = new Random(0);

    /**
     * Generates a random number within the specified interval.
     *
     * @param min The lowest threshold for the number to generate.
     * @param max The highest threshold for the number to generate.
     * @return A random number within specified thresholds.
     */
    public static double generateRandomDouble(double min, double max) {
        return min + random.nextDouble() * (max - min);
    }

    /**
     * Calculates the sigmoid value for the specified input.
     *
     * @param x The input value.
     * @return The calculated sigmoid value
     */
    public static double Sigmoid(double x) {
        return 1 / (1 + Math.pow(Math.E, -x));
    }

    /**
     * Calculates the value for the sigmoid derivative of the specified input.
     *
     * @param input The input whose sigmoid derivative should be calculated.
     * @return The sigmoid derivative value.
     */
    public static double calculateSigmoidDerivative(double input) {
        return input * (1 - input);
    }

    /**
     * Calculates the squared error (the length of array parameters must be the same)
     *
     * @param predicted The predicted data
     * @param expected  The expected output
     * @return The calculated squared error
     */
    public static double calculateError(double[] predicted, double[] expected) {
        double sum = 0;
        for (int i = 0; i < predicted.length; i++) {
            final double diff = expected[i] - predicted[i];
            final double halfSquaredDiff = Math.pow(diff, 2) / 2;
            sum += halfSquaredDiff;
        }
        return sum;
    }

    /**
     * Creates a list of weights of the specified dimension.
     *
     * @param numOfWeights The number of weights to create
     * @return The list of created numOfWeights weights
     */
    public static double[] generateRandomWeights(int numOfWeights) {
        double[] weights = new double[numOfWeights];
        for (int j = 0; j < numOfWeights; j++) {
            weights[j] = Utils.generateRandomDouble(Neuron.minWeightValue, Neuron.maxWeightValue);
        }
        return weights;
    }
}