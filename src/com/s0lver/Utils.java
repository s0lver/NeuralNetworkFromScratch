package com.s0lver;

import java.util.Random;

public class Utils {
    /**
     * For reproducibility purposes
     */
    static Random random = new Random(0);

    public static double generateRandomDouble(double min, double max) {
        return min + random.nextDouble() * (max - min);
    }

    public static double Sigmoid(double x) {
        return 1 / (1 + Math.pow(Math.E, -x));
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
        // the start at 1 is because currently the output layer has a bias neuron
        for (int i = 0; i < predicted.length; i++) {
            final double diff = expected[i] - predicted[i];
            final double halfSquaredDiff = Math.pow(diff, 2) / 2;
            sum += halfSquaredDiff;
        }
        return sum;
    }
}
