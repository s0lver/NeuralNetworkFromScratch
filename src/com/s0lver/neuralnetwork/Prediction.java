package com.s0lver.neuralnetwork;

import java.util.Arrays;

import static com.s0lver.neuralnetwork.Utils.calculateError;

/**
 * Models a prediction by a neural network.
 */
public class Prediction {
    /**
     * The error for the prediction
     */
    private final double error;

    /**
     * The produced output.
     */
    private final double[] output;

    /**
     * The expected output.
     */
    private final double[] expected;

    /**
     * Constructor
     *
     * @param output   The produced output.
     * @param expected The expected output.
     */
    public Prediction(double[] output, double[] expected) {
        this.output = output;
        this.expected = expected;
        this.error = calculateError(output, expected);
    }


    @Override
    public String toString() {
        return String.format("Predicted: %s, Expected: %s, error=%s", Arrays.toString(output), Arrays.toString(expected), error);
    }
}