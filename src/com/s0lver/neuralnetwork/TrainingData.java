package com.s0lver.neuralnetwork;

/**
 * Models the training data for a neural network.
 */
public class TrainingData {
    /**
     * The list of input data
     */
    private final double[] data;

    /**
     * The labels for the input data
     */
    private final double[] expectedOutput;

    /**
     * Constructor
     * @param data The input data.
     * @param expectedOutput The labels for the input data.
     */
    public TrainingData(double[] data, double[] expectedOutput) {
        this.data = data;
        this.expectedOutput = expectedOutput;
    }

    /**
     * Gets the data from the training data.
     * @return The data part.
     */
    public double[] getData() {
        return data;
    }

    /**
     * Gets the labels from the training data.
     * @return The label part.
     */
    public double[] getExpectedOutput() {
        return expectedOutput;
    }
}