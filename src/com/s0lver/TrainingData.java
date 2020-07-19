package com.s0lver;

public class TrainingData {
    private final double[] data;
    private final double[] expectedOutput;

    public TrainingData(double[] data, double[] expectedOutput) {
        this.data = data;
        this.expectedOutput = expectedOutput;
    }

    public double[] getData() {
        return data;
    }

    public double[] getExpectedOutput() {
        return expectedOutput;
    }
}