package com.s0lver;

import java.util.Arrays;

public class Prediction {
    private final double error;
    private final double[] output;
    private final double[] expected;

    public Prediction(double[] output, double[] expected, double error) {
        this.output = output;
        this.expected = expected;
        this.error = error;
    }

    @Override
    public String toString() {
        return String.format("Predicted: %s, Expected: %s, error=%s", Arrays.toString(output), Arrays.toString(expected), error);
    }
}
