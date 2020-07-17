package com.s0lver;

public class Neuron {
    private double[] weights;
    private double[] cacheWeights;
    private double gradient;
    private double output;

    static double minWeightValue;
    static double maxWeightValue;

    /**
     * Constructor for the neurons in hidden and output neurons
     *
     * @param weights The weights to assign
     */
    public Neuron(double[] weights) {
        this.weights = weights;
        this.cacheWeights = weights;
        this.gradient = 0;
    }

    /**
     * Constructor used for the bias neurons
     */
    public Neuron() {
        this.weights = null;
        this.gradient = 0;
        this.output = 1;
    }

    public static void setRangeWeight(double minWeightValue, double maxWeightValue) {
        Neuron.minWeightValue = minWeightValue;
        Neuron.maxWeightValue = maxWeightValue;
    }

    /**
     * Updates the weights with the calculated values after the backpropagation step.
     */
    public void updateWeights() {
        if (this.weights.length - 1 >= 0)
            System.arraycopy(this.cacheWeights, 1, this.weights, 1, this.weights.length - 1);
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public int getNumOfWeights() {
        return weights.length;
    }

    public double getWeight(int index) {
        return weights[index];
    }

    public void setGradient(double gradient) {
        this.gradient = gradient;
    }

    public void setCacheWeight(int index, double value) {
        cacheWeights[index] = value;
    }

    public double getGradient() {
        return gradient;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }
}
