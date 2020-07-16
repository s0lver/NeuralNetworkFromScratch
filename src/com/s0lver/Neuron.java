package com.s0lver;

public class Neuron {
    private double[] weights;
    private double[] cacheWeights;
    private double gradient;
    private double bias;
    private double value;

    static double minWeightValue;
    static double maxWeightValue;

    /**
     * Constructor for the neurons in hidden and output neurons
     *
     * @param weights The weights to assign
     * @param bias    The bias to assign
     */
    public Neuron(double[] weights, double bias) {
        this.weights = weights;
        this.bias = bias;
        this.cacheWeights = weights;
        this.gradient = 0;
    }

    /**
     * Constructor used for the input neurons
     *
     * @param value The value to assign.
     */
    public Neuron(double value) {
        this.weights = null;
        this.bias = -1;
        this.gradient = -1;
        this.value = value;
    }

    public static void setRangeWeight(double minWeightValue, double maxWeightValue) {
        Neuron.minWeightValue = minWeightValue;
        Neuron.maxWeightValue = maxWeightValue;
    }

    /**
     * Updates the weights with the calculated values after the backpropagation step.
     */
    public void updateWeight() {
        this.weights = cacheWeights;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    //    public double[] getWeights() {
    //        return weights;
    //    }

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
}
