package com.s0lver;

import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        final TrainingData[] trainingDataset = createXorDataset();
        ArrayList<int[]> dimensions = new ArrayList<>();
        dimensions.add(new int[]{0, 2});
        dimensions.add(new int[]{2, 6});
        dimensions.add(new int[]{6, 1});

        NeuralNetwork neuralNetwork = new NeuralNetwork(dimensions);

        System.out.println("Xor problem");
        System.out.println("Output before training:");

        for (TrainingData trainingData : trainingDataset) {
            final double[] prediction = neuralNetwork.forward(trainingData.getData());
            System.out.println(prediction[0]);
        }

        neuralNetwork.train(trainingDataset, 1000000, 0.05);

        System.out.println();
        System.out.println("Output after training");
        for (TrainingData trainingData : trainingDataset) {
            final double[] prediction = neuralNetwork.forward(trainingData.getData());
            System.out.println(prediction[0]);
        }
    }

    public static TrainingData[] createXorDataset() {
        return new TrainingData[]{
                new TrainingData(new double[]{0, 0}, new double[]{0}),
                new TrainingData(new double[]{0, 1}, new double[]{1}),
                new TrainingData(new double[]{1, 0}, new double[]{1}),
                new TrainingData(new double[]{1, 1}, new double[]{0}),
        };
    }
}
