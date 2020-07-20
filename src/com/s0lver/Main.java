
package com.s0lver;

import java.util.ArrayList;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        Main main = new Main();
        //        main.executeXorProblem();
        //        System.out.println();
        main.executeBlogExample();
    }

    private void executeBlogExample() {
        System.out.println("Example from post https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/comment-page-13/");
        final TrainingData[] trainingDataset = createBlogDataset();
        ArrayList<int[]> dimensions = new ArrayList<>();
        dimensions.add(new int[]{0, 2});
        dimensions.add(new int[]{2, 2});
        dimensions.add(new int[]{2, 2});

        NeuralNetwork neuralNetwork = new NeuralNetwork(dimensions);

        neuralNetwork.setWeightsForNeuronInLayer(1, 0, new double[]{0.35, 0.15, 0.2});
        neuralNetwork.setWeightsForNeuronInLayer(1, 1, new double[]{0.35, 0.25, 0.3});
        neuralNetwork.setWeightsForNeuronInLayer(2, 0, new double[]{0.6, 0.40, 0.45});
        neuralNetwork.setWeightsForNeuronInLayer(2, 1, new double[]{0.6, 0.50, 0.55});


        System.out.println("Output before training:");

        var prediction = neuralNetwork.predict(trainingDataset[0]);
        System.out.println(prediction);

        neuralNetwork.train(trainingDataset, 10000, 0.5);

        System.out.println("Output after training:");
        prediction = neuralNetwork.predict(trainingDataset[0]);
        System.out.println(prediction);

        for (Layer layer : neuralNetwork.getLayers()) {
            for (Neuron neuron : layer.getNeurons()) {
                System.out.println(Arrays.toString(neuron.getWeights()));
            }
        }
    }

    public void executeXorProblem() {
        final TrainingData[] trainingDataset = createXorDataset();
        ArrayList<int[]> dimensions = new ArrayList<>();
        dimensions.add(new int[]{0, 2});
        dimensions.add(new int[]{2, 6});
        dimensions.add(new int[]{6, 1});

        NeuralNetwork neuralNetwork = new NeuralNetwork(dimensions);

        System.out.println("Xor problem");
        System.out.println("Output before training:");
        for (TrainingData trainingData : trainingDataset) {
            var prediction = neuralNetwork.predict(trainingData);
            System.out.println(prediction);
        }
        neuralNetwork.train(trainingDataset, 1000000, 0.05);
        System.out.println("Output after training");
        for (TrainingData trainingData : trainingDataset) {
            var prediction = neuralNetwork.predict(trainingData);
            System.out.println(prediction);
        }
    }

    public static TrainingData[] createBlogDataset() {
        TrainingData trainingData = new TrainingData(new double[]{0.05, 0.1}, new double[]{0.01, 0.99});
        return new TrainingData[]{trainingData};
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