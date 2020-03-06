/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rbfclassificationproject;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.time.Clock;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

/**
 *
 * @author Savvas
 */
public class MLPNeuralNetwork implements Serializable{
    
    private List<List<PerceptronNeuron>> layers;

    /**
     *
     * @param inputSize the size of the input vectors
     * @param structure the first number is the number N of layers (including the output layer) and then N numbers follow, with the number of neurons for each layer.
     * @throws Exception
     */
    public MLPNeuralNetwork(int inputSize, String structure) throws Exception {
        
        this.layers = new ArrayList<>();
        
        String[] parameters = structure.split(" ");
        int numOfLayers = Integer.parseInt(parameters[0]);
        
        if (parameters.length - 1 < numOfLayers) { //check that there are parameters for each layer
            throw new Exception();
        }
        
        int previousLayer = inputSize;
        
        for (int i = 1; i < numOfLayers+1; i++) {
            int layerSize = Integer.parseInt(parameters[i]);
            List<PerceptronNeuron> layer = new ArrayList<>(layerSize);
            
            for (int j = 0; j < layerSize; j++) {
                layer.add(new PerceptronNeuron(previousLayer));
            }
            layers.add(layer);
            previousLayer = layerSize;
        }
    }
    
    /**
     *
     * @param input the input for which to calculate the output
     * @return the value of each of the output neurons
     */
    public List<Double> calculateOutput(List<Double> input) {
        
        List<Double> outputs = null;
        
        for (int i = 0; i < layers.size(); i++) {
            List<PerceptronNeuron> layer = layers.get(i);            
            outputs = new ArrayList<>(layer.size());
            for (int j = 0; j < layer.size(); j++) {
                PerceptronNeuron neuron = layer.get(j);
                outputs.add(neuron.output(input));
            }
            input = new ArrayList<>(outputs);
        }
        return outputs;
    }
    
    /**
     *
     * @param trainSet the vectors of the training set
     * @param targets the targets for each of the vectors of the training set
     * @param epochs the number of epochs to run
     * @param testSet the vectors of the test set, if no test set exists, set it to null
     * @param testTargets the labels for the images of the test set. If no test set exists, set it to null
     * @throws FileNotFoundException
     * @throws IOException
     */
    public double backPropagate(List<List<Double>> trainSet, List<List<Double>> targets, int epochs, List<List<Double>> testSet, List<Integer> testTargets) throws FileNotFoundException, IOException {
        
        List<Double> outputs = null;
        
        //this will be a stack that will hold the outputs of the hidden layers and they will be popped for the back propagation.
        LinkedList<List<Double>> intermediateOutputs = null;        
        
        //create a list with all the indexes, so that shuffling is easy.
        List<Integer> indexArray = new ArrayList<>(trainSet.size());
        for (int i = 0; i < trainSet.size(); i++) {
            indexArray.add(i);
        }
        
        int successful = 0;
        for (int c = 0; c < epochs; c++) { //number of iterations
            
            //test the test set and the training set before each iteration
            if(testSet != null && testTargets != null) {
                System.out.println("Testing test set...");
                int thisSuccessful = runTestSet(testSet, testTargets);
                if(thisSuccessful > successful) {
                    successful = thisSuccessful;
                }
                System.out.println("Result on test set after " + c + " epochs: " + thisSuccessful + "/" + testSet.size());
                
            }            
            System.out.println("Testing training set...");
            int thisSuccessful = runTrainSet(trainSet, targets);
            System.out.println("Result on training set after " + c + " epochs: " + thisSuccessful + "/" + trainSet.size());
            
            System.out.println("Running epoch " + c + "...");
                
            Collections.shuffle(indexArray);
            
            long time = System.currentTimeMillis();
            for (int t = 0; t < trainSet.size(); t++) { //run for each input vector
                if(t % 10000 == 0) {
                    System.out.println(t + " samples");
                }
                List<Double> input = trainSet.get(indexArray.get(t)); //one input vector
                
                //store here the outputs of all the hidden layers (including the first and last layers)
                intermediateOutputs = new LinkedList<>();
                intermediateOutputs.addFirst(new ArrayList<>(input));
                
                //the forward pass
                for (int i = 0; i < layers.size(); i++) {
                    List<PerceptronNeuron> layer = layers.get(i);
                    outputs = new ArrayList<>(layer.size());
                    for (int j = 0; j < layer.size(); j++) {
                        PerceptronNeuron neuron = layer.get(j);
                        outputs.add(neuron.output(input));
                    }
                    intermediateOutputs.addFirst(new ArrayList<>(outputs));
                    input = new ArrayList<>(outputs);
                }

                //the last output is not needed.
                intermediateOutputs.pollFirst();

                //the backward pass
                List<Double> higherDeltas = null;
                for (int j = layers.size()-1; j >=0; j--) {
                    List<PerceptronNeuron> layer = layers.get(j);
                    List<Double> previousInput = intermediateOutputs.pollFirst();

                    List<Double> layerDeltas = new ArrayList<>(layer.size()); //array with deltas or each of the neurons of the current

                    for (int k = 0; k < layer.size(); k++) {
                        if(j==layers.size()-1) { //if we are in the output layer, then adjust weights with the target.
                            double delta = layer.get(k).adjustWeights(previousInput, targets.get(indexArray.get(t)).get(k));
                            layerDeltas.add(delta);
                        }
                        else { //else, adjust weights with the deltas of the higher layer
                            PerceptronNeuron neuron = layer.get(k);
                            double neuronError = 0;
                            List<PerceptronNeuron> higherLayer = layers.get(j+1); //get the neurons of the next layer to see their weights
                            for (int l = 0; l < higherDeltas.size(); l++) {
                                neuronError += higherDeltas.get(l)*higherLayer.get(l).getWeight(k); // dj = dk*wkj
                            }
                            double delta = neuron.adjustWeightsHidden(previousInput, neuronError);
                            layerDeltas.add(delta);
                        }
                    }
                    higherDeltas = new ArrayList<>(layerDeltas); //store the current deltas to give them to the lower layer.
                }
            }
            System.out.println("Time for epoch: " + (System.currentTimeMillis() - time) + " ms");
        }
        //test the test set and the training set after all epochs
        if(testSet != null && testTargets != null) {
            System.out.println("Testing test set...");
            int thisSuccessful = runTestSet(testSet, testTargets);
            if(thisSuccessful > successful) {
                successful = thisSuccessful;
            }
            System.out.println("Result on test set after " + epochs + " epochs: " + thisSuccessful + "/" + testSet.size());
        }            
        System.out.println("Testing training set...");
        int thisSuccessful = runTrainSet(trainSet, targets);
        System.out.println("Result on training set after " + epochs + " epochs: " + thisSuccessful + "/" + trainSet.size());
        
        return 1.0 * successful / testSet.size();
    }
 
    /**
     *
     * @param testSet the vectors of the test set
     * @param testTargets the right label for each of the vectors of the test set
     * @return
     */
    public int runTestSet(List<List<Double>> testSet, List<Integer> testTargets) {
        int counter = 0;
        for (int i = 0; i < testSet.size(); i++) {
            List<Double> output = calculateOutput(testSet.get(i));
            double max = 0;
            int label = -1;
            for (int j = 0; j < output.size(); j++) {
                if (output.get(j) > max) {
                    max = output.get(j);
                    label = j;
                }
            }
            if(label == testTargets.get(i)) {
                counter++;
            }
        }
        return counter;
    }

    public int runTrainSet(List<List<Double>> trainSet,
                            List<List<Double>> targets) {
        int counter = 0;
        for (int i = 0; i < trainSet.size(); i++) {
            List<Double> output = calculateOutput(trainSet.get(i));
            double max = 0;
            int label = -1;
            for (int j = 0; j < output.size(); j++) {
                if (output.get(j) > max) {
                    max = output.get(j);
                    label = j;
                }
            }
            List<Double> sampleTargets = targets.get(i);
            double rightMax = 0.0;
            int rightLabel = -1;
            for (int j = 0; j < sampleTargets.size(); j++) {
                if(sampleTargets.get(j) > rightMax) {
                    rightMax = sampleTargets.get(j);
                    rightLabel = j;
                }
            }
            if(label == rightLabel) {
                counter++;
            }
        }
        return counter;
    }
}
