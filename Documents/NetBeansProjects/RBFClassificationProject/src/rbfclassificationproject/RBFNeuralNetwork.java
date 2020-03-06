/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rbfclassificationproject;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import javax.imageio.ImageIO;

/**
 *
 * @author Savvas
 */
public class RBFNeuralNetwork implements Serializable{

    private RBFLayer rbfLayer;
    private MLPNeuralNetwork perceptronLayer;
    private static int fileCounter = 0;

    /**
     * Creates an RBF Neural Network. Calculates the centers of the rbf neurons
     * and creates a layer with rbf neurons and a layer with perceptron neurons.
     * 
     * @param inputSize the length of the input vectors
     * @param trainingSet the training images
     * @param centersPerClass the centers for each class
     * @throws IOException
     * @throws Exception 
     */
    public RBFNeuralNetwork(int inputSize, List<ImageDouble> trainingSet, int centersPerClass) throws IOException, Exception {

        System.out.println("Calculating rbf centers...");
        List<List<Double>> allCenters = calculateRbfCenters(trainingSet, centersPerClass);

        List<RBFNeuron> rbfCenters = new ArrayList<>(allCenters.size());
        for (List<Double> center : allCenters) {
            rbfCenters.add(new RBFNeuron(center, 1));
        }

        //create the layers of the network
        this.rbfLayer = new RBFLayer(rbfCenters);
        this.perceptronLayer = new MLPNeuralNetwork(rbfCenters.size(), "1 10");
    }

    /**
     * Calculates the output of the network for a given input vector
     * 
     * @param input the input vector
     * @return the values of the output neurons
     */
    public List<Double> output(List<Double> input) {
        List<Double> rbfLayerOutput = this.rbfLayer.output(input);
        return this.perceptronLayer.calculateOutput(rbfLayerOutput);
    }

    /**
     * Trains the Perceptron Layer of the network with the Back Propagation algorithm.
     * 
     * @param trainSet the training set
     * @param targets the targets for each output neuron for each training sample
     * @param epochs the number of epochs to run
     * @param testSet the test set
     * @param testTargets the target labels for the test set
     * @return the best success rate for the test set
     * @throws IOException 
     */
    public double trainPerceptronLayer(List<List<Double>> trainSet,
                                     List<List<Double>> targets, int epochs,
                                     List<List<Double>> testSet,
                                     List<Integer> testTargets) throws IOException {
        
        //for each input vector, calculate the output of the rbf layer, and the perceptron layer will use this as input.
        System.out.println("Calculating output of the rbf layer for each training sample...");
        List<List<Double>> rbfLayerOutputs = new ArrayList<>(trainSet.size());
        for (int i = 0; i < trainSet.size(); i++) {
            List<Double> trainVector = trainSet.get(i);
            trainSet.set(i, null);
            rbfLayerOutputs.add(this.rbfLayer.output(trainVector));
            if(i%10000 == 0) {
                System.out.println(i);
            }
        }
        trainSet = null;
        
        //calculate the outputs of the rbf layer for the test set.
        List<List<Double>> rbfLayerTestOutputs = new ArrayList<>(testSet.size());
        for (int i = 0; i < testSet.size(); i++) {
            List<Double> testVector = testSet.get(i);
            testSet.set(i, null);
            rbfLayerTestOutputs.add(this.rbfLayer.output(testVector));
        }
        testSet = null;
        
        System.out.println("Running back propagation...");
        double percentage = perceptronLayer.backPropagate(rbfLayerOutputs, targets, epochs, rbfLayerTestOutputs,
                testTargets);
        
        return percentage;
    }

    /**
     * Calculates all the centers needed for the rbf layer
     * 
     * @param trainingSet all the images
     * @param kPerClass the number of centers to create for each class
     * @return the list of all the centers
     */
    private List<List<Double>> calculateRbfCenters(List<ImageDouble> trainingSet,
                                                   int kPerClass) {

        //create a list with 10 lists, one for each label. Each list will have all the training images of one label.
        List<List<ImageDouble>> imagesByLabel = new ArrayList<>(10);
        for (int i = 0; i < 10; i++) {
            imagesByLabel.add(new LinkedList()); //Initialize each list
        }

        //add each image to the appropriate list
        for (ImageDouble img : trainingSet) {
            imagesByLabel.get(img.getLabel()).add(img);
        }

        int k = kPerClass; //the number of rbf neurons for each class.

        List<List<Double>> allCenters = new ArrayList<>(k * 10);

        //for each cluster run the k-means algorithm
        for (int i = 0; i < 10; i++) {
            List<List<Double>> clusterCenters = runKMeans(k, imagesByLabel
                    .get(i));
            allCenters.addAll(clusterCenters);
        }

        return allCenters;
    }

    /**
     * Runs the kMeans algorithm to calculate the centers for a set of images
     * 
     * @param k the number of centers to create
     * @param clusterImages the set of images
     * @return the list of centers
     */
    private List<List<Double>> runKMeans(int k, List<ImageDouble> clusterImages) {
        //rand for choosing the initial centers
        Random rand = new Random(6);

        //the centers that will be created for each cluster (label)
        List<List<Double>> clusterCenters = new ArrayList<>(k);

        //initialize the centers as some random images
        for (int i = 0; i < k; i++) {
            ImageDouble randImg = clusterImages.get(rand.nextInt(
                    clusterImages.size()));
            double[] randImgData = randImg.getData();
            List<Double> center = new ArrayList<>(randImgData.length);
            for (int l = 0; l < randImgData.length; l++) {
                center.add(randImgData[l]);
            }
            clusterCenters.add(center);
        }

        double maxErrorAllowed = Math.exp(-3);
        double error = Double.MAX_VALUE;

        while (error > maxErrorAllowed) {
            //the images of the cluster, clustered by the center that is nearest to them
            List<List<ImageDouble>> imagesByCenter = new ArrayList<>(k);
            for (int i = 0; i < k; i++) {
                imagesByCenter.add(new ArrayList<>());
            }

            //find in which center each image belongs
            for (ImageDouble img : clusterImages) {
                double minDistance = Double.MAX_VALUE;
                int chosenCenter = -1;
                for (int i = 0; i < clusterCenters.size(); i++) {
                    double distance = distanceFromCenter(clusterCenters.get(
                            i), img);
                    if (distance < minDistance) {
                        minDistance = distance;
                        chosenCenter = i;
                    }
                }
                imagesByCenter.get(chosenCenter).add(img); //add the image to the images of this center
            }

            //calculate the new centers
            List<List<Double>> newClusterCenters = new ArrayList<>(k);
            for (List<ImageDouble> imagesOfCenter : imagesByCenter) {
                
                //initialize each new center
                List<Double> newCenter = new ArrayList(784);
                for (int i = 0; i < 784; i++) {
                    newCenter.add(0.0);
                }
                if(imagesOfCenter.size() != 0) { //if this center has no images, don't to the division, just let it be a center with all values 0.
                    //calculate the average values for the new centers of the clusters
                    for (ImageDouble img : imagesOfCenter) {
                        double[] data = img.getData();
                        for (int j = 0; j < data.length; j++) {
                            newCenter.set(j, newCenter.get(j) + data[j]);
                        }
                    }
                    for (int i = 0; i < newCenter.size(); i++) {
                        newCenter.set(i, newCenter.get(i) / imagesOfCenter
                                .size());
                    }
                }
                newClusterCenters.add(newCenter);
            }

            //calculate the change of the centers
            error = 0;
            for (int i = 0; i < clusterCenters.size(); i++) {
                error += distanceOfCenters(clusterCenters.get(i),
                        newClusterCenters.get(i));
            }
            error /= clusterCenters.size();

            clusterCenters = new ArrayList<>(newClusterCenters);
        }
        return clusterCenters;
    }

    /**
     * Calculates the distance of an ImageDouble from a center
     * 
     * @param center the center
     * @param img the image
     * @return the distance value
     */
    private double distanceFromCenter(List<Double> center, ImageDouble img) {
        double sum = 0;
        double[] data = img.getData();
        for (int i = 0; i < center.size(); i++) {
            sum += Math.pow(center.get(i) - data[i], 2);
        }
        return sum;
    }

    /**
     * Calculates the distance between two centers
     * 
     * @param c1 the first center
     * @param c2 the second center
     * @return  the distance value
     */
    private double distanceOfCenters(List<Double> c1, List<Double> c2) {

        double sum = 0;
        for (int i = 0; i < c1.size(); i++) {
            sum += Math.pow(c1.get(i) - c2.get(i), 2);
        }
        return Math.sqrt(sum);
    }

}
