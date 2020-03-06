/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rbfclassificationproject;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 *
 * @author Savvas
 */
public class RBFClassificationProject {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, Exception {
        
        runRBFProblem(50, 5);
    }

    private static double runRBFProblem(int centersPerClass, int epochs) throws IOException, Exception {
        System.out.println("Running RBF problem for " + centersPerClass + " centers per class and " + epochs + " epochs of training.");
        
        //read the data from the files and load it in ImageDouble objects.
        MNistReader mnist = new MNistReader();
        mnist.loadMNistFiles("train-labels.idx1-ubyte",
                "train-images.idx3-ubyte", "t10k-labels.idx1-ubyte",
                "t10k-images.idx3-ubyte");

        //the images loaded from the files
        List<ImageDouble> trainImages = mnist.getTrainImages();
        List<Image> testImages = mnist.getTestImages();
        mnist = null; //clear the mnist object
        
        //use a sublist of the training data
        //trainImages = new ArrayList<>(trainImages.subList(0, 10000));

        //read a previously created network
//        FileInputStream fis = new FileInputStream("rbfNetwork.ser");
//        ObjectInputStream ois = new ObjectInputStream(fis);
//        RBFNeuralNetwork rbfNetwork = (RBFNeuralNetwork) ois.readObject();
//        fis.close();
//        ois.close();

        //create the rbf network
        long time = System.currentTimeMillis();
        RBFNeuralNetwork rbfNetwork = new RBFNeuralNetwork(784, trainImages,
                centersPerClass);
        System.out.println("Time to calculate rbf centers: " + (System
                .currentTimeMillis() - time) / 1000 + " seconds.");
        
        //save the created network
//        FileOutputStream fos = new FileOutputStream("rbfNetwork.ser");
//        ObjectOutputStream oos = new ObjectOutputStream(fos);
//        oos.writeObject(rbfNetwork);
//        fos.close();
//        oos.close();


        System.out.println("Transforming image objects to vectors of doubles...");
        time = System.currentTimeMillis();

        //the vectors to give to the neural network
        List<List<Double>> trainingSetVectors = new ArrayList<>();
        List<List<Double>> trainingTargets = new LinkedList();
        List<List<Double>> testSet = new ArrayList(testImages.size());
        List<Integer> testTargets = new ArrayList(testImages.size());

        ImageDouble img = null;
        double[] data;
        List<Double> imgData = null;
        List<Double> imgTargets = null;
        //transorm train images to vectors of doubles
        for (int i = 0; i < trainImages.size(); i++) {
            img = trainImages.get(i);
            data = img.getData();
            imgData = new ArrayList<>(data.length);
            for (int j = 0; j < data.length; j++) {
                imgData.add(data[j]);
            }
            imgTargets = new ArrayList<>(10);
            for (int j = 0; j < 10; j++) {
                if (img.getLabel() == j) {
                    imgTargets.add(0.95);
                }
                else {
                    imgTargets.add(0.05);
                }
            }
            trainImages.set(i, null);
            trainingSetVectors.add(imgData);
            trainingTargets.add(imgTargets);
        }
        trainImages = null;

        //transorm test images to vectors of doubles
        for (int i = 0; i < testImages.size(); i++) {
            Image img2 = testImages.get(i);
            int[] data2 = img2.getData();
            List<Double> imgData2 = new ArrayList<>(data2.length);
            for (int j = 0; j < data2.length; j++) {
                imgData2.add(data2[j] / 256.0); //normalize input to [0, 1]
            }
            testSet.add(imgData2);
            testTargets.add(img2.getLabel());
        }
        System.out.println(
                "Loaded " + trainingSetVectors.size() + " vetors for the training set");
        System.out.println(
                "Loaded " + testSet.size() + " vectors for the test set");

        System.out.println(
                "Time to create vectors from images: " + (long) (System
                .currentTimeMillis() - time) / 1000 + " seconds");

        //train the perceptron layer with the training data
        time = System.currentTimeMillis();
        double percentage = rbfNetwork.trainPerceptronLayer(trainingSetVectors,
                trainingTargets, epochs,
                testSet,
                testTargets);
        time = (System.currentTimeMillis() - time)/1000;

        System.out.println(
                "RBF Network with " + centersPerClass + " centers per class, after " + epochs + " epochs of training, training time " + time + " seconds. Best Result: " + percentage * 100 + "%100");
        
        //save the trained network
        FileOutputStream fos = new FileOutputStream("rbfNetwork-c" + centersPerClass + "-e" +epochs + ".ser");
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(rbfNetwork);
        fos.close();
        oos.close();
        
        return percentage;
    }

}
