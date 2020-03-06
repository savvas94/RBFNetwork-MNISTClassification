package rbfclassificationproject;
   
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import javafx.util.Pair;
import javax.imageio.ImageIO;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/11/11
 * Time: 10:07 AM
 */
public class MNistReader {

    private List<ImageDouble> trainImages;
    private List<Image> testImages;

    /** the following constants are defined as per the values described at http://yann.lecun.com/exdb/mnist/ **/

    private static final int MAGIC_OFFSET = 0;
    private static final int OFFSET_SIZE = 4; //in bytes

    private static final int LABEL_MAGIC = 2049;
    private static final int IMAGE_MAGIC = 2051;

    private static final int NUMBER_ITEMS_OFFSET = 4;
    private static final int ITEMS_SIZE = 4;

    private static final int NUMBER_OF_ROWS_OFFSET = 8;
    private static final int ROWS_SIZE = 4;

    /**
     *
     */
    public static final int ROWS = 28;

    private static final int NUMBER_OF_COLUMNS_OFFSET = 12;
    private static final int COLUMNS_SIZE = 4;

    /**
     *
     */
    public static final int COLUMNS = 28;

    private static final int IMAGE_OFFSET = 16;
    private static final int IMAGE_SIZE = ROWS * COLUMNS;

    /**
     *
     */
    public MNistReader() {
        trainImages = new ArrayList<>();
        testImages = new ArrayList<>();
    }

    /**
     *
     * @param labelTrainFileName
     * @param imageTrainFileName
     * @throws IOException
     */
    public void loadMNistFiles(String labelTrainFileName, String imageTrainFileName, String labelTestFileName, String imageTestFileName) throws IOException {
        
        ByteArrayOutputStream labelTrainBuffer = new ByteArrayOutputStream();
        ByteArrayOutputStream imageTrainBuffer = new ByteArrayOutputStream();
        ByteArrayOutputStream labelTestBuffer = new ByteArrayOutputStream();
        ByteArrayOutputStream imageTestBuffer = new ByteArrayOutputStream();
        
        InputStream labelTrainInputStream = this.getClass().getResourceAsStream(labelTrainFileName);
        InputStream imageTrainInputStream = this.getClass().getResourceAsStream(imageTrainFileName);
        InputStream labelTestInputStream = this.getClass().getResourceAsStream(labelTestFileName);
        InputStream imageTestInputStream = this.getClass().getResourceAsStream(imageTestFileName);
        
        labelTrainInputStream = new DataInputStream(new FileInputStream(new File(labelTrainFileName)));
        imageTrainInputStream = new DataInputStream(new FileInputStream(new File(imageTrainFileName)));
        labelTestInputStream = new DataInputStream(new FileInputStream(new File(labelTestFileName)));
        imageTestInputStream = new DataInputStream(new FileInputStream(new File(imageTestFileName)));

        int read;
        byte[] buffer = new byte[16384];

        while((read = labelTrainInputStream.read(buffer, 0, buffer.length)) != -1) {
           labelTrainBuffer.write(buffer, 0, read);
        }

        labelTrainBuffer.flush();

        while((read = imageTrainInputStream.read(buffer, 0, buffer.length)) != -1) {
            imageTrainBuffer.write(buffer, 0, read);
        }

        imageTrainBuffer.flush();
        
        while((read = labelTestInputStream.read(buffer, 0, buffer.length)) != -1) {
            labelTestBuffer.write(buffer, 0, read);
        }

        labelTestBuffer.flush();
        
        while((read = imageTestInputStream.read(buffer, 0, buffer.length)) != -1) {
            imageTestBuffer.write(buffer, 0, read);
        }

        imageTestBuffer.flush();

        byte[] labelTrainBytes = labelTrainBuffer.toByteArray();
        byte[] imageTrainBytes = imageTrainBuffer.toByteArray();
        byte[] labelTestBytes = labelTestBuffer.toByteArray();
        byte[] imageTestBytes = imageTestBuffer.toByteArray();

        byte[] labelMagic = Arrays.copyOfRange(labelTrainBytes, 0, OFFSET_SIZE);
        byte[] imageMagic = Arrays.copyOfRange(imageTrainBytes, 0, OFFSET_SIZE);
        
        if(ByteBuffer.wrap(labelMagic).getInt() != LABEL_MAGIC)  {
            throw new IOException("Bad magic number in label file!");
        }
        
        if(ByteBuffer.wrap(imageMagic).getInt() != IMAGE_MAGIC) {
            throw new IOException("Bad magic number in image file!");
        }

        int numberOfLabelsTrain = ByteBuffer.wrap(Arrays.copyOfRange(labelTrainBytes, NUMBER_ITEMS_OFFSET, NUMBER_ITEMS_OFFSET + ITEMS_SIZE)).getInt();
        int numberOfImagesTrain = ByteBuffer.wrap(Arrays.copyOfRange(imageTrainBytes, NUMBER_ITEMS_OFFSET, NUMBER_ITEMS_OFFSET + ITEMS_SIZE)).getInt();
        int numberOfLabelsTest = ByteBuffer.wrap(Arrays.copyOfRange(labelTestBytes, NUMBER_ITEMS_OFFSET, NUMBER_ITEMS_OFFSET + ITEMS_SIZE)).getInt();
        int numberOfImagesTest = ByteBuffer.wrap(Arrays.copyOfRange(imageTestBytes, NUMBER_ITEMS_OFFSET, NUMBER_ITEMS_OFFSET + ITEMS_SIZE)).getInt();

        if(numberOfImagesTrain != numberOfLabelsTrain) {
            throw new IOException("The number of labels and images do not match!");
        }

        int numRows = ByteBuffer.wrap(Arrays.copyOfRange(imageTrainBytes, NUMBER_OF_ROWS_OFFSET, NUMBER_OF_ROWS_OFFSET + ROWS_SIZE)).getInt();
        int numCols = ByteBuffer.wrap(Arrays.copyOfRange(imageTrainBytes, NUMBER_OF_COLUMNS_OFFSET, NUMBER_OF_COLUMNS_OFFSET + COLUMNS_SIZE)).getInt();

        if(numRows != ROWS && numRows != COLUMNS) {
            throw new IOException("Bad image. Rows and columns do not equal " + ROWS + "x" + COLUMNS);
        }
        
        System.out.println("Reading training set...");
        for(int i = 0; i < numberOfLabelsTrain; i++) {
            int label = labelTrainBytes[OFFSET_SIZE + ITEMS_SIZE + i];
            byte[] imageData = Arrays.copyOfRange(imageTrainBytes, (i * IMAGE_SIZE) + IMAGE_OFFSET, (i * IMAGE_SIZE) + IMAGE_OFFSET + IMAGE_SIZE);
            
            double[] imageDataDouble = new double[imageData.length];
            for (int j = 0; j < imageData.length ; j++) {
                byte b = (byte) imageData[j];
                int pixel = b & 0xFF;
                imageDataDouble[j] = pixel / 255.0;
            }
            trainImages.add(new ImageDouble(imageDataDouble, label));
        }
        
        System.out.println("Reading test set...");
        for(int i = 0; i < numberOfLabelsTest; i++) {
            int label = labelTestBytes[OFFSET_SIZE + ITEMS_SIZE + i];
            byte[] imageData = Arrays.copyOfRange(imageTestBytes, (i * IMAGE_SIZE) + IMAGE_OFFSET, (i * IMAGE_SIZE) + IMAGE_OFFSET + IMAGE_SIZE);
            
            
            int[] imageDataInt = new int[imageData.length];
            for (int j = 0; j < imageData.length ; j++) {
                byte b = (byte) imageData[j];
                imageDataInt[j] = b & 0xFF;
            }
            testImages.add(new Image(imageDataInt, label));
        }
        
        System.out.println("done");

        return;
    }

    public List<ImageDouble> getTrainImages() {
        return trainImages;
    }

    public List<Image> getTestImages() {
        return testImages;
    }
}