/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rbfclassificationproject;

import java.util.Arrays;

/**
 * Class that holds information about images, identical to {@link rbfclassificationproject.Image}, with only
 * difference that the values for data are doubles [0, 1], instead of ints [0,255].
 * 
 * @author Savvas
 */
public class ImageDouble {
    
    private final double[] data;
    private final int label;

    public ImageDouble(double[] data, int label) {
        this.data = Arrays.copyOf(data, data.length);
        this.label = label;
    }

    public ImageDouble(Image img) {
        int[] imgData = img.getData();
        this.data = new double[imgData.length];
        for (int i = 0; i < imgData.length; i++) {
            this.data[i] = imgData[i] / 255.0;
        }
        this.label = img.getLabel();
    }
    
    public ImageDouble(ImageDouble img) {
        this.data = Arrays.copyOf(img.getData(), img.getData().length);
        this.label = img.getLabel();
    }

    public double[] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }
    
    public double getL2Difference(ImageDouble a) {
        int l2 = 0;
        for (int i = 0; i <this. data.length; i++) {
            l2 += Math.pow(this.data[i] - a.data[i],  2);
        }
        return l2;
    }
    
    public int getL1Difference(ImageDouble a) {
        int l1 = 0;
        for (int i = 0; i <this. data.length; i++) {
            l1 += Math.abs(this.data[i] - a.data[i]);
        }
        return l1;
    }
    
    //only compare if the pixels have value 0 or >0.
    public int getL1Difference2(ImageDouble a) {
        int diff = 0;
        for (int i = 0; i <this. data.length; i++) {
            if( (this.data[i] == 0 && a.data[i] > 0) || (this.data[i] > 0 && a.data[i] == 0) ){
                diff++;
            }
        }
        return diff;
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 47 * hash + Arrays.hashCode(this.data);
        hash = 47 * hash + this.label;
        return hash;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final ImageDouble other = (ImageDouble) obj;
        if (this.label != other.label) {
            return false;
        }
        if (!Arrays.equals(this.data, other.data)) {
            return false;
        }
        return true;
    }

    @Override
    public String toString() {
        return "Image{" + "label=" + label + '}';
    }
}
