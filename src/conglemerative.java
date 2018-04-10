/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author yang
 */

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class conglemerative {
    
    public static int classIndex;
    public static ArrayList<Integer> multiArray = new ArrayList<>();
    public static ArrayList<ArrayList<Integer>> multiArrayList = new ArrayList<>();
    public static ArrayList<Double> multiAccuracyList = new ArrayList<>();
    public static int count_no=0, count_yes=0, count_multi=0;
    public static double noArrayAccuracy;
    public static double yesArrayAccuracy;
    
    // preprocess the dataset, divide the participants into 3 clusters (ALL YES, ALL NO, and multi-node)
    public static void preprocessing() throws Exception{
        // to store user clusters
        ArrayList<Integer> noArray = new ArrayList<>();
        ArrayList<Integer> yesArray = new ArrayList<>();
        
        // ALL NO and ALL YES
        String t1="N0 [label=\"0";
        String t2="N0 [label=\"1";
        
        DataSource source = new DataSource("docs/samsung.arff");
        Instances allusers=source.getDataSet();
        classIndex=allusers.numAttributes()-1;
        allusers.setClassIndex(classIndex);
        //System.out.println(allusers.numInstances());

        for (int i = 0; i < allusers.numInstances(); i = i + 12) {
            int userID = (int)allusers.instance(i).value(0);
            Instances singleUserInstances = new Instances(allusers, i, 12);
            FilteredClassifier cls = wekaFunctions.train(singleUserInstances, classIndex); // train
//            double accuracy =  wekaFunctions.eval(cls, singleUserInstances,singleUserInstances); // eval
//            System.out.println("User #:" +userID);
//            System.out.println("Classifier :" +fc);
//            System.out.println("Accuracy :" +accuracy);

            //cls.graph() store this in a string and use string functions to parse it; put in if conditions to determine the clusters
            if(cls.graph().contains(t1)){
                count_no++;
                noArray.add(userID);
            }
            else if(cls.graph().contains(t2)){
                count_yes++;
                yesArray.add(userID);
            }
            else {
                count_multi++;
                ArrayList<Integer> currentUser = new ArrayList<>();
                currentUser.add(userID);
                multiArrayList.add(currentUser);
                //multiAccuracyList.add(wekaFunctions.eval(cls, singleUserInstances,singleUserInstances));               
            } 
        }
        System.out.println("Single 'NO' node trees: " + count_no);
        System.out.println("Single 'YES' node trees: " + count_yes);
        System.out.println("Multi-node trees: " + count_multi);         

        noArrayAccuracy = wekaFunctions.trainSelfEval(noArray);
        yesArrayAccuracy = wekaFunctions.trainSelfEval(yesArray);
        System.out.println("noArray's accuracy is: " + noArrayAccuracy);
        System.out.println("yesArray's accuracy is: " + yesArrayAccuracy);
    }  
        
    public static void shuffleMultiArray() throws Exception {
        Collections.shuffle(multiArrayList);
        for (Iterator<ArrayList<Integer>> iterator = multiArrayList.iterator(); iterator.hasNext();) {
            ArrayList<Integer> next = iterator.next();
            //System.out.println( next + ":" + wekaFunctions.trainSelfEval(next) );
            //multiArray.add(next.get(0));
            multiAccuracyList.add( wekaFunctions.trainSelfEval(next) );
        }

        System.out.println("multiAccuracyList's size is: " + multiAccuracyList.size());
   
    }
    
    /************************************************************************/
    /*Merge part of conglemerative, Strtege 1: pick smallest reduction first*/
    /**
     * @param ROUNDS*
     * @throws java.lang.Exception*********************************************************************/
    public static void merge1(int ROUNDS) throws Exception{
        
        ArrayList<ArrayList<Integer>> multiArrayListCopy = new ArrayList<ArrayList<Integer>>(multiArrayList);
        
        for (int round = 0; round < ROUNDS; round++) {
            double maxPairAccuracy = 0;
            double maxAccuracyDiff = -10000;
            double accuracySum = 0;
            int maxIndexLeftHand = 0;
            int maxIndexRightHand = 0;
            int index=1;
            
            for(int i = 0; i < multiArrayListCopy.size();i++){
                //System.out.println("element size: " + multiArrayListCopy.get(i).size());
                //System.out.println("element accuracy: " + multiAccuracyList.get(i));
                accuracySum += multiArrayListCopy.get(i).size()*multiAccuracyList.get(i);
            }
            System.out.println("Overall Accuracy: " + accuracySum);
            accuracySum = 0;
            
            System.out.println("Round: " + (round+1) );
            for (int i = 0; i < multiArrayListCopy.size(); i++) {
                for (int j = i+1; j < multiArrayListCopy.size(); j++) {
                    ArrayList<Integer> currentPair = new ArrayList<>(multiArrayListCopy.get(i));
                    currentPair.addAll(multiArrayListCopy.get(j));
                    
                    double currentPairAccuracy = wekaFunctions.trainSelfEval
                        (currentPair);
                    double pre = multiAccuracyList.get(i)*multiArrayListCopy.get(i).size()
                            +multiAccuracyList.get(j)*multiArrayListCopy.get(j).size();
                    
                    double after = currentPairAccuracy*currentPair.size();
                    double currentAccuracyDiff = after - pre;
                                      
//                    System.out.print((index++) + ": ");
//                    printRow(multiArrayListCopy.get(i));  
//                    printRow(multiArrayListCopy.get(j));  
//                    System.out.print("Diff: " + currentAccuracyDiff);                   
//                    System.out.println(" Max: " + maxAccuracyDiff);

                    if (currentAccuracyDiff > maxAccuracyDiff) {
                        maxAccuracyDiff = currentAccuracyDiff;
                        maxIndexLeftHand = i;
                        maxIndexRightHand = j;
                        maxPairAccuracy = currentPairAccuracy;
                    }
                    //System.out.println("Max: " + maxAccuracyDiff); 
                }
            }
           
            ArrayList<Integer> leftHand = multiArrayListCopy.get(maxIndexLeftHand);
            //System.out.println("LefthandAcc: " + multiAccuracyList.get(maxIndexLeftHand));
            leftHand.addAll(multiArrayListCopy.get(maxIndexRightHand));
            multiAccuracyList.set(maxIndexLeftHand, maxPairAccuracy);
            //System.out.println("LefthandAcc: " + multiAccuracyList.get(maxIndexLeftHand));
            //System.out.println("Lefthand: " + leftHand);
            multiArrayListCopy.remove(maxIndexRightHand);
            multiAccuracyList.remove(maxIndexRightHand);

            System.out.println("---------------------------------------");
            System.out.println("ROUNDS end: ");
            //System.out.println(multiArrayListCopy.size());
            //System.out.println(multiAccuracyList.size());
            for(int i = 0; i < multiArrayListCopy.size();i++){
//                if (ROUNDS == 434) {
//                    
//                }
                printRow(multiArrayListCopy.get(i));
                //accuracySum += multiArrayListCopy.get(i).size()*multiAccuracyList.get(i);
            }
            System.out.println();
            System.out.println(maxIndexLeftHand + "," + maxIndexRightHand + ": " + maxAccuracyDiff);
            
                //System.out.println("element size: " + multiArrayListCopy.get(i).size());
                //System.out.println("element accuracy: " + multiAccuracyList.get(i));
            
            
            for(int i = 0; i < multiArrayListCopy.size();i++){
                //System.out.println("element size: " + multiArrayListCopy.get(i).size());
                //System.out.println("element accuracy: " + multiAccuracyList.get(i));
                double sum = multiArrayListCopy.get(i).size()*multiAccuracyList.get(i);
                accuracySum += sum;
                //System.out.println(accuracySum);
            }
            System.out.println("Overall Accuracy: " + accuracySum);            
            
            System.out.println("======================================");

        }       
        
    }       

    /*************************************************************/
    /*Merge part of conglemerative, Strtege 2: pick highest first*/
    /**
     * @param ROUNDS
     * @throws java.lang.Exception***********************************************************/
    public static void merge2(int ROUNDS) throws Exception{
        
        double maxPairAccuracy = 0;
        int maxIndexLeftHand = 0;
        int maxIndexRightHand = 0;
        int index=1;
        
        ArrayList<ArrayList<Integer>> multiArrayListCopy = new ArrayList<ArrayList<Integer>>(multiArrayList);
        
        for (int round = 0; round < ROUNDS; round++) {            
            System.out.println("Round: " + (round+1) );
            System.out.println(multiArrayListCopy.size());
            System.out.println(multiAccuracyList.size());
            
            for (int i = 0; i < multiArrayListCopy.size(); i++) {
                for (int j = i+1; j < multiArrayListCopy.size(); j++) {
//            int i = 422;
//            int j = 423;
                    ArrayList<Integer> currentPair = new ArrayList<>(multiArrayListCopy.get(i));
                    currentPair.addAll(multiArrayListCopy.get(j));
                    
                    arffFunctions.generateArff(currentPair, "docs/samsung_header.txt", "currentPair.arff");
                    DataSource sourceCurrentPair = new DataSource("docs/currentPair.arff");
                    Instances dataCurrentPair = sourceCurrentPair.getDataSet();
                    double currentPairAccuracy = wekaFunctions.trainAndEval
                        (dataCurrentPair, dataCurrentPair, classIndex);
//                    System.out.print((index++) + ": ");
//                    for(Integer temp:multiArrayListCopy.get(i)){  
//                        System.out.print(temp);
//                        System.out.print(" ");
//                    }
//                    System.out.print(",");
//                    for(Integer temp:multiArrayListCopy.get(j)){  
//                        System.out.print(temp);
//                        System.out.print(" ");
//                    }
//                    System.out.println(":" + currentPairAccuracy);
                    if (currentPairAccuracy == 100.0) {
                        maxPairAccuracy = currentPairAccuracy;
                        maxIndexLeftHand = i;
                        maxIndexRightHand = j;
                        break;
                    }
                    else if (currentPairAccuracy > maxPairAccuracy) {
                        maxPairAccuracy = currentPairAccuracy;
                        maxIndexLeftHand = i;
                        maxIndexRightHand = j;
                    }
                }
                if (maxPairAccuracy == 100.0) {
                    break;                                                    
                }
            }

            System.out.println(maxIndexLeftHand + "," + maxIndexRightHand + ": " + maxPairAccuracy);
            ArrayList<Integer> leftHand = multiArrayListCopy.get(maxIndexLeftHand);
            //System.out.println("LefthandAcc: " + multiAccuracyList.get(maxIndexLeftHand));
            leftHand.addAll(multiArrayListCopy.get(maxIndexRightHand));
            multiAccuracyList.set(maxIndexLeftHand, maxPairAccuracy);
            //System.out.println("LefthandAcc: " + multiAccuracyList.get(maxIndexLeftHand));
            //System.out.println("Lefthand: " + leftHand);
            multiArrayListCopy.remove(maxIndexRightHand);
            multiAccuracyList.remove(maxIndexRightHand);
            
            System.out.println("---------------------------------------");
            System.out.println("ROUNDS end: ");
            for (ArrayList<Integer> next : multiArrayListCopy) {
                printRow(next);
            }
            System.out.println("======================================");
            maxPairAccuracy = 0;
            maxIndexLeftHand = 0;
            maxIndexRightHand = 0;
            index = 1;
        }       
    }

    public static void printRow(ArrayList<Integer> row){
        for(Integer temp:row){  
            System.out.print(temp);
            System.out.print(" ");
        }
        System.out.print(", ");
    }
    
    public static void printRow(int[] row) {
        for (int i : row) {
            System.out.print(i);
            System.out.print("\t");
        }
        System.out.println();
    }

    public static void printRow(float[] row) {
        for (float i : row) {
            System.out.print(i);
            System.out.print("\t");
        }
        System.out.println();
    }
    
    public static void test() throws IOException, Exception{
        
        int l1 = constantVar.cluster1.length;
        int l2 = constantVar.cluster2.length;
        int l3 = constantVar.cluster3.length;
//        int l4 = constantVar.cluster4.length;
//        int l5 = constantVar.cluster5.length;
        
        arffFunctions.generateArff(constantVar.cluster1, "docs/samsung_header.txt", "model1.arff");
        arffFunctions.generateArff(constantVar.cluster2, "docs/samsung_header.txt", "model2.arff");
        arffFunctions.generateArff(constantVar.cluster3, "docs/samsung_header.txt", "model3.arff");
//        arffFunctions.generateArff(constantVar.cluster4, "docs/samsung_header.txt", "model4.arff");
//        arffFunctions.generateArff(constantVar.cluster5, "docs/samsung_header.txt", "model5.arff");

        DataSource source1 = new DataSource("docs/model1.arff");
        DataSource source2 = new DataSource("docs/model2.arff");
        DataSource source3 = new DataSource("docs/model3.arff");
//        DataSource source4 = new DataSource("docs/model4.arff");
//        DataSource source5 = new DataSource("docs/model5.arff");

        Instances data1 = source1.getDataSet();
        Instances data2 = source2.getDataSet();
        Instances data3 = source3.getDataSet();
//        Instances data4 = source4.getDataSet();
//        Instances data5 = source5.getDataSet();

        data1.setClassIndex(classIndex);
        data2.setClassIndex(classIndex);
        data3.setClassIndex(classIndex);
//        data4.setClassIndex(classIndex);
//        data5.setClassIndex(classIndex);

        FilteredClassifier fc1 = wekaFunctions.train(data1, classIndex);
        FilteredClassifier fc2 = wekaFunctions.train(data2, classIndex);
        FilteredClassifier fc3 = wekaFunctions.train(data3, classIndex);
//        FilteredClassifier fc4 = wekaFunctions.train(data4, classIndex);
//        FilteredClassifier fc5 = wekaFunctions.train(data5, classIndex);

        double Array1Accuracy = wekaFunctions.eval(fc1, data1, data1);
        double Array2Accuracy = wekaFunctions.eval(fc2, data2, data2);
        double Array3Accuracy = wekaFunctions.eval(fc3, data3, data3);
//        double Array4Accuracy = wekaFunctions.eval(fc4, data4, data4);
//        double Array5Accuracy = wekaFunctions.eval(fc5, data5, data5);
        
        System.out.println("=============================================================================");
        System.out.println("fc1:\n " + fc1);
        System.out.println("fc2:\n " + fc2);
        System.out.println("fc3:\n " + fc3);
//        System.out.println("fc4:\n " + fc4);
//        System.out.println("fc5:\n " + fc5);
        
        System.out.println("Array1's accuracy: " + Array1Accuracy);
        System.out.println("Array2's accuracy: " + Array2Accuracy);
        System.out.println("Array3's accuracy: " + Array3Accuracy);
//        System.out.println("Array4's accuracy: " + Array4Accuracy);
//        System.out.println("Array5's accuracy: " + Array5Accuracy);
        
        System.out.println(l1+l2+l3+count_no+count_yes + " participants' Average Accuracy: "
                + (Array1Accuracy*l1
                        +Array2Accuracy*l2
                        +Array3Accuracy*l3
                        +noArrayAccuracy*count_no
                        +yesArrayAccuracy*count_yes)
                        /(l1+l2+l3+count_no+count_yes)
        );
        
//        System.out.println(l1+l2+l3+l4+l5+count_no+count_yes + " participants' Average Accuracy: "
//                + (Array1Accuracy*l1
//                        +Array2Accuracy*l2
//                        +Array3Accuracy*l3
//                        +Array4Accuracy*l4
//                        +Array5Accuracy*l5
//                        +noArrayAccuracy*count_no
//                        +yesArrayAccuracy*count_yes)
//                        /(l1+l2+l3+l4+l5+count_no+count_yes)
//        );
    }    
        
    public static void main(String[] args) throws Exception {
        preprocessing();
        //for (int i = 0; i < 10; i++) {
            shuffleMultiArray();
            merge1(1);
        //}
        
        //test();
    }
}    



