/*
/**
 * @author Yang
 */
import java.util.ArrayList;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class BottomUp {
    
    public static int ROUNDS = 434;
    public static int classIndex;
    
    public static double accuracy;
    public static float accuracy2;
    
    public static DataSource source;
    public static DataSource source1;

    public static ArrayList<Integer> array1 = new ArrayList<>();
    public static ArrayList<Integer> array2 = new ArrayList<>();
    public static int[][] array3 = new int[436][436];
    public static ArrayList<Integer> array4 = new ArrayList<>();
    public static ArrayList<Integer> cstore = new ArrayList<>();
    public static ArrayList<Integer> mix = new ArrayList<>();

    public static float[][] NoneSingleNodeUserAccuracyList = new float[436][436];
    public static float[][] accuracyDiff = new float[436][436];
    public static float[][] accuracyPair = new float[436][436];
    public static int numberOfElemtRow1 = 0;
    public static int numberOfElemtRow2 = 0;


    public static void main(String[] args) throws Exception{   
        int c1=0,c2=0,c3=0,c4=0,c5=0,c6=0;
        source=new DataSource("docs/samsung.arff");
        Instances allusers=source.getDataSet();
        classIndex=allusers.numAttributes()-1;
        allusers.setClassIndex(classIndex);
        // System.out.println(allusers.numInstances());

        int array3Index = 0;
        //for (int i = 0; i < 240; i = i + 12) {
        for (int i = 0; i < allusers.numInstances(); i = i + 12) {
            int userID = (int)allusers.instance(i).value(0);
            Instances singleuser = new Instances(allusers, i, 12);

            FilteredClassifier fc= wekaFunctions.train(singleuser, classIndex);  
            accuracy=  wekaFunctions.eval(fc,singleuser,singleuser); // evalCrossValidation was used here earlier
//            System.out.println("User #:" +userID);
//            System.out.println("Classifier :" +fc);
//            System.out.println("Accuracy :" +accuracy);
        
            String abm=fc.graph();  //can store this in a string and use string functions to parse it; put in if conditions to determine the clusters


            String t1="N0 [label=\"0";
            String t2="N0 [label=\"1";  
            String t3="N0 [label=\"cstore";
            
            //System.out.println(abm);
           
            if(abm.contains(t1))
            {
                c1++;
                array1.add(userID);
            }
            if(abm.contains(t2))
            {
                c2++;
                array2.add(userID);
            }
            if(abm.contains(t1)==false && abm.contains(t2)==false)
            {
                c3++;
                array3[array3Index][0] = (userID);
                NoneSingleNodeUserAccuracyList[array3Index++][0] = (float)accuracy;
            } 
            if(abm.contains(t3))
            {
                cstore.add(userID);
            }
            if(abm.contains(t1)==false && abm.contains(t2)==false && abm.contains(t3)==false)
            {
                mix.add(userID);
            }
               
//            File f = new File("docs/individual_accuracy.txt");    
//            if(!f.exists()){				
//                if (f.createNewFile()) {  
//                    System.out.println("Success!");           
//                }
//            }               
//            PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("docs/individual_accuracy.txt", true)));
//            out.println(userID+"->"+accuracy);
//        
//            // out.println(abm);
//            out.close();
        }
        System.out.println("Single 'NO' node trees :" +c1);
        System.out.println("Single 'YES' node trees :" +c2);
        System.out.println("Multi-node trees :" +c3);   
//        System.out.println("Users with multi-node trees :");
//        
//        System.out.println("array3's length is: "+array3.length);
//        for(int i=0; i<array3.length;i++){
//            System.out.println(array3[i][0]);
//        }
//        
//        System.out.println("cstore's length is: "+cstore.size());
//        for(int i=0; i<cstore.size();i++){
//            System.out.println(cstore.get(i));
//        }
//        
//        System.out.println("mix's length is: "+mix.size());
//        for(int i=0; i<mix.size();i++){
//            System.out.println(mix.get(i));
//        }

        arffFunctions.generateArff(array1, "docs/samsung_header.txt", "bottomsup1.arff");
        arffFunctions.generateArff(array2, "docs/samsung_header.txt", "bottomsup2.arff");
        //user.generateArff(array3,"bottomsup3.arff");   
        arffFunctions.generateArff(cstore, "docs/samsung_header.txt", "cstore.arff");
        arffFunctions.generateArff(mix, "docs/samsung_header.txt", "mix.arff");


// 2ND PART OF BOTTOMSUP------------------------------------------------------------------------------------------------------------------------------------------
        // print NoneSingleNodeUserAccuracyList, Yang
//        for (int i = 0; i < NoneSingleNodeUserAccuracyList.length; i++ ){
//            System.out.print((i+1) + ":\t");
//            printRow(NoneSingleNodeUserAccuracyList[i]);
//        }
        float sum = 0;
        for (float[] row : NoneSingleNodeUserAccuracyList) {
            for (int j = 0; row[j] != 0; j++) {
                sum += row[j];
            }
        }
        System.out.println("None Single Node users' Overall Accuracy: " + sum/NoneSingleNodeUserAccuracyList.length);

        for(int round = 0; round < ROUNDS; round++){
            System.out.println("Round " + (round+1) );            

            for(int i=0;i<436;i++)
            {
                if (array3[i][0] == 0){
                    for(int j=0;j<436;j++){
                        accuracyDiff[i][j] = -100;
                    }
                }
                else {
                    for(int j=i+1;j<436;j++)
                    {
                        if (array3[j][0] == 0){
                            accuracyDiff[i][j] = -100;
                        }
                        else {
                            array4.clear();
                            for (int k:array3[i]){
                               if (k != 0)
                                   array4.add(k);
                            }
                            for (int l:array3[j]){
                                if (l != 0)
                                    array4.add(l);
                            }
                            //System.out.println("array4: " + array4);
                            
                            numberOfElemtRow1 = 0;
                            numberOfElemtRow2 = 0;
                            while(NoneSingleNodeUserAccuracyList[i][numberOfElemtRow1]!=0){
                                numberOfElemtRow1++;
                                //System.out.println("number of Elements in Row1: " + numberOfElemtRow1);
                            }
                            while(NoneSingleNodeUserAccuracyList[j][numberOfElemtRow2]!=0){
                                numberOfElemtRow2++;
                                //System.out.println("number of Elements in Row2: " + numberOfElemtRow2);
                            }
                            float averageAccuracy = ( NoneSingleNodeUserAccuracyList[i][0]*numberOfElemtRow1 + NoneSingleNodeUserAccuracyList[j][0]*numberOfElemtRow2 )/(numberOfElemtRow1+numberOfElemtRow2);
                            //System.out.println(averageAccuracy);
                            arffFunctions.generateArff(array4, "docs/samsung_header.txt", "whatever.arff");   // will be stored in ../newly_generated ; pairs are stored in this file
                            source1=new DataSource("docs/whatever.arff");
                            Instances weirdusers=source1.getDataSet();
                            weirdusers.setClassIndex(classIndex);
                            FilteredClassifier fc2 = wekaFunctions.train(weirdusers, classIndex);
                            String abm2=fc2.graph();
                            accuracy2= (float)wekaFunctions.eval(fc2,weirdusers,weirdusers); 
                            //System.out.println(array4+": "+(accuracy2-averageAccuracy) );
                            
                            accuracyDiff[i][j] = accuracy2 - averageAccuracy;
                            accuracyPair[i][j] = accuracy2;

                            // File file = new File("../newly_generated/numbers.txt");    
                            // if(!file.exists()){				
                            //     if (file.createNewFile()) {  
                            //         System.out.println("Success!");  
                                    
                            //     } else {  
                            //         System.out.println("Fail!");  
                            //     }  
                            // }
                            // if(i!=j)
                            // {
                            //     PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("../newly_generated/numbers.txt",true))); // add "true" as a second parameter to append file
                            //     out.println(array3[i][0]+","+array3[j][0]+"->"+accuracy2);
                            //     // out.println(accuracy2);
                            //     // out.println(abm2);
                            //     out.close();
                            // }

                            // if(abm2.contains(t1) && array1 does not have elements at i and j)
                            // {
                            //     array1.add(array3.get(i));
                            //     array1.add(array3.get(i));
                            // }
                        }
                    }
                }
        // TO DO:		
                    // source 
                    // instance
                    // train
                    // print result to a file
                    // we can also have an if loop  to avoid writing the entries that have same values for i and j
                    // traverse file and find out which ones to merge               
                    // try the same thing done above using unfruned TreeSet
                    // store the result in a file; merge those users whose average of accuracies is more or less equal to the new combined accuracy we have
                    // stop merging based on how many clusters we need
            }
            float maxAccuracyDiff = accuracyDiff[0][1];
            int index1 = 0;
            int index2 = 1;
            int firstIndex = 0;
            int secondIndex = 0;

//             for (int i = 0; i < 436; i++ ){
//                 for (int j = 0; j < 436; j++ ){
//                     System.out.print(accuracyDiff[i][j] + "\t");
//                 }
//                 System.out.println();
//             }

            for(int i=0;i<436;i++)
            {
                for(int j=i+1;j<436;j++)
                {
                    if (accuracyDiff[i][j] > maxAccuracyDiff) {
                        maxAccuracyDiff = accuracyDiff[i][j];
                        index1 = i;
                        index2 = j;
                    }
                    //System.out.println(array3[i][0] + " and " + array3[j][0] + ": " + accuracyDiff[i][j]);
                }
            }
            System.out.println("Max pair: Row " + (index1+1) + " and Row " + (index2+1) + ", Accuracy Difference: " + maxAccuracyDiff);
            
            while (array3[index1][firstIndex] != 0){
                firstIndex++;
            }
            for (secondIndex = 0; array3[index2][secondIndex] != 0 ; secondIndex++){
                array3[index1][firstIndex] = array3[index2][secondIndex];
                array3[index2][secondIndex] = 0;
                firstIndex++;
            }
            if(ROUNDS==434){
                System.out.print("User cluters after merge:\n");
                System.out.print("=================================================================================\n");
                for (int i = 0; i < 436; i++ ){
                    System.out.print((i+1) + ":\t");
                    printRow(array3[i]);
                }
                System.out.print("=================================================================================\n");
            }
            //System.out.println(numberOfElemtRow1+numberOfElemtRow2);
            for (int i = 0; i < (numberOfElemtRow1+numberOfElemtRow2); i++){
                NoneSingleNodeUserAccuracyList[index1][i] = accuracyPair[index1][index2];
            }
            for (int j = 0; j < numberOfElemtRow2; j++){
                NoneSingleNodeUserAccuracyList[index2][j] = 0;
            }
            numberOfElemtRow1 = 0;
            numberOfElemtRow2 = 0;
            if(ROUNDS==434){
                for (int i = 0; i < 436; i++ ){
                    System.out.print((i+1) + ":\t");
                    printRow(NoneSingleNodeUserAccuracyList[i]);
                }
            }
            sum = 0;
            for (int i = 0; i < 436; i++ ){
                for (int j = 0; NoneSingleNodeUserAccuracyList[i][j] != 0; j++){
                    sum+=NoneSingleNodeUserAccuracyList[i][j];
                }
            }
            System.out.println("436 users' Average Accuracy: " + sum/436);
    
            for (int i = 0; i < 436; i++ ){
                for (int j = 0; j < 436; j++ ){
                    accuracyDiff[i][j] = 0;
                    accuracyPair[i][j] = 0;
                }
            }
        }
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
}    


