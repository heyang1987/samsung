import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class user {
    
    public static ArrayList<Integer> getUserIDArray() throws Exception {
        ArrayList<Integer> userArray = new ArrayList<Integer>();
        DataSource source = new DataSource("docs/samsung.arff");
        Instances data = source.getDataSet();

        for (int i = 0; i < data.numInstances(); i= i + 12) {
            userArray.add((int)data.instance(i).value(0));
        }
        //System.out.println(userArray);
        return userArray;
    }
    
    public static boolean splitUser() throws Exception {

        DataSource source = new DataSource("docs/samsung.arff");
        Instances data = source.getDataSet();

        for (int i = 0; i < data.numInstances(); i++) {
            String destFileName = "users/" + (int)data.instance(i).value(0) + ".txt";
            File file = new File(destFileName);
            if(!file.exists()){
                try {
                    if (file.createNewFile()) {  
                        System.out.println("Create " + destFileName + "--Success!");  
                    } else {  
                        System.out.println("Create " + destFileName + "--Fail!");  
                    }  
                } catch (IOException e) {
                    e.printStackTrace();  
                    System.out.println("Create " + destFileName + "--Fail! " + e.getMessage());
                    return false;
                }
            }
            PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(destFileName, true)));
            out.println(data.instance(i));
            out.close();
        }
        return true;
    }
	
    public static void main (String[] args) throws Exception {
            splitUser();
    }

}
