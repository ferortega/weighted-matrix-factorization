package experiments;

import cf4j.Kernel;

public class DatasetStats {

      private static String CF4J_FILE = "datasets/ml100k.cf4j";
//    private static String CF4J_FILE = "datasets/ml1m.cf4j";
//    private static String CF4J_FILE = "datasets/nf3m.cf4j";

    public static void main (String [] args) {
        Kernel.getInstance().readKernel(CF4J_FILE);
        String info = Kernel.getInstance().getKernelInfo();
        System.out.print(info);
    }
}
