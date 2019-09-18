package experiments;

import cf4j.Kernel;
import cf4j.Processor;
import cf4j.model.matrixFactorization.Bmf;
import cf4j.model.matrixFactorization.Pmf;
import cf4j.model.predictions.FactorizationPrediction;
import cf4j.qualityMeasures.MAE;
import cf4j.utils.PrintableQualityMeasure;
import cf4j.utils.Range;

public class BuildCf4jFile {

	private static String DATASET = "datasets/ml100k.txt";
	private static String DATASET_SEPARATOR = "::";

	private static double TEST_USERS = 0.2;
	private static double TEST_ITEMS = 0.2;

	private static String CF4J_FILE = "datasets/ml100k.cf4j";


//	private static String DATASET = "datasets/ml1m.dat";
//	private static String DATASET_SEPARATOR = "::";
//
//	private static double TEST_USERS = 0.2;
//	private static double TEST_ITEMS = 0.2;
//
//	private static String CF4J_FILE = "datasets/ml1m.cf4j";

//	private static String DATASET = "datasets/netflix_3M.txt";
//	private static String DATASET_SEPARATOR = "::";
//
//	private static double TEST_USERS = 0.2;
//	private static double TEST_ITEMS = 0.2;
//
//	private static String CF4J_FILE = "datasets/nf3m.cf4j";

	public static void main (String [] args) {
		Kernel.getInstance().open(DATASET, TEST_USERS, TEST_ITEMS, DATASET_SEPARATOR);
		Kernel.getInstance().writeKernel(CF4J_FILE);
	}
}
