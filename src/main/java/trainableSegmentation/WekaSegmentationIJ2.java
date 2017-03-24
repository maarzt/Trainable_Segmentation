/**
 *
 * License: GPL
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *
 * Authors: Ignacio Arganda-Carreras (iarganda@mit.edu), Verena Kaynig (verena.kaynig@inf.ethz.ch),
 *          Albert Cardona (acardona@ini.phys.ethz.ch)
 */

package trainableSegmentation;

import java.awt.Rectangle;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import net.imglib2.Cursor;
import net.imglib2.roi.labeling.ImgLabeling;
import net.imglib2.roi.labeling.LabelRegion;
import net.imglib2.roi.labeling.LabelRegions;
import net.imglib2.type.numeric.integer.IntType;

import hr.irb.fastRandomForest.FastRandomForest;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.Prefs;
import ij.gui.Roi;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;


/**
 * This class contains all the library methods to perform image segmentation
 * based on the Weka classifiers.
 */
public class WekaSegmentationIJ2 {

	/** maximum number of classes (labels) allowed */
	public static final int MAX_NUM_CLASSES = 100;

	/** array of lists of Rois for each slice (vector index)
	 * and each class (arraylist index) of the training image */
	/** image to be used in the training */
	private ImagePlus trainingImage;
	/** result image after classification */
	private ImagePlus classifiedImage;
	/** features to be used in the training */
	private FeatureStackArray featureStackArray = null;

	/** set of instances from loaded data (previously saved segmentation) */
	private Instances loadedTrainingData = null;
	/** set of instances from the user's traces */
	private Instances traceTrainingData = null;
	/** current classifier */
	private AbstractClassifier classifier = null;
	/** train header */
	private Instances trainHeader = null;

	/** default classifier (Fast Random Forest) */
	private FastRandomForest rf;

	/** flag to update the feature stack (used when there is any change on the features) */
	private boolean updateFeatures = false;

	/** array of boolean flags to update (or not) specific feature stacks during training */
	private boolean[] featureStackToUpdateTrain;

	/** array of boolean flags to update (or not) specific feature stacks during test */
	private boolean[] featureStackToUpdateTest;

	/** current number of classes */
	private int numOfClasses = 0;
	/** names of the current classes */
	private String[] classLabels = new String[MAX_NUM_CLASSES];

	// Random Forest parameters
	/** current number of trees in the fast random forest classifier */
	private int numOfTrees = 200;
	/** current number of random features per tree in the fast random forest classifier */
	private int randomFeatures = 2;
	/** maximum depth per tree in the fast random forest classifier */
	private int maxDepth = 0;
	/** list of class names on the loaded data */
	private ArrayList<String> loadedClassNames = null;

	/** expected membrane thickness */
	private int membraneThickness = 1;
	/** size of the patch to use to enhance the membranes */
	private int membranePatchSize = 19;

	/** minimum sigma to use on the filters */
	private float minimumSigma = 1f;
	/** maximum sigma to use on the filters */
	private float maximumSigma = 16f;

	private boolean isProcessing3D = false;
	private FeatureStack3D fs3d = null;

	/** flags of filters to be used in 2D */
	private boolean[] enabledFeatures = new boolean[]{
			true, 	/* Gaussian_blur */
			true, 	/* Sobel_filter */
			true, 	/* Hessian */
			true, 	/* Difference_of_gaussians */
			true, 	/* Membrane_projections */
			false, 	/* Variance */
			false, 	/* Mean */
			false, 	/* Minimum */
			false, 	/* Maximum */
			false, 	/* Median */
			false,	/* Anisotropic_diffusion */
			false, 	/* Bilateral */
			false, 	/* Lipschitz */
			false, 	/* Kuwahara */
			false,	/* Gabor */
			false, 	/* Derivatives */
			false, 	/* Laplacian */
			false,	/* Structure */
			false,	/* Entropy */
			false	/* Neighbors */
	};

	/** use neighborhood flag */
	private boolean useNeighbors = false;

	/** list of the names of features to use */
	private ArrayList<String> featureNames = null;

	/**
	 * flag to set the resampling of the training data in order to guarantee
	 * the same number of instances per class (class balance)
	 * */
	private boolean balanceClasses = false;

	/** executor service to launch threads for the library operations */
	private ExecutorService exe = Executors.newFixedThreadPool(  Prefs.getThreads() );

	/**
	 * Default constructor.
	 *
	 * @param trainingImage The image to be segmented/trained
	 */
	public WekaSegmentationIJ2(ImagePlus trainingImage)
	{
		this();
		setTrainingImage(trainingImage);
	}

	/**
	 * No-image constructor. If you use this constructor, the image has to be
	 * set using setTrainingImage().
	 */
	public WekaSegmentationIJ2()
	{
		// set class label names
		for(int i=0; i<MAX_NUM_CLASSES; i++)
			this.classLabels[ i ] = new String("class " + (i+1));

		// Initialization of Fast Random Forest classifier
		rf = new FastRandomForest();
		rf.setNumTrees(numOfTrees);
		//this is the default that Breiman suggests
		//rf.setNumFeatures((int) Math.round(Math.sqrt(featureStack.getSize())));
		//but this seems to work better
		rf.setNumFeatures(randomFeatures);
		// Random seed
		rf.setSeed( (new Random()).nextInt() );
		// Set number of threads
		rf.setNumThreads( Prefs.getThreads() );

		classifier = rf;

		// start with two classes
		addClass();
		addClass();
	}

	/**
	 * Set the training image (single image or stack)
	 *
	 * @param imp training image
	 */
	public void setTrainingImage(ImagePlus imp)
	{
		this.trainingImage = imp;

		// Initialize feature stack (no features yet)
		featureStackArray = new FeatureStackArray(trainingImage.getImageStackSize(),
				minimumSigma, maximumSigma, useNeighbors, membraneThickness, membranePatchSize,
				enabledFeatures );

		featureStackToUpdateTrain = new boolean[trainingImage.getImageStackSize()];
		featureStackToUpdateTest = new boolean[trainingImage.getImageStackSize()];
		Arrays.fill(featureStackToUpdateTest, true);

		// update list of examples
		for(int i=0; i < trainingImage.getImageStackSize(); i++)
		{
			if ( !isProcessing3D )
				// Initialize each feature stack (one per slice)
				featureStackArray.set(
						new FeatureStack(
								trainingImage.getImageStack().getProcessor(i+1) ), i);
		}

	}

	/**
	 * Adds a ROI to the list of examples for a certain class
	 * and slice.
	 *
	 * @param classNum the number of the class
	 * @param roi the ROI containing the new example
	 * @param n number of the current slice
	 */
	public void addExample(int classNum, Roi roi, int n)
	{
		// In 2D: if the feature stack of that slice is not set
		// to be updated during training
		if( !isProcessing3D && !featureStackToUpdateTrain[n - 1] )
		{
			boolean updated = false;
			// if it does not contain any trace yet, set it
			// to be updated during training
			if(!updated && featureStackToUpdateTest[n - 1])
			{
				//IJ.log("Feature stack for slice " + n
				//		+ " needs to be updated");
				featureStackToUpdateTrain[n-1] = true;
				featureStackToUpdateTest[n-1] = false;
				updateFeatures = true;
			}
		}

	}

	/**
	 * Add new segmentation class.
	 */
	public void addClass()
	{

		// increase number of available classes
		numOfClasses ++;
	}

	/**
	 * Returns the current classifier.
	 */
	public AbstractClassifier getClassifier() {
		return classifier;
	}

	/**
	 * Homogenize number of instances per class
	 *
	 * @param data input set of instances
	 * @return resampled set of instances
	 * @deprecated use balanceTrainingData
	 */
	public static Instances homogenizeTrainingData(Instances data)
	{
		return WekaSegmentation.balanceTrainingData( data );
	}

	/**
	 * Balance number of instances per class
	 *
	 * @param data input set of instances
	 * @return resampled set of instances
	 */
	public static Instances balanceTrainingData( Instances data )
	{
		final Resample filter = new Resample();
		Instances filteredIns = null;
		filter.setBiasToUniformClass(1.0);
		try {
			filter.setInputFormat(data);
			filter.setNoReplacement(false);
			filter.setSampleSizePercent(100);
			filteredIns = Filter.useFilter(data, filter);
		} catch (Exception e) {
			IJ.log("Error when resampling input data!");
			e.printStackTrace();
		}
		return filteredIns;

	}

	/**
	 * Homogenize number of instances per class (in the loaded training data)
	 * @deprecated use balanceTrainingData
	 */
	public void homogenizeTrainingData()
	{
		balanceTrainingData();
	}

	/**
	 * Balance number of instances per class (in the loaded training data)
	 */
	public void balanceTrainingData()
	{
		final Resample filter = new Resample();
		Instances filteredIns = null;
		filter.setBiasToUniformClass(1.0);
		try {
			filter.setInputFormat(this.loadedTrainingData);
			filter.setNoReplacement(false);
			filter.setSampleSizePercent(100);
			filteredIns = Filter.useFilter(this.loadedTrainingData, filter);
		} catch (Exception e) {
			IJ.log("Error when resampling input data!");
			e.printStackTrace();
		}
		this.loadedTrainingData = filteredIns;
	}

	/**
	 * Filter feature stack based on the list of feature names to use
	 *
	 * @param featureNames list of feature names to use
	 * @param featureStack feature stack to filter
	 */
	public static void filterFeatureStackByList(
			ArrayList<String> featureNames,
			FeatureStack featureStack)
	{
		if (null == featureNames)
			return;

		if (Thread.currentThread().isInterrupted() )
			return;

		IJ.log("Filtering feature stack by selected attributes...");

		for(int i=1; i<=featureStack.getSize(); i++)
		{
			final String featureName = featureStack.getSliceLabel(i);
			//IJ.log(" " + featureName + "...");
			if(!featureNames.contains(featureName))
			{
				// Remove feature
				featureStack.removeFeature( featureName );
				// decrease i to avoid skipping any name
				i--;
			}
		}
	}

	public Instances createTrainingInstances(ImgLabeling<String, IntType> labeling)
	{

		// create initial set of instances
		final Instances trainingData =  new Instances( "segment", createAttributes(), 1 );
		// Set the index of the class attribute
		trainingData.setClassIndex(featureStackArray.getNumOfFeatures());

		IJ.log("Training input:");

		final boolean colorFeatures = this.trainingImage.getType() == ImagePlus.COLOR_RGB;

		// For all classes
		for(int classIndex = 0; classIndex < numOfClasses; classIndex++)
		{
			int nl = 0;
			// Read all lists of examples
			for(int sliceNum = 1; sliceNum <= trainingImage.getImageStackSize(); sliceNum ++) {
				final FeatureStack fs = featureStackArray.get( sliceNum - 1 );
				final LabelRegions<String> regions = new LabelRegions<>(labeling);
				LabelRegion<String> region = regions.getLabelRegion(String.valueOf(classIndex + 1));
				Cursor<Void> cursor = region.cursor();
				while(cursor.hasNext()) {
					cursor.next();
					nl++;
					trainingData.add( fs.createInstance(cursor.getIntPosition(0), cursor.getIntPosition(1), classIndex) );
				}
			}

			IJ.log("# of pixels selected as " + getClassLabels()[classIndex] + ": " +nl);
		}

		if (trainingData.numInstances() == 0)
			return null;

		return trainingData;
	}

	private ArrayList<Attribute> createAttributes() {
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int i=1; i<=featureStackArray.getNumOfFeatures(); i++)
		{
			String attString = featureStackArray.getLabel(i);
			attributes.add(new Attribute(attString));
		}

		final ArrayList<String> classes;

		if(null == this.loadedTrainingData)
		{
			classes = new ArrayList<String>();
			for(int i = 0; i < numOfClasses ; i ++)
			{
				for(int n=0; n<trainingImage.getImageStackSize(); n++)
				{
					if(!classes.contains(getClassLabels()[i]))
						classes.add(getClassLabels()[i]);
				}
			}
		}
		else
		{
			classes = this.loadedClassNames;
		}

		attributes.add(new Attribute("class", classes));
		return attributes;
	}

	public void filterFeatureStackByList()
	{
		if (null == this.featureNames)
			return;

		for(int i=1; i<=this.featureStackArray.getNumOfFeatures(); i++)
		{
			final String featureName = this.featureStackArray.getLabel(i);
			if(!this.featureNames.contains(featureName))
			{
				// Remove feature
				for(int j=0; j<this.featureStackArray.getSize(); j++)
					this.featureStackArray.get(j).removeFeature( featureName );
				// decrease i to avoid skipping any name
				i--;
			}
		}

	}

	/**
	 * Train classifier with the current instances
	 * @param labeling
	 */
	public boolean trainClassifier(ImgLabeling<String, IntType> labeling)
	{
		if (Thread.currentThread().isInterrupted() )
		{
			IJ.log("Classifier training was interrupted.");
			return false;
		}

		// At least two lists of different classes of examples need to be non empty
		int nonEmpty = 2;
		int sliceWithTraces = 0;

		if (nonEmpty < 2 && null == loadedTrainingData)
		{
			IJ.showMessage("Cannot train without at least 2 sets of examples!");
			return false;
		}

		// Create feature stack if necessary (training from traces
		// and the features stack is empty or the settings changed)
		if(nonEmpty > 1 && featureStackArray.isEmpty() || updateFeatures)
		{
			IJ.showStatus("Creating feature stack...");
			IJ.log("Creating feature stack...");
			long start = System.currentTimeMillis();

			// set the reference slice to one with traces
			featureStackArray.setReference( sliceWithTraces );

			if ( !isProcessing3D &&
					!featureStackArray.updateFeaturesMT(featureStackToUpdateTrain))
			{
				IJ.log("Feature stack was not updated.");
				IJ.showStatus("Feature stack was not updated.");
				return false;
			}
			Arrays.fill(featureStackToUpdateTrain, false);
			filterFeatureStackByList();
			updateFeatures = false;

			long end = System.currentTimeMillis();
			IJ.log("Feature stack array is now updated (" + featureStackArray.getSize()
					+ " slice(s) with " + featureStackArray.getNumOfFeatures()
					+ " feature(s), took " + (end-start) + "ms).");
		}

		IJ.showStatus("Creating training instances...");
		Instances data = null;
		{
			final long start = System.currentTimeMillis();
			traceTrainingData = data = createTrainingInstances(labeling);
			final long end = System.currentTimeMillis();
			IJ.log("Creating training data took: " + (end-start) + "ms");
		}

		// Update train header
		this.trainHeader = new Instances(data, 0);

		// Resample data if necessary
		if(balanceClasses)
		{
			final long start = System.currentTimeMillis();
			IJ.showStatus("Balancing classes distribution...");
			IJ.log("Balancing classes distribution...");
			data = balanceTrainingData(data);
			final long end = System.currentTimeMillis();
			IJ.log("Done. Balancing classes distribution took: " + (end-start) + "ms");
		}

		IJ.showStatus("Training classifier...");
		IJ.log("Training classifier...");

		if (Thread.currentThread().isInterrupted() )
		{
			IJ.log("Classifier training was interrupted.");
			return false;
		}

		// Train the classifier on the current data
		final long start = System.currentTimeMillis();
		try{
			classifier.buildClassifier(data);
		}
		catch (InterruptedException ie)
		{
			IJ.log("Classifier construction was interrupted.");
			return false;
		}
		catch(Exception e){
			IJ.showMessage(e.getMessage());
			e.printStackTrace();
			return false;
		}

		// Print classifier information
		IJ.log( this.classifier.toString() );

		final long end = System.currentTimeMillis();

		IJ.log("Finished training in "+(end-start)+"ms");
		return true;
	}

	/**
	 * Apply current classifier to a given image. If the input image is a
	 * stack, the classification task will be carried out by slice in
	 * parallel by each available thread (number of threads read on
	 * Prefs.getThreads()). Each thread will sequentially process a whole
	 * number of slices (first feature calculation then classification).
	 *
	 * @param imp image (2D single image or stack)
	 * @return result image (classification)
	 */
	public ImagePlus applyClassifier(final ImagePlus imp)
	{
		return applyClassifier(imp, 0, false);
	}


	/**
	 * Apply current classifier to a given image. It divides the
	 * whole slices of the input image into the selected number of threads.
	 * Each thread will sequentially process a whole  number of slices (first
	 * feature calculation then classification).
	 *
	 * @param imp image (2D single image or stack)
	 * @param numThreads The number of threads to use. Set to zero for
	 * auto-detection (set by the user on the ImageJ preferences)
	 * @param probabilityMaps create probability maps for each class instead of
	 * a classification
	 * @return result image
	 */
	public ImagePlus applyClassifier(
			final ImagePlus imp,
			int numThreads,
			final boolean probabilityMaps)
	{
		if (numThreads == 0)
			numThreads = Prefs.getThreads();

		final int numSliceThreads = Math.min(imp.getStackSize(), numThreads);
		final int numClasses      = numOfClasses;
		final int numChannels     = (probabilityMaps ? numClasses : 1);

		IJ.log("Processing slices of " + imp.getTitle() + " in " + numSliceThreads + " thread(s)...");

		// Set proper class names (skip empty list ones)
		ArrayList<String> classNames = new ArrayList<String>();
		if( null == loadedClassNames )
		{
			for(int i = 0; i < numOfClasses; i++)
				for(int j=0; j<trainingImage.getImageStackSize(); j++)
					{
						classNames.add(getClassLabels()[i]);
						break;
					}
		}
		else
			classNames = loadedClassNames;

		final ImagePlus[] classifiedSlices = new ImagePlus[imp.getStackSize()];

		class ApplyClassifierThread extends Thread
		{

			private final int startSlice;
			private final int numSlices;
			private final int numFurtherThreads;
			private final ArrayList<String> classNames;

			public ApplyClassifierThread(
					int startSlice,
					int numSlices,
					int numFurtherThreads,
					ArrayList<String> classNames)
			{

				this.startSlice         = startSlice;
				this.numSlices          = numSlices;
				this.numFurtherThreads  = numFurtherThreads;
				this.classNames         = classNames;
			}

			@Override
			public void run()
			{

				for (int i = startSlice; i < startSlice + numSlices; i++)
				{
					final ImagePlus slice = new ImagePlus(imp.getImageStack().getSliceLabel(i), imp.getImageStack().getProcessor(i));
					// Create feature stack for slice
					IJ.showStatus("Creating features...");
					IJ.log("Creating features for slice " + i +  "...");
					final FeatureStack sliceFeatures = new FeatureStack(slice);
					// Use the same features as the current classifier
					sliceFeatures.setEnabledFeatures(featureStackArray.getEnabledFeatures());
					sliceFeatures.setMaximumSigma(maximumSigma);
					sliceFeatures.setMinimumSigma(minimumSigma);
					sliceFeatures.setMembranePatchSize(membranePatchSize);
					sliceFeatures.setMembraneSize(membraneThickness);
					sliceFeatures.updateFeaturesST();
					filterFeatureStackByList(featureNames, sliceFeatures);
					final Instances sliceData = sliceFeatures.createInstances(classNames);
					sliceData.setClassIndex(sliceData.numAttributes() - 1);

					IJ.log("Classifying slice " + i + " in " + numFurtherThreads + " thread(s)...");
					final ImagePlus classImage = applyClassifier(sliceData, slice.getWidth(), slice.getHeight(), numFurtherThreads, probabilityMaps);

					if( null == classImage )
					{
						IJ.log("Error while applying classifier!");
						return;
					}

					classImage.setTitle("classified_" + slice.getTitle());

					classifiedSlices[i-1] = classImage;
				}
			}
		}

		final int numFurtherThreads = (int)Math.ceil((double)(numThreads - numSliceThreads)/numSliceThreads) + 1;
		final ApplyClassifierThread[] threads = new ApplyClassifierThread[numSliceThreads];

		// calculate optimum number of slices per thread
		int[] numSlicesPerThread = new int [ numSliceThreads ];
		for(int i=0; i<imp.getImageStackSize(); i++)
		{
			numSlicesPerThread[ i % numSliceThreads ] ++;
		}

		int aux = 0;
		for (int i = 0; i < numSliceThreads; i++)
		{

			int startSlice = aux + 1;

			aux += numSlicesPerThread[ i ];

			IJ.log("Starting thread " + i + " processing " + numSlicesPerThread[ i ] + " slices, starting with " + startSlice);
			threads[i] = new ApplyClassifierThread(startSlice, numSlicesPerThread[ i ], numFurtherThreads, classNames );

			threads[i].start();
		}

		// create classified image
		final ImageStack classified = new ImageStack(imp.getWidth(), imp.getHeight());

		// join threads
		for(Thread thread : threads)
			try {
				thread.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}

		// assemble classified image
		for (int i = 0; i < imp.getStackSize(); i++)
			for (int c = 0; c < numChannels; c++)
				classified.addSlice("", classifiedSlices[i].getStack().getProcessor(c+1));

		ImagePlus result = new ImagePlus("Classification result", classified);

		if (probabilityMaps)
		{
			result.setDimensions(numOfClasses, imp.getNSlices(), imp.getNFrames());
			if (imp.getNSlices()*imp.getNFrames() > 1)
				result.setOpenAsHyperStack(true);
		}
		result.setCalibration(trainingImage.getCalibration());

		return result;
	}

	/**
	 * Apply current classifier to set of instances
	 * @param data set of instances
	 * @param w image width
	 * @param h image height
	 * @param numThreads The number of threads to use. Set to zero for
	 * auto-detection.
	 * @return result image
	 */
	public ImagePlus applyClassifier(final Instances data, int w, int h, int numThreads, boolean probabilityMaps)
	{
		if (numThreads == 0)
			numThreads = Prefs.getThreads();

		final int numClasses   = data.numClasses();
		final int numInstances = data.numInstances();
		final int numChannels  = (probabilityMaps ? numClasses : 1);
		final int numSlices    = (numChannels*numInstances)/(w*h);

		IJ.showStatus("Classifying image...");

		final long start = System.currentTimeMillis();

		ExecutorService exe = Executors.newFixedThreadPool(numThreads);
		final double[][][] results = new double[numThreads][][];
		final Instances[] partialData = new Instances[numThreads];
		final int partialSize = numInstances / numThreads;
		Future<double[][]>[] fu = new Future[numThreads];

		final AtomicInteger counter = new AtomicInteger();

		for(int i = 0; i < numThreads; i++)
		{
			if (Thread.currentThread().isInterrupted())
			{
				exe.shutdown();
				return null;
			}
			if(i == numThreads - 1)
				partialData[i] = new Instances(data, i*partialSize, numInstances - i*partialSize);
			else
				partialData[i] = new Instances(data, i*partialSize, partialSize);

			AbstractClassifier classifierCopy = null;
			try {
				// The Weka random forest classifiers do not need to be duplicated on each thread
				// (that saves much memory)
				if( classifier instanceof FastRandomForest || classifier instanceof RandomForest )
					classifierCopy = classifier;
				else
					classifierCopy = (AbstractClassifier) (AbstractClassifier.makeCopy( classifier ));

			} catch (Exception e) {
				IJ.log("Error: classifier could not be copied to classify in a multi-thread way.");
				e.printStackTrace();
			}
			fu[i] = exe.submit(classifyInstances(partialData[i], classifierCopy, counter, probabilityMaps));
		}

		ScheduledExecutorService monitor = Executors.newScheduledThreadPool(1);
		ScheduledFuture task = monitor.scheduleWithFixedDelay(new Runnable() {
			@Override
			public void run() {
				IJ.showProgress(counter.get(), numInstances);
			}
		}, 0, 1, TimeUnit.SECONDS);

		// Join threads
		for(int i = 0; i < numThreads; i++)
		{
			try {
				results[i] = fu[i].get();
			} catch (InterruptedException e) {
				//e.printStackTrace();
				return null;
			} catch (ExecutionException e) {
				e.printStackTrace();
				return null;
			} finally {
				exe.shutdown();
				task.cancel(true);
				monitor.shutdownNow();
				IJ.showProgress(1);
			}
		}

		exe.shutdown();

		// Create final array
		double[][] classificationResult;
		classificationResult = new double[numChannels][numInstances];

		for(int i = 0; i < numThreads; i++)
			for (int c = 0; c < numChannels; c++)
				System.arraycopy(results[i][c], 0, classificationResult[c], i*partialSize, results[i][c].length);

		IJ.showProgress(1.0);
		final long end = System.currentTimeMillis();
		IJ.log("Classifying whole image data took: " + (end-start) + "ms");

		double[]         classifiedSlice = new double[w*h];
		final ImageStack classStack      = new ImageStack(w, h);

		for (int i = 0; i < numSlices/numChannels; i++)
		{
			for (int c = 0; c < numChannels; c++)
			{
				System.arraycopy(classificationResult[c], i*(w*h), classifiedSlice, 0, w*h);
				ImageProcessor classifiedSliceProcessor = new FloatProcessor(w, h, classifiedSlice);
				classStack.addSlice(probabilityMaps ? getClassLabels()[c] : "", classifiedSliceProcessor);
			}
		}
		ImagePlus classImg = new ImagePlus(probabilityMaps ? "Probability maps" : "Classification result", classStack);

		classImg.setCalibration(trainingImage.getCalibration());
		return classImg;
	}


	/**
	 * Classify instances concurrently
	 *
	 * @param data set of instances to classify
	 * @param classifier current classifier
	 * @param counter auxiliary counter to be able to update the progress bar
	 * @param probabilityMaps return a probability map for each class instead of a
	 * classified image
	 * @return classification result
	 */
	private static Callable<double[][]> classifyInstances(
			final Instances data,
			final AbstractClassifier classifier,
			final AtomicInteger counter,
			final boolean probabilityMaps)
	{
		if (Thread.currentThread().isInterrupted())
			return null;

		return new Callable<double[][]>(){

			@Override
			public double[][] call(){

				final int numInstances = data.numInstances();
				final int numClasses   = data.numClasses();

				final double[][] classificationResult;

				if (probabilityMaps)
					classificationResult = new double[numClasses][numInstances];
				else
					classificationResult = new double[1][numInstances];

				for (int i=0; i<numInstances; i++)
				{
					try{

						if (0 == i % 4000)
						{
							if (Thread.currentThread().isInterrupted())
								return null;
							counter.addAndGet(4000);
						}

						if (probabilityMaps)
						{
							double[] prob = classifier.distributionForInstance(data.get(i));
							for(int k = 0 ; k < numClasses; k++)
								classificationResult[k][i] = prob[k];
						}
						else
						{
							classificationResult[0][i] = classifier.classifyInstance(data.get(i));
						}

					}catch(Exception e){

						IJ.showMessage("Could not apply Classifier!");
						e.printStackTrace();
						return null;
					}
				}
				return classificationResult;
			}
		};
	}

	/**
	 * Set the new enabled features
	 * @param newFeatures new enabled feature flags
	 */
	/**
	 * Get the current class labels
	 * @return array containing all the class labels
	 */
	public String[] getClassLabels()
	{
		return classLabels;
	}

}
