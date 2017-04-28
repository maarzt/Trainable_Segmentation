package trainableSegmentation.ij2;

import ij.ImagePlus;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imglib2.Cursor;
import net.imglib2.Dimensions;
import net.imglib2.FinalDimensions;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.img.ImagePlusAdapter;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.outofbounds.OutOfBoundsBorderFactory;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import net.imglib2.view.composite.RealComposite;
import org.junit.Ignore;
import org.junit.Test;
import org.scijava.Context;
import trainableSegmentation.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertTrue;

/**
 * @author Matthias Arzt
 */
public class FeatureStackTest {

	private static ImagePlus bridgeImage = Utils.loadImage("bridge.png");

	private static Img<FloatType> bridgeImg = ImagePlusAdapter.convertFloat(bridgeImage);

	private static OpService ops = new Context(OpService.class).getService(OpService.class);

	public static void main(String... args) {
		new FeatureStackTest().testCalculateHessian();
	}

	@Test
	public void testEmptyStack() {
		RandomAccessibleInterval<FloatType> expected = generateFeatureStack(bridgeImage, nCopiesAsArray(20, false));
		RandomAccessibleInterval<FloatType> result = createEmptyStack(bridgeImg);
		Utils.assertImagesEqual(expected, result);
	}

	private RandomAccessibleInterval<FloatType> createEmptyStack(Img<FloatType> image) {
		return image;
	}

	@Test
	public void testGaussStack() {
		RandomAccessibleInterval<FloatType> expected = generateSingleFeature(bridgeImage, FeatureStack.GAUSSIAN);
		RandomAccessibleInterval<FloatType> result = createGaussStack(bridgeImg);
		assertTrue(Utils.psnr(expected, result) > 40);
	}

	private RandomAccessibleInterval<FloatType> createGaussStack(Img<FloatType> image) {
		List<RandomAccessibleInterval<FloatType>> features = new ArrayList<>();
		features.add(image);
		final double minimumSigma = 1;
		final double maximumSigma = 16;
		for(double sigma = minimumSigma; sigma <= maximumSigma; sigma *= 2)
			features.add(ops.filter().gauss(image, sigma*0.4));
		return Views.stack(features);
	}

	@Ignore("not implemented yet")
	@Test
	public void testHessianStack() {
		RandomAccessibleInterval<FloatType> expected = generateSingleFeature(bridgeImage, FeatureStack.HESSIAN);
		RandomAccessibleInterval<FloatType> result = createHessianStack(bridgeImg);
		viewDifference(expected, result);
		ImageJFunctions.show(expected);
		ImageJFunctions.show(result);
		float psnr = Utils.psnr(expected, result);
		System.out.print(psnr);
		assertTrue(psnr > 40);
	}

	private RandomAccessibleInterval<FloatType> createHessianStack(Img<FloatType> img) {
		return null;
	}

	@Test
	public void testCalculateHessian() {
		RandomAccessibleInterval<FloatType> expected = ImagePlusAdapter.wrapFloat(FeatureStack.calculateHessianOnChannel(bridgeImage, 8));
		RandomAccessibleInterval<FloatType>	actual = calculateHessianOnChannel(bridgeImg, 8);
		assertTrue(Utils.psnr(expected, actual) > 60);
	}

	private RandomAccessibleInterval<FloatType> calculateHessianOnChannel(Img<FloatType> bridgeImg, float sigma) {
		double[] sigmas = {0.4 * sigma, 0.4 * sigma};
		
		RandomAccessibleInterval<FloatType> blurred = gauss(bridgeImg, sigmas);
		RandomAccessibleInterval<FloatType> dx = deriveX(blurred);
		RandomAccessibleInterval<FloatType> dy = deriveY(blurred);
		RandomAccess<FloatType> dxx = deriveX(dx).randomAccess();
		RandomAccess<FloatType> dxy = deriveY(dx).randomAccess();
		RandomAccess<FloatType> dyy = deriveY(dy).randomAccess();

		Img<FloatType> features = ops.create().img(extendDimension(bridgeImg, 8), new FloatType());

		Cursor<RealComposite<FloatType>> cursor = Views.iterable(Views.collapseReal(features)).cursor();
		while (cursor.hasNext()) {
			cursor.next();
			dxx.setPosition(cursor);
			dxy.setPosition(cursor);
			dyy.setPosition(cursor);
			float s_xx = dxx.get().get();
			float s_xy = dxy.get().get();
			float s_yy = dyy.get().get();
			calculateHessianPerPixel(cursor.get(), s_xx, s_xy, s_yy);
		}

		return features;
	}

	private Dimensions extendDimension(Dimensions dimension, int size) {
		int numDimensions = dimension.numDimensions();
		long[] result = new long[numDimensions + 1];
		for (int j = 0; j < numDimensions; j++) result[j] = dimension.dimension(j);
		result[numDimensions] = size;
		return new FinalDimensions(result);
	}

	private static final int HESSIAN = 0;
	private static final int TRACE = 1;
	private static final int DETERMINANT = 2;
	private static final int EIGENVALUE_1 = 3;
	private static final int EIGENVALUE_2 = 4;
	private static final int ORIENTATION = 5;
	private static final int SQUARE_EIGENVALUE_DIFFERENCE = 6;
	private static final int NORMALIZED_EIGENVALUE_DIFFERENCE = 7;

	private void calculateHessianPerPixel(RealComposite<FloatType> output,
										  float s_xx, float s_xy, float s_yy)
	{
		final double t = Math.pow(1, 0.75);

		// Hessian module: sqrt (a^2 + b*c + d^2)
		output.get(HESSIAN).set((float) Math.sqrt(s_xx*s_xx + s_xy*s_xy+ s_yy*s_yy));
		// Trace: a + d
		final float trace = s_xx + s_yy;
		output.get(TRACE).set(trace);
		// Determinant: a*d - c*b
		final float determinant = s_xx*s_yy-s_xy*s_xy;
		output.get(DETERMINANT).set(determinant);

		// Ratio
		//ipRatio.set((float)(trace*trace) / determinant);
		// First eigenvalue: (a + d) / 2 + sqrt( ( 4*b^2 + (a - d)^2) / 2 )
		output.get(EIGENVALUE_1).set((float) ( trace/2.0 + Math.sqrt((4*s_xy*s_xy + (s_xx - s_yy)*(s_xx - s_yy)) / 2.0 ) ) );
		// Second eigenvalue: (a + d) / 2 - sqrt( ( 4*b^2 + (a - d)^2) / 2 )
		output.get(EIGENVALUE_2).set((float) ( trace/2.0 - Math.sqrt((4*s_xy*s_xy + (s_xx - s_yy)*(s_xx - s_yy)) / 2.0 ) ) );
		// Orientation
		if (s_xy < 0.0) // -0.5 * acos( (a-d) / sqrt( 4*b^2 + (a - d)^2)) )
		{
			float orientation =(float)( -0.5 * Math.acos((s_xx	- s_yy)
					/ Math.sqrt(4.0 * s_xy * s_xy + (s_xx - s_yy) * (s_xx - s_yy)) ));
			if (Float.isNaN(orientation))
				orientation = 0;
			output.get(ORIENTATION).set(orientation);
		}
		else 	// 0.5 * acos( (a-d) / sqrt( 4*b^2 + (a - d)^2)) )
		{
			float orientation =(float)( 0.5 * Math.acos((s_xx	- s_yy)
					/ Math.sqrt(4.0 * s_xy * s_xy + (s_xx - s_yy) * (s_xx - s_yy)) ));
			if (Float.isNaN(orientation))
				orientation = 0;
			output.get(ORIENTATION).set(orientation);
		}
		// Gamma-normalized square eigenvalue difference
		output.get(SQUARE_EIGENVALUE_DIFFERENCE).set((float) ( Math.pow(t,4) * trace*trace * ( (s_xx - s_yy)*(s_xx - s_yy) + 4*s_xy*s_xy ) ) );
		// Square of Gamma-normalized eigenvalue difference
		output.get(NORMALIZED_EIGENVALUE_DIFFERENCE).set((float) ( Math.pow(t,2) * ( (s_xx - s_yy)*(s_xx - s_yy) + 4*s_xy*s_xy ) ) );
	}

	private ImageProcessor imageProcessor(RandomAccessibleInterval<FloatType> dy) {
		int width = (int) dy.dimension(0);
		int height = (int) dy.dimension(1);
		FloatProcessor processor = new FloatProcessor(width, height);
		RandomAccess<FloatType> ra = dy.randomAccess();
		for (int x=0; x<width; x++)
			for (int y=0; y<height; y++) {
				ra.setPosition(x, 0);
				ra.setPosition(y, 1);
				processor.setf(x, y, ra.get().get());
			}
		return processor;	
	}

	private RandomAccessibleInterval<FloatType> gauss(Img<FloatType> bridgeImg, double[] sigmas) {
		RandomAccessibleInterval<FloatType> blurred = ops.create().img(bridgeImg);
		ops.filter().gauss(blurred, bridgeImg, sigmas, new OutOfBoundsBorderFactory<>());
		return blurred;
	}

	private static final float[] SOBEL_FILTER_X_VALUES = {1f,2f,1f,0f,0f,0f,-1f,-2f,-1f};
	private static final float[] SOBEL_FILTER_Y_VALUES = {1f,0f,-1f,2f,0f,-2f,1f,0f,-1f};

	RandomAccessibleInterval<FloatType> SOBEL_FILTER_X = ArrayImgs.floats(SOBEL_FILTER_X_VALUES, 3, 3);
	RandomAccessibleInterval<FloatType> SOBEL_FILTER_Y = ArrayImgs.floats(SOBEL_FILTER_Y_VALUES, 3, 3);

	private RandomAccessibleInterval<FloatType> deriveX(RandomAccessibleInterval<FloatType> in) {
		return convolve(in, SOBEL_FILTER_X);
	}

	private RandomAccessibleInterval<FloatType> deriveY(RandomAccessibleInterval<FloatType> in) {
		return convolve(in, SOBEL_FILTER_Y);
	}

	private RandomAccessibleInterval<FloatType> convolve(RandomAccessibleInterval<FloatType> blurred, RandomAccessibleInterval<FloatType> kernel) {
		return ops.filter().convolve(blurred, kernel, new OutOfBoundsBorderFactory<>());
	}

	static Interval extendInterval(Interval image) {
		int numDimensions = image.numDimensions();

		long[] min = new long[numDimensions];
		long[] max = new long[numDimensions];

		for (int d = 0; d < numDimensions; ++d )
		{
			min[ d ] = -image.min( d ) - 90 ;
			max[ d ] = image.max( d ) + 90;
		}

		return new FinalInterval( min, max );
	}

	private void viewDifference(RandomAccessibleInterval<FloatType> expected, RandomAccessibleInterval<FloatType> result) {
		RandomAccessible<FloatType> error = Converters.convert(Views.pair(expected, result), (in, out) -> out.set(in.getA().get() - in.getB().get()), new FloatType());
		view(Views.interval(error, expected));
	}

	private void view(IntervalView<FloatType> interval) {
		ImageJ ij = new ImageJ();
		ij.ui().showUI();
		ij.ui().show(interval);
	}

	@Test
	public void testLegacyDefaultFeatureGeneration() {
		RandomAccessibleInterval<FloatType> expectedImage = Utils.loadImageFloatType("features-expected.tiff");
		boolean[] enabledFeatures = {true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false};
		RandomAccessibleInterval<FloatType> features = generateFeatureStack(bridgeImage, enabledFeatures);
		Utils.assertImagesEqual(expectedImage, features);
	}

	private static RandomAccessibleInterval<FloatType> generateSingleFeature(ImagePlus image, int feature) {
		boolean[] enabledFeatures = nCopiesAsArray(20, false);
		enabledFeatures[feature] = true;
		return generateFeatureStack(image, enabledFeatures);
	}

	private static RandomAccessibleInterval<FloatType> generateFeatureStack(ImagePlus image, boolean[] enabledFeatures) {
		final FeatureStack sliceFeatures = new FeatureStack(image);
		sliceFeatures.setEnabledFeatures(enabledFeatures);
		sliceFeatures.setMaximumSigma(16);
		sliceFeatures.setMinimumSigma(1);
		sliceFeatures.setMembranePatchSize(19);
		sliceFeatures.setMembraneSize(1);
		sliceFeatures.updateFeaturesST();
		return sliceFeatures.asRandomAccessibleInterval();
	}

	public static boolean[] nCopiesAsArray(int count, boolean value) {
		boolean[] array = new boolean[count];
		Arrays.fill(array, value);
		return array;
	}
}