package trainableSegmentation.ij2;

import ij.ImagePlus;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import net.imagej.ImageJ;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.img.ImagePlusAdapter;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.junit.Ignore;
import org.junit.Test;
import trainableSegmentation.Utils;

import java.util.Arrays;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * @author Matthias Arzt
 */
public class FeatureStackTest {

	private static ImagePlus bridgeImage = Utils.loadImage("nuclei.tif");

	private static Img<FloatType> bridgeImg = ImagePlusAdapter.convertFloat(bridgeImage);

	public static void main(String... args) {
		new FeatureStackTest().testLegacyLipschitz();
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
		RandomAccessibleInterval<FloatType> result = FeatureStack2.createGaussStack(bridgeImg);
		assertImagesEqual(40, expected, result);
	}

	private void assertImagesEqual(float expectedPsnr, RandomAccessibleInterval<FloatType> expected, RandomAccessibleInterval<FloatType> result) {
		float psnr = Utils.psnr(expected, result);
		if(psnr < expectedPsnr)
			fail("Actual PSNR is lower than expected. Actual: " + psnr + " Expected: " + expectedPsnr);
	}

	@Test
	public void testHessianStack() {
		RandomAccessibleInterval<FloatType> expected = generateSingleFeature(bridgeImage, FeatureStack.HESSIAN);
		RandomAccessibleInterval<FloatType> result = FeatureStack2.createHessianStack(bridgeImg);
		assertImagesEqual(40, expected, result);
	}

	@Test
	public void testCalculateHessian() {
		RandomAccessibleInterval<FloatType> expected = ImagePlusAdapter.wrapFloat(FeatureStack.calculateHessianOnChannel(bridgeImage, 8));
		RandomAccessibleInterval<FloatType>	actual = FeatureStack2.calculateHessianOnChannel(bridgeImg, 8);
		assertImagesEqual(40, expected, actual);
	}

	@Test
	public void testDifferenceOfGaussian() {
		RandomAccessibleInterval<FloatType> expected = generateSingleFeature(bridgeImage, FeatureStack.DOG);
		RandomAccessibleInterval<FloatType> result = FeatureStack2.createDifferenceOfGaussiansStack(bridgeImg);
		assertImagesEqual(40, expected, result);
	}

	@Test
	public void testSingleDifferenceOfGaussian() {
		RandomAccessibleInterval<FloatType> expected = ImagePlusAdapter.wrapFloat(FeatureStack.calculateDoG(bridgeImage, 8, 4));
		RandomAccessibleInterval<FloatType> result = FeatureStack2.calculateDifferenceOfGaussians(bridgeImg, 8, 4);
		assertImagesEqual(30, expected, result);
	}

	@Test
	public void testSobel() {
		RandomAccessibleInterval<FloatType> expected = generateSingleFeature(bridgeImage, FeatureStack.SOBEL);
		RandomAccessibleInterval<FloatType> result = FeatureStack2.createSobelStack(bridgeImg);
		assertImagesEqual(40, expected, result);
	}

	@Ignore
	@Test
	public void testLipschitz() {
		RandomAccessibleInterval<FloatType> expected = generateSingleFeature(bridgeImage, FeatureStack.LIPSCHITZ);
		RandomAccessibleInterval<FloatType> result = FeatureStack2.createLipschitzStack(bridgeImg);
		assertImagesEqual(40, expected, result);
	}

	@Test
	public void testLegacyLipschitz() {
		RandomAccessibleInterval<FloatType> expected = Utils.loadImageFloatType("nuclei-lipschitz-feature.tif");
		RandomAccessibleInterval<FloatType> result = generateSingleFeature(bridgeImage, FeatureStack.LIPSCHITZ);
		assertImagesEqual(40, expected, result);
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
		RandomAccessibleInterval<FloatType> expectedImage = Utils.loadImageFloatType("nuclei-features.tif");
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