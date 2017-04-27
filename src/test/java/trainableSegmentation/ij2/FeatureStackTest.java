package trainableSegmentation.ij2;

import ij.ImagePlus;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.img.ImagePlusAdapter;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
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
		new FeatureStackTest().testHessianStack();
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

	@Test
	public void testHessianStack() {
		RandomAccessibleInterval<FloatType> expected = generateSingleFeature(bridgeImage, FeatureStack.HESSIAN);
		RandomAccessibleInterval<FloatType> result = createGaussStack(bridgeImg);
		viewDifference(expected, result);
		ImageJFunctions.show(expected);
		ImageJFunctions.show(result);
		float psnr = Utils.psnr(expected, result);
		System.out.print(psnr);
		assertTrue(psnr > 40);
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

	private RandomAccessibleInterval<FloatType> createGaussStack(Img<FloatType> image) {
		List<RandomAccessibleInterval<FloatType>> features = new ArrayList<>();
		features.add(image);
		final double minimumSigma = 1;
		final double maximumSigma = 16;
		for(double sigma = minimumSigma; sigma <= maximumSigma; sigma *= 2)
			features.add(ops.filter().gauss(image, sigma*0.4));
		return Views.stack(features);
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