package trainableSegmentation.ij2;

import ij.ImagePlus;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.gauss.Gauss;
import net.imglib2.img.ImagePlusAdapter;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.junit.Test;
import trainableSegmentation.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Matthias Arzt
 */
public class FeatureStackTest {

	private static ImagePlus bridgeImage = Utils.loadImage("bridge.png");

	private static Img<FloatType> bridgeImg = ImagePlusAdapter.convertFloat(bridgeImage);

	public static void main(String... args) {
		RandomAccessibleInterval<FloatType> gaussFeatures = generateSingleFeature(bridgeImage, FeatureStack.GAUSSIAN);
		ImageJFunctions.show(gaussFeatures);
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
		Utils.assertImagesEqual(expected, result);
	}

	private RandomAccessibleInterval<FloatType> createGaussStack(Img<FloatType> image) {
		List<RandomAccessibleInterval<FloatType>> features = new ArrayList<>();
		features.add(image);
		final int minimumSigma = 1;
		final int maximumSigma = 16;
		for(int sigma = minimumSigma; sigma <= maximumSigma; sigma *= 2)
			features.add(Gauss.toFloat(sigma, image));
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