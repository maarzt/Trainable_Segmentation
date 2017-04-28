package trainableSegmentation;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeNotNull;
import hr.irb.fastRandomForest.FastRandomForest;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.process.ImageConverter;

import java.net.URL;
import java.util.function.Consumer;

import org.junit.Test;

public class BasicTest
{
	@Test
	public void test1()
	{
		final ImagePlus image = Utils.createImage( "test", 2, 2, 17, 2, 123, 54 );
		final ImagePlus labels = Utils.createImage( "labels", 2, 2, 255, 255, 0, 0 );

		WekaSegmentation segmentator = new WekaSegmentation( image );

		if ( false == segmentator.addBinaryData(image, labels, "class 2", "class 1") )
			assertTrue("Error while adding binary data to segmentator", false);


		assertTrue("Failed to train classifier", true == segmentator.trainClassifier());

		segmentator.applyClassifier( false );

		ImagePlus result = segmentator.getClassifiedImage();


		assertTrue("Failed to apply trained classifier", null != result);

		float[] pix = (float[]) result.getProcessor().getPixels();
		byte[] pixTrue = (byte[]) labels.getProcessor().getPixels();
		for( int i=0; i<pix.length; i++)
		{
			assertTrue("Misclassified training sample", pix[i] * 255 == (pixTrue[i]&0xff) );
		}
	}

	@Test
	public void bridge() {
		final ImagePlus bridge = Utils.loadImagePlusFromResource( "bridge.png" );
		assumeNotNull( bridge );
		final ImagePlus bridgeExpect = Utils.loadImagePlusFromResource( "bridge-expected.png" );
		assumeNotNull( bridgeExpect );

		ImagePlus output = segmentBridge(bridge);
		Utils.assertImagesEqual(bridgeExpect, output);
	}

	@Test
	public void testDefaultFeatureGenerationST() {
		testDefaultFeaturesOnBridge(FeatureStack::updateFeaturesST);
	}

	@Test
	public void testDefaultFeatureGenerationMT() {
		testDefaultFeaturesOnBridge(FeatureStack::updateFeaturesMT);
	}

	private void testDefaultFeaturesOnBridge(Consumer<FeatureStack> updateFeaturesMethod) {
		// setup
		final ImagePlus bridge = Utils.loadImagePlusFromResource("/bridge.png");
		// process
		final FeatureStack featureStack = new FeatureStack(bridge);
		updateFeaturesMethod.accept(featureStack);
		final ImagePlus features = new ImagePlus("features", featureStack.getStack());
		// test
		final ImagePlus expected = Utils.loadImagePlusFromResource("/features-expected.tiff");
		Utils.assertImagesEqual(expected, features);
	}

	private static ImagePlus segmentBridge(final ImagePlus bridge) {
		WekaSegmentation segmentator = new WekaSegmentation( bridge );
		segmentator.addExample( 0, new Roi( 10, 10, 50, 50 ), 1 );
		segmentator.addExample( 1, new Roi( 400, 400, 30, 30 ), 1 );

		FastRandomForest rf = (FastRandomForest) segmentator.getClassifier();
		rf.setSeed( 69 );
		assertTrue( segmentator.trainClassifier() );

		segmentator.applyClassifier( false );
		ImagePlus output = segmentator.getClassifiedImage();
		new ImageConverter( output ).convertToGray8();
		return output;
	}

	public static void main(String...strings) {
		final String pathOfClass = "/" + BasicTest.class.getName().replace('.', '/') + ".class";
		final URL url = BasicTest.class.getResource(pathOfClass);
		if ( !"file".equals( url.getProtocol() ) ) throw new RuntimeException( "Need to run from test-classes/" );
		final String suffix = "/target/test-classes" + pathOfClass;
		final String path = url.getPath();
		if ( !path.endsWith( suffix ) ) throw new RuntimeException( "Unexpected class location: " + path );
		final String resources = path.substring( 0, path.length() - suffix.length() ) + "/src/test/resources/";
		final ImagePlus bridge = new ImagePlus( "http://imagej.nih.gov/ij/images/bridge.gif" );
		IJ.save( bridge, resources + "bridge.png" );
		final ImagePlus bridgeExpected = segmentBridge( bridge );
		IJ.save( bridgeExpected, resources + "bridge-expected.png" );
	}

}