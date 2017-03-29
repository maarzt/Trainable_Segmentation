package trainableSegmentation;

import ij.ImagePlus;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.img.ImagePlusAdapter;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.roi.labeling.ImgLabeling;
import net.imglib2.roi.labeling.LabelingType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.convert.ConvertService;
import trainableSegmentation.ij2.Classifier;

/**
 * @author Matthias Arzt
 */
public class WekaSegmentationImageJ2Test {

	static public ImgLabeling<String, IntType> getBridgeLabeling() {
		final ImgLabeling<String, IntType> labeling = new ImgLabeling<String, IntType>(ArrayImgs.ints(512, 512));
		RandomAccess<LabelingType<String>> ra = labeling.randomAccess();
		Views.interval(labeling, new long[]{10, 10}, new long[]{59, 59}).forEach( x -> x.add("1") );
		Views.interval(labeling, new long[]{400, 400}, new long[]{429, 429}).forEach( x -> x.add("2") );
		return labeling;
	}

	public ImagePlus convert(Img<IntType> img) {
		ConvertService cs = new Context(ConvertService.class).getService(ConvertService.class);
		return cs.convert(img, ImagePlus.class);
	}

	public void testClassification() {
		ImagePlus image = loadImage("bridge.png");
		ImgLabeling<String, IntType> labeling = getBridgeLabeling();
		Classifier classifier = Classifier.train(image, labeling);
		Img<IntType> resultImage = classifier.apply(image);
		RandomAccessibleInterval<UnsignedByteType> expectedImage = ImagePlusAdapter.wrapByte(loadImage("bridge-expected.png"));
		RandomAccessibleInterval<IntType> eI = Converters.convert(expectedImage, (b, i) -> i.set(b.get()), new IntType());
		Utils.assertImagesEqual(eI, resultImage);
	}

	public static void main(String... args) {
		new WekaSegmentationImageJ2Test().testClassification();
	}

	private static ImagePlus loadImage(String s) {
		return Utils.loadImagePlusFromResource(s);
	}


}
