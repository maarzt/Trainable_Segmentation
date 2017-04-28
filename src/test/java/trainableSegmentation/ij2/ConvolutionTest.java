package trainableSegmentation.ij2;

import ij.ImagePlus;
import ij.plugin.filter.Convolver;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.ImagePlusAdapter;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.outofbounds.OutOfBoundsBorderFactory;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.junit.Test;
import org.scijava.Context;
import trainableSegmentation.Utils;

import java.util.Arrays;
import java.util.StringJoiner;

/**
 * Created by arzt on 28.04.17.
 */
public class ConvolutionTest {

	private static OpService ops = new Context(OpService.class).getService(OpService.class);

	private static ImagePlus bridgeImage = toGray(Utils.loadImage("bridge.png"));

	private static ImagePlus toGray(ImagePlus image) {
		return new ImagePlus(image.getTitle(), image.getProcessor().convertToFloat());
	}

	@Test
	public void test() {
		float[] pixels = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
		int width = 4;
		int height = 4;
		ImagePlus imagePlus = bridgeImage;
		float[] sobelFilter_x = {1f,2f,1f,0f,0f,0f,-1f,-2f,-1f};
		float[] kernel = sobelFilter_x;
		ImagePlus resultPlus = imagePlus.duplicate();
		int kernelWidth = 3;
		int kernelHeight = 3;
		new Convolver().convolve(resultPlus.getProcessor(), kernel, kernelWidth, kernelHeight);
		toString(resultPlus);

		Img<FloatType> image = ImagePlusAdapter.convertFloat(imagePlus);

		RandomAccessibleInterval<FloatType> kernelAsImg = ArrayImgs.floats(kernel, kernelWidth, kernelHeight);
		RandomAccessibleInterval<FloatType> result = ops.filter().convolve(image, kernelAsImg, new OutOfBoundsBorderFactory<>());
		Utils.assertImagesEqual(resultPlus, result);
	}



	private String toString(RandomAccessibleInterval<FloatType> s) {
		StringJoiner joiner = new StringJoiner(", ");
		for(FloatType voxel : Views.iterable(s))
			joiner.add(voxel.toString());
		return joiner.toString();
	}

	private void toString(ImagePlus image) {
		System.out.println(Arrays.deepToString(image.getProcessor().getFloatArray()));
	}

}
